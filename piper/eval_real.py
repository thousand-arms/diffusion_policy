"""Real-time continuous policy evaluation on the Piper + INDEMIND setup.

Two-thread architecture with fixed dt tick:
  - Inference thread: continuously runs get_obs → policy → stores prediction
  - Main loop: fixed dt ticks, executes actions from current prediction.
    Adopts new predictions at batch boundaries (every N actions).

Usage (from repo root):
    python -m piper.eval_real -c /path/to/checkpoint.ckpt
"""

import pathlib
import sys
import threading
import time

import click
import cv2
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffusion_policy.common.precise_sleep import precise_wait
from piper.common.checkpoint_util import load_policy
from piper.common.pose_util import rel_action_to_world
from piper.env.piper_driver import PiperDriver
from piper.env.piper_env import PiperEnv, CAMERA_OBS_LATENCY_S, ROBOT_ACTION_LATENCY_S


class _PredictionStore:
    """Thread-safe store for the latest prediction from the inference thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._actions = None
        self._timestamps = None
        self._anchor = None
        self._version = 0  # increments on each new prediction

    def put(self, actions, timestamps, anchor):
        with self._lock:
            self._actions = actions
            self._timestamps = timestamps
            self._anchor = anchor
            self._version += 1

    def get(self):
        """Returns (actions, timestamps, anchor, version)."""
        with self._lock:
            return self._actions, self._timestamps, self._anchor, self._version


def _inference_loop(env, policy, device, dt, store, stop_event):
    """Continuously run inference and store predictions."""
    horizon = None
    while not stop_event.is_set():
        obs = env.get_obs()
        if obs is None:
            time.sleep(0.01)
            continue

        obs_time = obs["timestamp"][-1]
        anchor = obs["anchor_mat"]

        obs_torch = {
            k: torch.from_numpy(v).unsqueeze(0).to(device)
            for k, v in obs.items()
            if k not in ("timestamp", "anchor_mat")
        }

        t0 = time.monotonic()
        with torch.no_grad():
            result = policy.predict_action(obs_torch)
        actions = result["action_pred"][0].detach().cpu().numpy()
        infer_ms = (time.monotonic() - t0) * 1000

        if horizon is None:
            horizon = len(actions)
        timestamps = np.arange(horizon, dtype=np.float64) * dt + obs_time

        store.put(actions, timestamps, anchor)

        obs_lat = (time.monotonic() - obs_time) * 1000
        print(
            f"[infer] {infer_ms:.0f}ms  obs_lat={obs_lat:.0f}ms",
            flush=True,
        )


@click.command()
@click.option("--checkpoint", "-c", required=True, type=click.Path(exists=True))
@click.option("--can-port", default="can0")
@click.option("--speed-pct", default=100, help="Piper MOVE P speed percentage")
@click.option("--dt", default=0.06, type=float, help="control dt (down_sample_steps / camera_hz)")
@click.option("--n-exec-steps", default=3, type=int, help="actions per batch before checking for new prediction")
@click.option("--max-duration", default=120.0, type=float, help="max seconds of policy control")
@click.option("--max-step-m", default=0.05, type=float, help="safety: max step delta (m)")
@click.option("--dry-run", is_flag=True, help="print targets without sending to robot")
def main(
    checkpoint,
    can_port,
    speed_pct,
    dt,
    n_exec_steps,
    max_duration,
    max_step_m,
    dry_run,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[policy] loading {checkpoint}")
    policy, cfg = load_policy(checkpoint, device)
    n_obs_steps = cfg.n_obs_steps
    horizon = cfg.horizon
    print(f"[policy] n_obs_steps={n_obs_steps}  horizon={horizon}  device={device}")
    print(
        f"[config] dt={dt*1000:.0f}ms  n_exec_steps={n_exec_steps}  "
        f"cam_lat={CAMERA_OBS_LATENCY_S*1000:.0f}ms  "
        f"robot_lat={ROBOT_ACTION_LATENCY_S*1000:.0f}ms"
    )

    print("[robot] connecting Piper...")
    driver = PiperDriver(can_port=can_port, speed_pct=speed_pct)

    print("[env] starting...")
    env = PiperEnv(
        driver,
        n_obs_steps=n_obs_steps,
        dt=dt,
        max_step_m=max_step_m,
    )
    env.start()

    # Wait for obs buffer to fill
    t0 = time.time()
    while env.get_obs() is None:
        if time.time() - t0 > 5.0:
            env.stop()
            raise RuntimeError("Timed out waiting for observations.")
        time.sleep(0.05)

    # Warm up inference
    print("[warmup] running first inference...")
    obs = env.get_obs()
    obs_torch = {
        k: torch.from_numpy(v).unsqueeze(0).to(device)
        for k, v in obs.items()
        if k not in ("timestamp", "anchor_mat")
    }
    with torch.no_grad():
        policy.predict_action(obs_torch)
    print("[warmup] done.")

    cv2.namedWindow("eval_real", cv2.WINDOW_AUTOSIZE)

    # ---- Preview loop ----
    print("[ready] press 'c' to start policy control, 'q' to quit.")
    while True:
        preview = env.get_preview()
        if preview is not None:
            left, right = preview
            cv2.imshow("eval_real", np.concatenate([left, right], axis=1))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("c"):
            break
        if key in (ord("q"), 27):
            env.stop()
            cv2.destroyAllWindows()
            return

    # ---- Policy control ----
    store = _PredictionStore()
    stop_event = threading.Event()
    robot_lat = env.robot_action_latency

    infer_thread = threading.Thread(
        target=_inference_loop,
        args=(env, policy, device, dt, store, stop_event),
        daemon=True,
    )
    infer_thread.start()

    # Wait for first prediction
    while store.get()[0] is None:
        time.sleep(0.01)

    print(f"[running] policy control active. press 'q' to stop.")
    print(f"[timing] dt={dt*1000:.0f}ms  batch={n_exec_steps}  robot_lat={robot_lat*1000:.0f}ms")

    t_start = time.monotonic()
    tick = 0
    total_sent = 0
    last_adopted_version = -1

    # Current batch state
    cur_actions = None
    cur_timestamps = None
    cur_anchor = None
    cur_exec_idx = 0
    cur_exec_count = 0
    batch_num = 0

    try:
        while True:
            t_next = t_start + (tick + 1) * dt

            elapsed = time.monotonic() - t_start
            if elapsed > max_duration:
                print(f"[done] max duration {max_duration}s reached.")
                break

            # Check if we need a new prediction (batch exhausted or no current plan)
            need_new = (cur_actions is None) or (cur_exec_count >= n_exec_steps)

            if need_new:
                actions, timestamps, anchor, version = store.get()
                if actions is not None and version != last_adopted_version:
                    cur_actions = actions
                    cur_timestamps = timestamps
                    cur_anchor = anchor
                    last_adopted_version = version

                    # Find first valid action: after now + robot_latency
                    now = time.monotonic()
                    first_valid = int(np.searchsorted(
                        cur_timestamps, now + robot_lat))
                    first_valid = max(first_valid, 2)
                    cur_exec_idx = first_valid
                    cur_exec_count = 0
                    batch_num += 1

            # Execute actions for this batch
            if cur_actions is not None and cur_exec_idx < horizon and cur_exec_count < n_exec_steps:
                end_idx = min(cur_exec_idx + (n_exec_steps - cur_exec_count), horizon)
                batch_actions = cur_actions[cur_exec_idx:end_idx]
                batch_timestamps = cur_timestamps[cur_exec_idx:end_idx]

                if dry_run:
                    for i, idx in enumerate(range(cur_exec_idx, cur_exec_idx + len(batch_actions))):
                        pos_mm = batch_actions[i, :3] * 1000
                        print(
                            f"[tick {tick:3d}] b={batch_num} [{idx}] "
                            f"pos=({pos_mm[0]:+.1f},{pos_mm[1]:+.1f},{pos_mm[2]:+.1f})mm",
                            flush=True,
                        )
                else:
                    n_sent = env.exec_actions(batch_actions, batch_timestamps, cur_anchor)
                    total_sent += n_sent
                    print(
                        f"[tick {tick:3d}] b={batch_num} [{cur_exec_idx}-{cur_exec_idx + n_sent - 1}] "
                        f"sent={n_sent}",
                        flush=True,
                    )

                cur_exec_idx = end_idx
                cur_exec_count = n_exec_steps  # batch done, check for new prediction next tick

            # Preview (non-blocking)
            preview = env.get_preview()
            if preview is not None:
                left, right = preview
                frame = np.concatenate([left, right], axis=1)
                cv2.putText(
                    frame,
                    f"tick={tick} sent={total_sent} batch={batch_num} elapsed={elapsed:.1f}s",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("eval_real", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            precise_wait(t_next)
            tick += 1

    finally:
        stop_event.set()
        infer_thread.join(timeout=2.0)
        env.stop()
        cv2.destroyAllWindows()
        elapsed = time.monotonic() - t_start
        print(f"[done] {tick} ticks, {total_sent} actions sent in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
