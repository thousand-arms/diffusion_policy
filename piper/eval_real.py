"""Real-time continuous policy evaluation on the Piper + INDEMIND setup.

UMI-style synchronous architecture:
  - Main thread: get_obs -> predict -> exec_actions (non-blocking) -> precise_wait
  - PiperDriver's interpolation thread at ~150Hz reads pose_interp(t_now)
    and streams EndPoseCtrl (MOVE P). Trim-and-insert on new waypoints.

Usage:
    python -m piper.eval_real -c /path/to/checkpoint.ckpt
"""

import os
import pathlib
import sys
import time

import click
import cv2
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffusion_policy.common.precise_sleep import precise_wait
from piper.common.checkpoint_util import load_policy
from piper.common.viz import plot_trajectory
from piper.env.piper_driver import PiperDriver
from piper.env.piper_env import PiperEnv, CAMERA_OBS_LATENCY_S, ROBOT_ACTION_LATENCY_S


_OBS_NON_TENSOR_KEYS = ("timestamp", "anchor_mat")


def _obs_to_torch(obs: dict, device: torch.device) -> dict:
    return {
        k: torch.from_numpy(v).unsqueeze(0).to(device)
        for k, v in obs.items()
        if k not in _OBS_NON_TENSOR_KEYS
    }


def _wait_for_obs(env: PiperEnv, timeout_s: float = 5.0):
    t0 = time.time()
    while env.get_obs() is None:
        if time.time() - t0 > timeout_s:
            raise RuntimeError("Timed out waiting for observations.")
        time.sleep(0.05)


def _preview_loop(env: PiperEnv, window: str) -> bool:
    """Show live preview; return True if user pressed 'c', False on 'q'."""
    print("[ready] press 'c' to start policy control, 'q' to quit.")
    while True:
        preview = env.get_preview()
        if preview is not None:
            left, right = preview
            cv2.imshow(window, np.concatenate([left, right], axis=1))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("c"):
            return True
        if key in (ord("q"), 27):
            return False


def _select_new_actions(
    actions: np.ndarray,
    horizon: int,
    dt: float,
    obs_time: float,
    t_start: float,
    action_exec_latency: float,
):
    """Filter to actions whose target time is still in the future.

    Returns (actions, timestamps). If nothing is in the future, schedules
    just the last action on the next dt boundary as a fallback.
    """
    timestamps = np.arange(horizon, dtype=np.float64) * dt + obs_time
    now = time.monotonic()
    keep = timestamps > (now + action_exec_latency)

    if keep.any():
        return actions[keep], timestamps[keep]

    # Over budget — fallback to last action on next dt
    next_step = int(np.ceil((now - t_start) / dt))
    fallback_t = t_start + next_step * dt
    print(f"[over-budget] using last action at fallback t=+{fallback_t - now:.3f}s",
          flush=True)
    return actions[[-1]], np.array([fallback_t])


def _draw_preview(env: PiperEnv, window: str, text: str):
    preview = env.get_preview()
    if preview is None:
        return
    left, right = preview
    frame = np.concatenate([left, right], axis=1)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)
    cv2.imshow(window, frame)


@click.command()
@click.option("--checkpoint", "-c", required=True, type=click.Path(exists=True))
@click.option("--can-port", default="can0")
@click.option("--speed-pct", default=100, help="Piper MOVE P speed percentage")
@click.option("--dt", default=0.06, type=float, help="control dt (down_sample_steps / camera_hz)")
@click.option("--steps-per-inference", default=3, type=int,
              help="main loop advances this many dt per inference cycle")
@click.option("--max-duration", default=120.0, type=float)
@click.option("--max-step-m", default=0.05, type=float, help="safety: max step delta (m)")
@click.option("--smooth-hz", default=150, type=int, help="interpolation controller rate (Hz)")
@click.option("--max-pos-speed", default=0.5, type=float, help="interp max linear speed (m/s)")
@click.option("--max-rot-speed", default=2.0, type=float, help="interp max angular speed (rad/s)")
@click.option("--action-exec-latency", default=0.01, type=float,
              help="only keep actions with target_time > now + this (s)")
@click.option("--dry-run", is_flag=True, help="print targets without sending to robot")
@click.option("--verbose-interp", is_flag=True, help="print interp thread stats")
@click.option("--visualize", is_flag=True,
              help="record scheduled waypoints + sent poses and plot after stop")
def main(
    checkpoint,
    can_port,
    speed_pct,
    dt,
    steps_per_inference,
    max_duration,
    max_step_m,
    smooth_hz,
    max_pos_speed,
    max_rot_speed,
    action_exec_latency,
    dry_run,
    verbose_interp,
    visualize,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[policy] loading {checkpoint}")
    policy, cfg = load_policy(checkpoint, device)
    n_obs_steps = cfg.n_obs_steps
    horizon = cfg.horizon
    print(f"[policy] n_obs_steps={n_obs_steps}  horizon={horizon}  device={device}")
    print(
        f"[config] dt={dt*1000:.0f}ms  steps_per_inference={steps_per_inference}  "
        f"cycle={steps_per_inference*dt*1000:.0f}ms  "
        f"cam_lat={CAMERA_OBS_LATENCY_S*1000:.0f}ms  "
        f"robot_lat={ROBOT_ACTION_LATENCY_S*1000:.0f}ms"
    )
    print(f"[interp] send_hz={smooth_hz}  max_pos={max_pos_speed}m/s  "
          f"max_rot={max_rot_speed}rad/s")

    print("[robot] connecting Piper...")
    driver = PiperDriver(
        can_port=can_port,
        speed_pct=speed_pct,
        smooth=not dry_run,
        smooth_hz=smooth_hz,
        max_pos_speed=max_pos_speed,
        max_rot_speed=max_rot_speed,
        verbose_interp=verbose_interp,
        record_interp=visualize,
    )

    print("[env] starting...")
    env = PiperEnv(driver, n_obs_steps=n_obs_steps, dt=dt, max_step_m=max_step_m)
    env.start()
    t_start = None
    try:
        _wait_for_obs(env)

        print("[warmup] running first inference...")
        with torch.no_grad():
            policy.predict_action(_obs_to_torch(env.get_obs(), device))
        print("[warmup] done.")

        window = "eval_real"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        if not _preview_loop(env, window):
            return

        t_start = _run_policy_loop(
            env, driver, policy, device,
            dt=dt,
            horizon=horizon,
            steps_per_inference=steps_per_inference,
            max_duration=max_duration,
            action_exec_latency=action_exec_latency,
            dry_run=dry_run,
            window=window,
        )
    finally:
        if visualize and t_start is not None:
            sent_log, waypoint_log = driver.get_recorded_logs()
        env.stop()
        cv2.destroyAllWindows()

        if visualize and t_start is not None:
            out_dir = os.path.join(str(ROOT), "tmp", "eval_viz")
            plot_trajectory(sent_log, waypoint_log, t_start=t_start, out_root=out_dir)


def _run_policy_loop(
    env, driver, policy, device, *,
    dt, horizon, steps_per_inference, max_duration, action_exec_latency,
    dry_run, window,
) -> float:
    """Synchronous inference loop. Scheduling is non-blocking.

    Returns the monotonic t_start for post-run visualization.
    """
    t_start = time.monotonic()
    iter_idx = 0
    cycle_idx = 0
    total_sent = 0

    print("[running] policy control active. press 'q' to stop.")

    while True:
        t_cycle_start = time.monotonic()
        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

        elapsed = t_cycle_start - t_start
        if elapsed > max_duration:
            print(f"[done] max duration {max_duration}s reached.")
            break

        # --- get obs + infer ---
        obs = env.get_obs()
        obs_time = obs["timestamp"][-1]
        anchor = obs["anchor_mat"]
        obs_lat_ms = (time.monotonic() - obs_time) * 1000

        t_infer = time.monotonic()
        with torch.no_grad():
            result = policy.predict_action(_obs_to_torch(obs, device))
        actions = result["action_pred"][0].detach().cpu().numpy()
        infer_ms = (time.monotonic() - t_infer) * 1000

        # --- schedule future actions ---
        new_actions, new_timestamps = _select_new_actions(
            actions, horizon, dt, obs_time, t_start, action_exec_latency
        )
        n_scheduled = (len(new_actions) if dry_run
                       else env.exec_actions(new_actions, new_timestamps, anchor))
        total_sent += n_scheduled

        # --- log ---
        now = time.monotonic()
        cycle_ms = (now - t_cycle_start) * 1000
        first_rel = (new_timestamps[0] - now) * 1000
        last_rel = (new_timestamps[-1] - now) * 1000
        status = driver.get_arm_status_code()
        status_str = f"  ARM_STATUS={status:#04x}" if status != 0 else ""
        print(
            f"[cycle {cycle_idx:3d}] infer={infer_ms:.0f}ms  "
            f"obs_lat={obs_lat_ms:.0f}ms  "
            f"scheduled={n_scheduled}/{len(actions)}  "
            f"t[+{first_rel:.0f}..+{last_rel:.0f}]ms  "
            f"cycle={cycle_ms:.0f}ms{status_str}",
            flush=True,
        )

        _draw_preview(env, window,
                      f"cycle={cycle_idx} sent={total_sent} elapsed={elapsed:.1f}s")
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

        # Grab fresh obs at next cycle rather than the exact boundary
        frame_latency = 1.0 / 60.0
        precise_wait(t_cycle_end - frame_latency)
        iter_idx += steps_per_inference
        cycle_idx += 1

    elapsed = time.monotonic() - t_start
    print(f"[done] {cycle_idx} cycles, {total_sent} actions sent in {elapsed:.1f}s.")
    return t_start


if __name__ == "__main__":
    main()
