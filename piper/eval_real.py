"""Real-time continuous policy evaluation on the Piper + INDEMIND setup.

UMI-style synchronous architecture:
  - Main thread: get_obs -> predict -> exec_actions (non-blocking queue) -> precise_wait
  - PiperDriver interpolation thread at ~150Hz: reads pose_interp(t_now),
    sends EndPoseCtrl (MOVE P). Trim-and-insert on new waypoints.

Usage (from repo root):
    python -m piper.eval_real -c /path/to/checkpoint.ckpt
"""

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
from piper.env.piper_driver import PiperDriver
from piper.env.piper_env import PiperEnv, CAMERA_OBS_LATENCY_S, ROBOT_ACTION_LATENCY_S


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
    print(
        f"[interp] send_hz={smooth_hz}  max_pos={max_pos_speed}m/s  "
        f"max_rot={max_rot_speed}rad/s"
    )

    print("[robot] connecting Piper...")
    driver = PiperDriver(
        can_port=can_port,
        speed_pct=speed_pct,
        smooth=not dry_run,
        smooth_hz=smooth_hz,
        max_pos_speed=max_pos_speed,
        max_rot_speed=max_rot_speed,
        verbose_interp=verbose_interp,
    )

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

    # ---- Policy control (UMI-style synchronous) ----
    print(f"[running] policy control active. press 'q' to stop.")

    t_start = time.monotonic()
    iter_idx = 0
    cycle_idx = 0
    total_sent = 0

    try:
        while True:
            t_cycle_start = time.monotonic()
            # Next cycle ends after steps_per_inference dts of planned motion
            t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

            elapsed = time.monotonic() - t_start
            if elapsed > max_duration:
                print(f"[done] max duration {max_duration}s reached.")
                break

            # --- get obs ---
            obs = env.get_obs()
            obs_time = obs["timestamp"][-1]
            anchor = obs["anchor_mat"]
            obs_lat_ms = (time.monotonic() - obs_time) * 1000

            obs_torch = {
                k: torch.from_numpy(v).unsqueeze(0).to(device)
                for k, v in obs.items()
                if k not in ("timestamp", "anchor_mat")
            }

            # --- inference ---
            t_infer = time.monotonic()
            with torch.no_grad():
                result = policy.predict_action(obs_torch)
            actions = result["action_pred"][0].detach().cpu().numpy()
            infer_ms = (time.monotonic() - t_infer) * 1000

            # --- schedule actions ---
            action_timestamps = np.arange(horizon, dtype=np.float64) * dt + obs_time

            now = time.monotonic()
            is_new = action_timestamps > (now + action_exec_latency)
            new_actions = actions[is_new]
            new_timestamps = action_timestamps[is_new]

            if len(new_actions) == 0:
                # Over budget: inference took longer than all horizon dts.
                # Fallback: schedule last action on next dt.
                next_step = int(np.ceil((now - t_start) / dt))
                fallback_t = t_start + next_step * dt
                new_actions = actions[[-1]]
                new_timestamps = np.array([fallback_t])
                print(
                    f"[over-budget] using last action at fallback t=+{fallback_t-now:.3f}s",
                    flush=True,
                )

            if dry_run:
                n_scheduled = len(new_actions)
            else:
                n_scheduled = env.exec_actions(
                    new_actions, new_timestamps, anchor,
                    compensate_latency=True, verbose=False,
                )
            total_sent += n_scheduled

            cycle_ms = (time.monotonic() - t_cycle_start) * 1000
            first_t_rel = new_timestamps[0] - now
            last_t_rel = new_timestamps[-1] - now
            arm_status_msg = driver.get_arm_status()
            arm_status_code = getattr(arm_status_msg, "arm_status", arm_status_msg)
            status_str = ""
            if arm_status_code != 0:
                status_str = f"  ARM_STATUS={int(arm_status_code):#04x}"
            print(
                f"[cycle {cycle_idx:3d}] infer={infer_ms:.0f}ms  "
                f"obs_lat={obs_lat_ms:.0f}ms  "
                f"scheduled={n_scheduled}/{len(actions)}  "
                f"t[+{first_t_rel*1000:.0f}..+{last_t_rel*1000:.0f}]ms  "
                f"cycle={cycle_ms:.0f}ms{status_str}",
                flush=True,
            )

            # --- preview ---
            preview = env.get_preview()
            if preview is not None:
                left, right = preview
                frame = np.concatenate([left, right], axis=1)
                cv2.putText(
                    frame,
                    f"cycle={cycle_idx} sent={total_sent} elapsed={elapsed:.1f}s",
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

            # Wait for this cycle's planned motion to roughly complete.
            # Use a small frame_latency so we grab fresh obs instead of the
            # one from the exact cycle boundary.
            frame_latency = 1.0 / 60.0
            precise_wait(t_cycle_end - frame_latency)
            iter_idx += steps_per_inference
            cycle_idx += 1

    finally:
        env.stop()
        cv2.destroyAllWindows()
        elapsed = time.monotonic() - t_start
        print(f"[done] {cycle_idx} cycles, {total_sent} actions sent in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
