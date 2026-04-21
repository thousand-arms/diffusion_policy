"""Real-time continuous policy evaluation on the Piper + INDEMIND setup.

Simple synchronous loop: infer → execute 3 actions → repeat.
All actions per cycle share the same anchor, avoiding inter-prediction jitter.

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
from piper.common.pose_util import rel_action_to_world
from piper.env.piper_driver import PiperDriver
from piper.env.piper_env import PiperEnv


@click.command()
@click.option("--checkpoint", "-c", required=True, type=click.Path(exists=True))
@click.option("--can-port", default="can0")
@click.option("--speed-pct", default=100, help="Piper MOVE P speed percentage")
@click.option("--dt", default=0.06, type=float, help="control dt (down_sample_steps / camera_hz)")
@click.option("--n-exec-steps", default=3, type=int, help="actions to execute per inference cycle")
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
    from piper.env.piper_env import CAMERA_OBS_LATENCY_S, ROBOT_ACTION_LATENCY_S
    print(f"[config] dt={dt*1000:.0f}ms  n_exec_steps={n_exec_steps}  "
          f"cam_lat={CAMERA_OBS_LATENCY_S*1000:.0f}ms  "
          f"robot_lat={ROBOT_ACTION_LATENCY_S*1000:.0f}ms")

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

    # ---- Preview loop: show stereo, wait for 'c' to start ----
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

    # ---- Policy control: fixed-cycle loop (UMI pattern) ----
    # Each cycle covers steps_per_inference * dt of wall time.
    # Within a cycle: infer → execute actions → wait for cycle end.
    steps_per_inference = n_exec_steps
    cycle_dt = steps_per_inference * dt
    robot_lat = env.robot_action_latency

    print(f"[running] policy control active. press 'q' to stop.")
    print(f"[timing] cycle={cycle_dt*1000:.0f}ms  "
          f"steps_per_inference={steps_per_inference}  robot_lat={robot_lat*1000:.0f}ms")

    t_start = time.monotonic()
    iter_idx = 0
    cycle = 0
    total_sent = 0

    try:
        while True:
            t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

            elapsed = time.monotonic() - t_start
            if elapsed > max_duration:
                print(f"[done] max duration {max_duration}s reached.")
                break

            # 1. Get observation
            obs = env.get_obs()
            if obs is None:
                time.sleep(0.01)
                continue
            anchor = obs["anchor_mat"]
            obs_time = obs["timestamp"][-1]

            # 2. Run inference
            obs_torch = {
                k: torch.from_numpy(v).unsqueeze(0).to(device)
                for k, v in obs.items()
                if k not in ("timestamp", "anchor_mat")
            }
            t_infer = time.monotonic()
            with torch.no_grad():
                result = policy.predict_action(obs_torch)
            actions = result["action_pred"][0].detach().cpu().numpy()  # (H, 9)
            infer_ms = (time.monotonic() - t_infer) * 1000

            # 3. Select actions to execute this cycle
            # Action timestamps anchored to obs capture time (latency-corrected)
            action_timestamps = np.arange(horizon, dtype=np.float64) * dt + obs_time
            now = time.monotonic()

            # Actions must be: after now + robot_latency AND before cycle end
            valid = (action_timestamps > now + robot_lat) & \
                    (action_timestamps <= t_cycle_end + robot_lat)
            valid_indices = np.where(valid)[0]

            if len(valid_indices) == 0:
                # Fallback: use first non-stale actions
                first_valid = int(np.searchsorted(action_timestamps, now + robot_lat))
                first_valid = max(first_valid, 2)
                valid_indices = np.arange(
                    first_valid, min(first_valid + n_exec_steps, horizon))

            print(
                f"[cycle {cycle:3d}] infer={infer_ms:.0f}ms  "
                f"idx={valid_indices[0]}-{valid_indices[-1]}  "
                f"elapsed={elapsed:.1f}s",
                end="",
                flush=True,
            )

            # 4. Execute actions with dt spacing
            sent_this_cycle = 0
            for i, idx in enumerate(valid_indices):
                action_9d = actions[idx]
                T_target = rel_action_to_world(action_9d, anchor)
                target_pos = T_target[:3, 3]

                if dry_run:
                    pos_mm = action_9d[:3] * 1000
                    print(
                        f"\n  [{idx}] pos=({pos_mm[0]:+.1f},{pos_mm[1]:+.1f},{pos_mm[2]:+.1f})mm",
                        end="",
                    )
                else:
                    current_pos = driver.get_camera_pose_mat()[:3, 3]
                    delta_mm = np.linalg.norm(target_pos - current_pos) * 1000
                    arm_status = driver.get_arm_status().arm_status

                    if delta_mm / 1000 > max_step_m:
                        print(f"\n  [{idx}] SKIP delta={delta_mm:.1f}mm arm={arm_status}", end="")
                    else:
                        driver.set_camera_pose_mat(T_target)
                        sent_this_cycle += 1
                        print(f"\n  [{idx}] delta={delta_mm:.1f}mm arm={arm_status} SENT", end="")

                # Wait dt before sending next (except after last)
                if i < len(valid_indices) - 1:
                    precise_wait(time.monotonic() + dt)

            total_sent += sent_this_cycle
            print(flush=True)

            # 5. Check for quit (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            # Update preview
            preview = env.get_preview()
            if preview is not None:
                left, right = preview
                frame = np.concatenate([left, right], axis=1)
                cv2.putText(
                    frame,
                    f"cycle={cycle} sent={total_sent} elapsed={elapsed:.1f}s",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("eval_real", frame)

            # 6. Wait for cycle end, then advance
            precise_wait(t_cycle_end)
            iter_idx += steps_per_inference
            cycle += 1

    finally:
        env.stop()
        cv2.destroyAllWindows()
        elapsed = time.monotonic() - t_start
        print(f"[done] {cycle} cycles, {total_sent} actions sent in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
