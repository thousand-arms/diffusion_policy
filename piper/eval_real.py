"""Real-time continuous policy evaluation on the Piper + INDEMIND setup.

UMI-style synchronous architecture:
  - Main thread: get_obs -> predict -> exec_actions (non-blocking) -> precise_wait
  - PiperDriver's interpolation thread at ~150Hz reads pose_interp(t_now)
    and streams EndPoseCtrl (MOVE P). Trim-and-insert on new waypoints.

Usage:
    python -m piper.eval_real -c /path/to/checkpoint.ckpt
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import click
import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffusion_policy.common.precise_sleep import precise_wait
from piper.common.checkpoint_util import load_policy
from piper.common.viz import plot_trajectory
from piper.env.piper_driver import PiperDriver
from piper.env.piper_env import PiperEnv, CAMERA_OBS_LATENCY_S, ROBOT_ACTION_LATENCY_S


_OBS_NON_TENSOR_KEYS = ("timestamp", "anchor_mat")

DEFAULT_POSE_PATH = ROOT / "piper" / "env" / "default_cam_pose.yaml"


def _load_default_pose_mat(path: pathlib.Path) -> np.ndarray | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open() as f:
        data = yaml.safe_load(f)
    if not data:
        return None
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(data["rot_axis_angle_rad"]).as_matrix()
    T[:3, 3] = data["pos_m"]
    return T


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


def _trigger_reset(
    driver: PiperDriver,
    default_pose_mat: np.ndarray,
    reset_speed_m_s: float,
) -> None:
    """Schedule a waypoint to the default camera pose at reset_speed_m_s.

    Uses a near-future target_time to force the interpolator to trim any
    in-flight policy plan, and a per-call max_pos_speed override so the
    motion runs at reset_speed_m_s regardless of the driver's usual cap.
    """
    curr_pos = driver.get_camera_pose()[0]
    dist_m = float(np.linalg.norm(default_pose_mat[:3, 3] - curr_pos))
    driver.schedule_waypoint(
        default_pose_mat,
        time.monotonic() + 0.05,
        max_pos_speed=reset_speed_m_s,
    )
    expected_s = dist_m / reset_speed_m_s
    print(f"[reset] moving {dist_m*1000:.0f}mm to default pose "
          f"over ~{expected_s:.1f}s at {reset_speed_m_s*1000:.0f}mm/s")


def _preview_loop(
    env: PiperEnv,
    driver: PiperDriver,
    window: str,
    default_pose_mat: np.ndarray | None,
    reset_speed_m_s: float = 0.08,
) -> str:
    """Idle command loop. Returns 'start' or 'quit'."""
    print("[ready] (s)tart policy | (r)eset to default pose | (q)uit")
    if default_pose_mat is None:
        print("[ready] note: no default pose saved — 'r' will be ignored. "
              "Run `python -m piper.scripts.set_default_camera_pose` first.")
    while True:
        preview = env.get_preview()
        if preview is not None:
            left, right = preview
            cv2.imshow(window, np.concatenate([left, right], axis=1))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            return "start"
        if key == ord("r"):
            if default_pose_mat is None:
                print("[reset] skipped — no default pose configured.")
            else:
                _trigger_reset(driver, default_pose_mat, reset_speed_m_s)
        if key in (ord("q"), 27):
            return "quit"


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
@click.option("--default-pose", default=str(DEFAULT_POSE_PATH), type=click.Path(),
              help="yaml with pos_m + rot_axis_angle_rad for the 'r' reset pose")
@click.option("--reset-speed-m-s", default=0.08, type=float,
              help="linear speed for the 'r' reset motion (m/s)")
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
    default_pose,
    reset_speed_m_s,
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

        default_pose_mat = _load_default_pose_mat(pathlib.Path(default_pose))
        if default_pose_mat is None:
            print(f"[default-pose] none loaded from {default_pose}")
        else:
            print(f"[default-pose] loaded from {default_pose}")

        window = "eval_real"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        reset_enabled = (not dry_run) and default_pose_mat is not None
        reset_pose = default_pose_mat if reset_enabled else None

        while True:
            action = _preview_loop(
                env, driver, window, reset_pose,
                reset_speed_m_s=reset_speed_m_s,
            )
            if action == "quit":
                break
            session_t_start, exit_reason = _run_policy_loop(
                env, driver, policy, device,
                dt=dt,
                horizon=horizon,
                steps_per_inference=steps_per_inference,
                max_duration=max_duration,
                action_exec_latency=action_exec_latency,
                dry_run=dry_run,
                window=window,
            )
            if t_start is None:
                t_start = session_t_start
            if exit_reason == "reset":
                if reset_pose is not None:
                    _trigger_reset(driver, reset_pose, reset_speed_m_s)
                continue
            break  # quit or timeout
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
):
    """Synchronous inference loop. Scheduling is non-blocking.

    Returns (t_start, exit_reason) where exit_reason is one of
    'reset' | 'quit' | 'timeout'.
    """
    t_start = time.monotonic()
    iter_idx = 0
    cycle_idx = 0
    total_sent = 0
    exit_reason = "timeout"

    print("[running] policy control active. (r)eset, (q)uit.")

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
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            exit_reason = "quit"
            break
        if key == ord("r"):
            exit_reason = "reset"
            print("[pause] inference paused — resetting to default pose.")
            break

        # Grab fresh obs at next cycle rather than the exact boundary
        frame_latency = 1.0 / 60.0
        precise_wait(t_cycle_end - frame_latency)
        iter_idx += steps_per_inference
        cycle_idx += 1

    elapsed = time.monotonic() - t_start
    print(f"[done] {cycle_idx} cycles, {total_sent} actions sent in {elapsed:.1f}s "
          f"({exit_reason}).")
    return t_start, exit_reason


if __name__ == "__main__":
    main()
