"""Keyframe-driven demo runner for the diffusion policy.

Reads piper/env/camera_poses.yaml (a default pose + a list of keyframes),
brings the arm to the default pose on start, warms up inference, and then
runs a command loop:

  s  move to next keyframe at --move-speed-m-s, then run inference until
     'r' or max_duration. If all keyframes have been visited, just
     return to default (no inference).
  r  during inference: pause inference and return to default.
     when idle:         move to default (no-op if already there).
  w  reset the keyframe index so the next 's' starts from keyframe 1.
  q  quit.

Usage:
    python -m piper.demo
    python -m piper.demo -c /path/to/checkpoint.ckpt
"""

from __future__ import annotations

import pathlib
import sys
import time

import click
import cv2
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from piper.common.checkpoint_util import load_policy
from piper.env.camera_poses import CAMERA_POSES_PATH, load_camera_poses
from piper.env.piper_driver import PiperDriver
from piper.env.piper_env import PiperEnv
from piper.eval_real import (
    _draw_preview,
    _obs_to_torch,
    _run_policy_loop,
    _wait_for_obs,
)

DEFAULT_CHECKPOINT = (
    "/mnt/thousand-arms-data/outputs/trace/t_joint/white_line/"
    "2026.04.22/17.55.42_train_diffusion_unet_image_trace_trace_image/"
    "checkpoints/epoch=0160-val_pos_mse=0.000015.ckpt"
)


def _move_to(
    env: PiperEnv,
    driver: PiperDriver,
    target_mat: np.ndarray,
    speed_m_s: float,
    window: str,
    label: str,
) -> str:
    """Schedule a waypoint to target and block until arrival (or 'q').

    Returns 'arrived' or 'quit'.
    """
    curr_pos = driver.get_camera_pose()[0]
    dist_m = float(np.linalg.norm(target_mat[:3, 3] - curr_pos))
    # Near-future target_time forces trim; per-call speed cap sets pace.
    driver.schedule_waypoint(
        target_mat, time.monotonic() + 0.05, max_pos_speed=speed_m_s,
    )
    move_s = dist_m / max(speed_m_s, 1e-6)
    print(f"[move] {label}: {dist_m*1000:.0f}mm over ~{move_s:.1f}s "
          f"at {speed_m_s*1000:.0f}mm/s")
    deadline = time.monotonic() + move_s + 0.3  # small settling buffer
    while time.monotonic() < deadline:
        _draw_preview(env, window, f"moving to {label}")
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
    return "arrived"


def _idle_loop(
    env: PiperEnv,
    window: str,
    next_idx: int,
    n_keyframes: int,
) -> str:
    """Show preview; return one of 's','r','w','q' when pressed."""
    if next_idx >= n_keyframes:
        print(f"[idle] all {n_keyframes} keyframes visited. "
              "(s)=back to default | (r)=to default | (w)=reset list | (q)=quit")
    else:
        print(f"[idle] next keyframe: {next_idx+1}/{n_keyframes}. "
              "(s)=go+infer | (r)=to default | (w)=reset list | (q)=quit")
    while True:
        label = (f"idle  next={min(next_idx+1, n_keyframes)}/{n_keyframes}"
                 if n_keyframes > 0 else "idle (no keyframes)")
        _draw_preview(env, window, label)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            return "s"
        if key == ord("r"):
            return "r"
        if key == ord("w"):
            return "w"
        if key in (ord("q"), 27):
            return "q"


@click.command()
@click.option("--checkpoint", "-c", default=DEFAULT_CHECKPOINT,
              type=click.Path(exists=True))
@click.option("--poses", default=str(CAMERA_POSES_PATH), type=click.Path(exists=True),
              help="yaml with 'default' pose + 'keyframes' list")
@click.option("--can-port", default="can0")
@click.option("--speed-pct", default=100)
@click.option("--dt", default=0.06, type=float)
@click.option("--steps-per-inference", default=3, type=int)
@click.option("--max-duration", default=30.0, type=float,
              help="max seconds to run inference at a single keyframe")
@click.option("--max-step-m", default=0.05, type=float)
@click.option("--smooth-hz", default=150, type=int)
@click.option("--max-pos-speed", default=0.5, type=float)
@click.option("--max-rot-speed", default=2.0, type=float)
@click.option("--action-exec-latency", default=0.01, type=float)
@click.option("--move-speed-m-s", default=0.08, type=float,
              help="linear speed for keyframe and default moves (m/s)")
@click.option("--verbose-interp", is_flag=True)
def main(
    checkpoint,
    poses,
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
    move_speed_m_s,
    verbose_interp,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_mat, keyframes = load_camera_poses(pathlib.Path(poses))
    if default_mat is None:
        print(f"[poses] no default pose in {poses}. "
              f"Run `python -m piper.scripts.record_camera_poses` first.")
        return
    print(f"[poses] loaded default + {len(keyframes)} keyframe(s) from {poses}")

    print(f"[policy] loading {checkpoint}")
    policy, cfg = load_policy(checkpoint, device)
    n_obs_steps = cfg.n_obs_steps
    horizon = cfg.horizon
    print(f"[policy] n_obs_steps={n_obs_steps}  horizon={horizon}  device={device}")

    print("[robot] connecting Piper...")
    driver = PiperDriver(
        can_port=can_port,
        speed_pct=speed_pct,
        smooth=True,
        smooth_hz=smooth_hz,
        max_pos_speed=max_pos_speed,
        max_rot_speed=max_rot_speed,
        verbose_interp=verbose_interp,
    )

    print("[env] starting...")
    env = PiperEnv(driver, n_obs_steps=n_obs_steps, dt=dt, max_step_m=max_step_m)
    env.start()
    try:
        _wait_for_obs(env)

        print("[warmup] running first inference...")
        with torch.no_grad():
            policy.predict_action(_obs_to_torch(env.get_obs(), device))
        print("[warmup] done.")

        window = "demo"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

        if _move_to(env, driver, default_mat, move_speed_m_s,
                    window, "default") == "quit":
            return

        idx = 0
        while True:
            cmd = _idle_loop(env, window, idx, len(keyframes))
            if cmd == "q":
                break
            if cmd == "w":
                idx = 0
                print(f"[keyframes] index reset — next 's' starts at keyframe 1.")
                continue
            if cmd == "r":
                if _move_to(env, driver, default_mat, move_speed_m_s,
                            window, "default") == "quit":
                    break
                continue

            # cmd == "s"
            if idx >= len(keyframes):
                print("[keyframes] end reached — returning to default, no action.")
                if _move_to(env, driver, default_mat, move_speed_m_s,
                            window, "default") == "quit":
                    break
                continue

            target = keyframes[idx]
            label = f"keyframe {idx+1}/{len(keyframes)}"
            if _move_to(env, driver, target, move_speed_m_s,
                        window, label) == "quit":
                break
            idx += 1
            print(f"[inference] starting at {label}")
            _, exit_reason = _run_policy_loop(
                env, driver, policy, device,
                dt=dt,
                horizon=horizon,
                steps_per_inference=steps_per_inference,
                max_duration=max_duration,
                action_exec_latency=action_exec_latency,
                dry_run=False,
                window=window,
            )
            if exit_reason == "quit":
                break
            # 'reset' or 'timeout' -> return to default before next idle
            if _move_to(env, driver, default_mat, move_speed_m_s,
                        window, "default") == "quit":
                break
    finally:
        env.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
