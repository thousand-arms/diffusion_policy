"""Step-by-step policy evaluation on the Piper + INDEMIND setup.

Shows a live stereo preview. Press 's' to step: grabs the latest obs,
runs inference, and executes the predicted horizon. Press 'q' to quit.

Usage (from repo root):
    python -m piper.step_eval -c /path/to/checkpoint.ckpt
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

from piper.common.checkpoint_util import load_policy
from piper.env.piper_driver import PiperDriver
from piper.env.step_env import StepEnv


@click.command()
@click.option("--checkpoint", "-c", required=True, type=click.Path(exists=True))
@click.option("--can-port", default="can0")
@click.option("--speed-pct", default=30, help="Piper MOVE P speed percentage")
@click.option("--step-dt", default=0.1, help="sleep between horizon waypoints (s)")
@click.option("--n-exec-steps", default=0, help="steps of action_pred to execute; 0=all")
@click.option("--max-step-m", default=0.05, help="abort if a step exceeds this (m)")
@click.option("--confirm", is_flag=True, help="prompt before each waypoint")
def main(
    checkpoint,
    can_port,
    speed_pct,
    step_dt,
    n_exec_steps,
    max_step_m,
    confirm,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[policy] loading {checkpoint}")
    policy, cfg = load_policy(checkpoint, device)
    n_obs_steps = cfg.n_obs_steps
    print(f"[policy] n_obs_steps={n_obs_steps}  horizon={cfg.horizon}  device={device}")

    print("[robot] connecting Piper…")
    driver = PiperDriver(can_port=can_port, speed_pct=speed_pct)

    print("[env] starting…")
    env = StepEnv(driver, max_step_m=max_step_m)
    env.start()

    # wait for buffer to fill
    t0 = time.time()
    while env.get_obs(n_obs_steps) is None:
        if time.time() - t0 > 5.0:
            env.stop()
            raise RuntimeError("Timed out waiting for observations.")
        time.sleep(0.05)

    cv2.namedWindow("step_eval", cv2.WINDOW_AUTOSIZE)
    print("[ready] press 's' to step, 'q' to quit (focus the preview window).")

    try:
        while True:
            preview = env.get_preview()
            if preview is not None:
                left, right = preview
                cv2.imshow("step_eval", np.concatenate([left, right], axis=1))
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            if key != ord("s"):
                continue

            # --- step ---
            result = env.get_obs(n_obs_steps)
            if result is None:
                print("[step] obs buffer not ready, skipping.")
                continue
            obs, anchor = result

            # to torch
            obs_torch = {
                k: torch.from_numpy(v).unsqueeze(0).to(device)
                for k, v in obs.items()
            }

            t_infer = time.time()
            with torch.no_grad():
                action_pred = policy.predict_action(obs_torch)["action_pred"][0]
            action = action_pred.detach().cpu().numpy()
            print(f"[step] inference {(time.time() - t_infer)*1000:.0f} ms  action={action.shape}")

            # save 3D trajectory plot before executing
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from diffusion_policy.common.wandb_viz import make_stereo_image
            import os
            plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tmp', 'step_eval_plots')
            os.makedirs(plot_dir, exist_ok=True)
            step_count = len([f for f in os.listdir(plot_dir) if f.startswith('step_') and f.endswith('_traj.png')])
            pos = action[:, :3] * 1000  # to mm
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            # anchor frame: x-right, y-down, z-forward
            ax.plot(pos[:, 0], pos[:, 2], -pos[:, 1], 's-', color='#F44336', markersize=4, linewidth=1.5, label='pred')
            ax.scatter([0], [0], [0], marker='*', c='black', s=100, zorder=5, label='anchor')
            ax.set_xlabel('X right (mm)')
            ax.set_ylabel('Z forward (mm)')
            ax.set_zlabel('up (mm)')
            ax.set_title(f'Step {step_count} — predicted trajectory\n(anchor frame: X=right, Z=forward, up=-Y)')
            ax.legend()
            # equal aspect
            max_range = max(abs(pos).max(), 1.0)
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
            # view from behind and slightly above the camera
            ax.view_init(elev=25, azim=-60)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'step_{step_count:03d}_traj.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)
            # also save stereo obs
            cam0 = obs['cam0'][n_obs_steps - 1]  # (3, H, W)
            cam1 = obs['cam1'][n_obs_steps - 1]
            stereo = make_stereo_image(cam0, cam1)
            plt.imsave(os.path.join(plot_dir, f'step_{step_count:03d}_stereo.png'), stereo)
            # print trajectory summary
            print(f"[step] pred trajectory: start=({pos[0,0]:+.1f},{pos[0,1]:+.1f},{pos[0,2]:+.1f})mm  end=({pos[-1,0]:+.1f},{pos[-1,1]:+.1f},{pos[-1,2]:+.1f})mm")
            print(f"[step] saved plots to {plot_dir}/step_{step_count:03d}_*")

            executed = env.execute_actions(
                action_pred=action,
                anchor=anchor,
                n_steps=n_exec_steps,
                step_dt=step_dt,
                confirm=confirm,
            )
            print(f"[step] executed {executed} waypoints.")
            driver.print_status()

    finally:
        env.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
