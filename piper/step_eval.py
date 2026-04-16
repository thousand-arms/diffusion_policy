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
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from piper.env.piper_driver import PiperDriver
from piper.env.step_env import StepEnv

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy(checkpoint: str, device: torch.device):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload)

    policy: BaseImagePolicy = workspace.model
    if getattr(cfg.training, "use_ema", False) and getattr(
        workspace, "ema_model", None
    ):
        policy = workspace.ema_model

    policy.eval().to(device)
    return policy, cfg


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
