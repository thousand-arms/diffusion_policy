"""Diagnose policy predictions by running inference on a training sample.

Loads the checkpoint, grabs a training sample from the dataset, runs
predict_action, and compares the prediction to ground truth.  Also prints
normalizer stats so you can spot scale issues.

Usage:
    python -m piper.diagnose_policy -c /path/to/latest.ckpt
"""

import pathlib
import sys

import click
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

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
@click.option("--sample-idx", default=0, help="dataset sample index to test")
def main(checkpoint, sample_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[policy] loading {checkpoint}")
    policy, cfg = load_policy(checkpoint, device)
    n_obs_steps = cfg.n_obs_steps
    horizon = cfg.horizon
    print(f"[policy] n_obs_steps={n_obs_steps}  horizon={horizon}  device={device}")

    # --- normalizer stats ---
    print("\n=== NORMALIZER STATS ===")
    normalizer = policy.normalizer
    for key in ["cam_pos", "cam_rot_6d", "action"]:
        params = normalizer[key].params_dict
        scale = params["scale"].cpu().numpy().flatten()
        offset = params["offset"].cpu().numpy().flatten()
        stats = params.get("input_stats", {})
        print(f"\n  {key}:")
        print(f"    scale  = {scale}")
        print(f"    offset = {offset}")
        if "min" in stats:
            print(f"    data min = {stats['min'].cpu().numpy().flatten()}")
            print(f"    data max = {stats['max'].cpu().numpy().flatten()}")

    # --- load dataset ---
    print("\n=== LOADING DATASET ===")
    dataset_cfg = cfg.task.dataset
    dataset = hydra.utils.instantiate(dataset_cfg)
    print(f"dataset length: {len(dataset)}")

    sample = dataset[sample_idx]
    obs = sample["obs"]
    gt_action = sample["action"]  # (horizon, 9)

    print(f"\n=== SAMPLE {sample_idx} ===")
    for k, v in obs.items():
        print(f"  obs[{k}]: shape={tuple(v.shape)}  dtype={v.dtype}  "
              f"range=[{v.min():.4f}, {v.max():.4f}]")
    print(f"  action: shape={tuple(gt_action.shape)}  "
          f"range=[{gt_action.min():.4f}, {gt_action.max():.4f}]")

    # ground truth obs poses
    print(f"\n  obs cam_pos:\n    {obs['cam_pos'].numpy()}")
    print(f"  obs cam_rot_6d:\n    {obs['cam_rot_6d'].numpy()}")

    # ground truth action (first few steps)
    print(f"\n  gt action (first 4 steps):")
    for i in range(min(4, gt_action.shape[0])):
        a = gt_action[i].numpy()
        pos = a[:3]
        rot = a[3:]
        print(f"    [{i}] pos={pos}  rot6d={rot}  |pos|={np.linalg.norm(pos)*1000:.1f}mm")

    # --- run inference ---
    print("\n=== INFERENCE ===")
    obs_torch = {k: v.unsqueeze(0).to(device) for k, v in obs.items()}

    with torch.no_grad():
        result = policy.predict_action(obs_torch)

    action_pred = result["action_pred"][0].cpu().numpy()  # (horizon, 9)
    action = result["action"][0].cpu().numpy()  # (n_action_steps, 9)

    print(f"  action_pred shape: {action_pred.shape}")
    print(f"  action shape:      {action.shape}")

    print(f"\n  action_pred (first 4 steps):")
    for i in range(min(4, action_pred.shape[0])):
        a = action_pred[i]
        pos = a[:3]
        rot = a[3:]
        print(f"    [{i}] pos={pos}  rot6d={rot}  |pos|={np.linalg.norm(pos)*1000:.1f}mm")

    print(f"\n  action (executable, first 4 steps):")
    for i in range(min(4, action.shape[0])):
        a = action[i]
        pos = a[:3]
        rot = a[3:]
        print(f"    [{i}] pos={pos}  rot6d={rot}  |pos|={np.linalg.norm(pos)*1000:.1f}mm")

    # --- compare pred vs gt ---
    print(f"\n=== PRED vs GT (action, first 4 steps) ===")
    start = n_obs_steps - 1
    for i in range(min(4, action.shape[0])):
        gt = gt_action[start + i].numpy()
        pred = action[i]
        pos_err = np.linalg.norm(pred[:3] - gt[:3]) * 1000
        rot_err = np.linalg.norm(pred[3:] - gt[3:])
        print(f"  step {i}: pos_err={pos_err:.1f}mm  rot_err={rot_err:.4f}")

    # --- sanity: what does the model predict if obs[-1] is identity? ---
    print(f"\n=== IDENTITY OBS CHECK ===")
    print(f"  obs[-1] cam_pos should be ~[0,0,0]: {obs['cam_pos'][-1].numpy()}")
    print(f"  obs[-1] cam_rot_6d should be ~[1,0,0,0,1,0]: {obs['cam_rot_6d'][-1].numpy()}")


if __name__ == "__main__":
    main()
