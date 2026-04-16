"""Measure end-to-end policy inference latency.

Builds a DiffusionUnetImagePolicy from the trace_image workspace config with
random init (no checkpoint), runs predict_action on a random obs dict 100
times, and prints mean/median wall-clock latency including host->device
transfer and the .cpu().numpy() postprocess.
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time
import numpy as np
import torch
import hydra
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

CONFIG_DIR = str(ROOT / "diffusion_policy" / "config")
CONFIG_NAME = "train_diffusion_unet_image_trace_workspace"
N_RUNS = 100
N_WARMUP = 5

# required by configs that use ${eval:...} interpolations
OmegaConf.register_new_resolver("eval", eval, replace=True)


def build_random_obs(shape_meta, n_obs_steps):
    """Return numpy obs dict (T, *shape) matching the shape_meta['obs'] spec."""
    obs = {}
    for key, attr in shape_meta["obs"].items():
        shape = tuple(attr["shape"])
        obs[key] = np.random.rand(n_obs_steps, *shape).astype(np.float32)
    return obs


def main():
    device = torch.device("cuda")

    # compose config (resolves `defaults: task: trace_image`)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=CONFIG_NAME)
    OmegaConf.resolve(cfg)

    # build workspace; keep the randomly-initialized policy (no checkpoint)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    policy: BaseImagePolicy = workspace.model

    # populate the normalizer with identity params — training would fit these
    # from the dataset, but for pure latency measurement identity is fine and
    # has the same compute cost as any fitted affine transform.
    for key in cfg.task.shape_meta["obs"]:
        policy.normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    policy.normalizer["action"] = SingleFieldLinearNormalizer.create_identity()

    policy.eval().to(device)
    policy.num_inference_steps = 16  # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {n_params/1e6:.2f} M")
    print(f"down_dims: {list(cfg.policy.down_dims)}")
    print(f"horizon: {cfg.horizon}")

    # random obs on host
    cpu_obs = build_random_obs(cfg.task.shape_meta, cfg.n_obs_steps)

    times = []
    with torch.no_grad():
        for i in range(N_WARMUP + N_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # pre: host -> device + add batch dim
            obs_dict = {
                k: torch.from_numpy(v).unsqueeze(0).to(device)
                for k, v in cpu_obs.items()
            }
            # inference
            result = policy.predict_action(obs_dict)
            # post: device -> host
            action = result["action"][0].detach().cpu().numpy()

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= N_WARMUP:
                times.append(t1 - t0)

    times_ms = np.array(times) * 1000
    print(f"Inference latency over {N_RUNS} runs (ms):")
    print(f"  mean   = {times_ms.mean():.2f}")
    print(f"  median = {np.median(times_ms):.2f}")
    print(f"  min    = {times_ms.min():.2f}")
    print(f"  max    = {times_ms.max():.2f}")


if __name__ == "__main__":
    main()
