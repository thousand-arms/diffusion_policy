"""Checkpoint loading utilities for the Piper pipeline."""

import dill
import hydra
import torch
from omegaconf import OmegaConf

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy(checkpoint: str, device: torch.device):
    """Load a policy from a training checkpoint.

    Args:
        checkpoint: path to .ckpt file.
        device: torch device to move the policy to.

    Returns:
        (policy, cfg) where policy is the EMA model (if available)
        in eval mode on the specified device.
    """
    payload = torch.load(
        open(checkpoint, "rb"), pickle_module=dill, map_location="cpu"
    )
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
