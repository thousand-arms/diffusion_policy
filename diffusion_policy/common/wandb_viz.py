"""Wandb visualization helpers for trajectory prediction training."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb


def make_trajectory_figure(
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    n_obs_steps: int,
) -> plt.Figure:
    """Plot gt vs pred trajectories as 2D projections.

    Args:
        gt_pos:   (T, 3) ground truth positions (relative frame).
        pred_pos: (T, 3) predicted positions (relative frame).
        n_obs_steps: number of obs steps (to mark obs/action boundary).

    Returns a matplotlib Figure (caller should close it after use).
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    projections = [('X', 'Y', 0, 1), ('X', 'Z', 0, 2), ('Y', 'Z', 1, 2)]

    for ax, (xlabel, ylabel, i, j) in zip(axes, projections):
        # ground truth
        ax.plot(gt_pos[:, i] * 1000, gt_pos[:, j] * 1000,
                'o-', color='#2196F3', markersize=3, linewidth=1.2,
                label='gt', alpha=0.8)
        # prediction
        ax.plot(pred_pos[:, i] * 1000, pred_pos[:, j] * 1000,
                's--', color='#F44336', markersize=3, linewidth=1.2,
                label='pred', alpha=0.8)
        # mark anchor (obs[-1]) — should be at origin
        ax.plot(0, 0, '*', color='black', markersize=10, zorder=5,
                label='anchor')
        # mark obs/action boundary
        if n_obs_steps < len(gt_pos):
            ax.axvline(x=gt_pos[n_obs_steps - 1, i] * 1000,
                       color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel(f'{xlabel} (mm)')
        ax.set_ylabel(f'{ylabel} (mm)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle('Trajectory (relative frame)', fontsize=10)
    fig.tight_layout()
    return fig


def make_stereo_image(cam0: np.ndarray, cam1: np.ndarray) -> np.ndarray:
    """Concatenate a stereo pair side-by-side for display.

    Args:
        cam0: (3, H, W) float32 [0, 1]
        cam1: (3, H, W) float32 [0, 1]

    Returns (H, 2*W, 3) uint8.
    """
    # (3, H, W) -> (H, W, 3), take first channel since they're triplicated grayscale
    left = (cam0[0] * 255).clip(0, 255).astype(np.uint8)
    right = (cam1[0] * 255).clip(0, 255).astype(np.uint8)
    # stack as 3-channel for wandb
    left = np.stack([left] * 3, axis=-1)
    right = np.stack([right] * 3, axis=-1)
    return np.concatenate([left, right], axis=1)


def log_sample_visualizations(
    obs_dict: dict,
    gt_action,
    pred_action,
    n_obs_steps: int,
    prefix: str,
    sample_idx: int = 0,
) -> dict:
    """Build wandb log dict with stereo images and trajectory plots.

    Args:
        obs_dict:    dict of tensors (B, T, ...) — raw (unnormalized) obs.
        gt_action:   (B, T, 9) tensor — ground truth action.
        pred_action: (B, T, 9) tensor — predicted action.
        n_obs_steps: number of observation steps.
        prefix:      'train' or 'val'.
        sample_idx:  which batch element to visualize.

    Returns a dict of wandb-loggable items.
    """
    log = {}

    # move to numpy, pick one sample
    cam0 = obs_dict['cam0'][sample_idx].detach().cpu().numpy()  # (T, 3, H, W)
    cam1 = obs_dict['cam1'][sample_idx].detach().cpu().numpy()
    gt = gt_action[sample_idx].detach().cpu().numpy()           # (T, 9)
    pred = pred_action[sample_idx].detach().cpu().numpy()       # (T, 9)

    # stereo image from last obs step
    stereo = make_stereo_image(
        cam0[n_obs_steps - 1], cam1[n_obs_steps - 1])
    log[f'{prefix}/stereo_obs'] = wandb.Image(
        stereo, caption=f'{prefix} obs (last step)')

    # trajectory plot
    fig = make_trajectory_figure(
        gt_pos=gt[:, :3],
        pred_pos=pred[:, :3],
        n_obs_steps=n_obs_steps,
    )
    log[f'{prefix}/trajectory'] = wandb.Image(fig)
    plt.close(fig)

    return log
