"""Pose math utilities for the Piper pipeline."""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Decode 6D rotation [r00,r10,r20, r01,r11,r21] -> (3,3) via Gram-Schmidt.

    Layout matches trace_dataset._relativize_poses: first two columns of the
    rotation matrix, stacked as a flat (6,) vector.
    """
    a1 = rot6d[:3]
    a2 = rot6d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def relativize_poses(
    positions: np.ndarray,
    rotvecs: np.ndarray,
    anchor_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relativize poses to the anchor step.

    Args:
        positions: (N, 3) absolute positions in world frame.
        rotvecs:   (N, 3) axis-angle rotations in world frame.
        anchor_idx: index of the anchor step.

    Returns:
        rel_pos:    (N, 3) float32 relativized positions.
        rel_rot_6d: (N, 6) float32 relativized 6D rotations.
        anchor_mat: (4, 4) float64 world-frame pose of the anchor.
    """
    n = positions.shape[0]
    pose_mat = np.zeros((n, 4, 4), dtype=np.float64)
    pose_mat[:, :3, :3] = R.from_rotvec(rotvecs).as_matrix()
    pose_mat[:, :3, 3] = positions
    pose_mat[:, 3, 3] = 1.0

    anchor = pose_mat[anchor_idx].copy()
    anchor_inv = np.linalg.inv(anchor)
    rel = anchor_inv @ pose_mat

    rel_pos = rel[:, :3, 3].astype(np.float32)
    rel_rot_6d = np.concatenate(
        [rel[:, :3, 0], rel[:, :3, 1]], axis=-1
    ).astype(np.float32)

    return rel_pos, rel_rot_6d, anchor


def rel_action_to_world(action_9d: np.ndarray, anchor_mat: np.ndarray) -> np.ndarray:
    """Convert a single (9,) relative action to a world-frame 4x4 pose.

    Args:
        action_9d:  (9,) relative action — pos (3) + rot6d (6).
        anchor_mat: (4, 4) world-frame anchor pose.

    Returns:
        T_world: (4, 4) float64 target pose in world frame.
    """
    T_rel = np.eye(4)
    T_rel[:3, :3] = rot6d_to_matrix(action_9d[3:])
    T_rel[:3, 3] = action_9d[:3]
    return anchor_mat @ T_rel
