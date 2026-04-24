"""Camera-frame pose storage: one 'default' pose plus a list of 'keyframes'.

Poses are stored in the robot base frame (camera center) as position (m)
and rotation (axis-angle, rad) — matching the zarr dataset convention.

Yaml schema:
    default:
      pos_m: [x, y, z]
      rot_axis_angle_rad: [rx, ry, rz]
    keyframes:
      - pos_m: [...]
        rot_axis_angle_rad: [...]
      - ...
"""

from __future__ import annotations

import pathlib

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

CAMERA_POSES_PATH = pathlib.Path(__file__).resolve().parent / "camera_poses.yaml"


def _entry_to_mat(entry: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(entry["rot_axis_angle_rad"]).as_matrix()
    T[:3, 3] = entry["pos_m"]
    return T


def _mat_to_entry(T: np.ndarray) -> dict:
    pos = T[:3, 3]
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    return {
        "pos_m": [float(x) for x in pos],
        "rot_axis_angle_rad": [float(x) for x in rotvec],
    }


def load_camera_poses(path: pathlib.Path = CAMERA_POSES_PATH):
    """Return (default_mat, keyframe_mats). (None, []) if missing/empty."""
    if not path.exists() or path.stat().st_size == 0:
        return None, []
    with path.open() as f:
        data = yaml.safe_load(f)
    if not data:
        return None, []
    default = _entry_to_mat(data["default"]) if data.get("default") else None
    keyframes = [_entry_to_mat(e) for e in (data.get("keyframes") or [])]
    return default, keyframes


def save_camera_poses(
    default: np.ndarray,
    keyframes: list,
    path: pathlib.Path = CAMERA_POSES_PATH,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "default": _mat_to_entry(default),
        "keyframes": [_mat_to_entry(T) for T in keyframes],
    }
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
