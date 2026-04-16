"""Step-by-step inference environment for the Piper + INDEMIND setup.

Combines PiperDriver (robot control) with pyindemind (stereo camera) into
a minimal env that can:
  1. Buffer observations in a background thread
  2. Return a policy-ready obs dict (preprocessed images + relativized poses)
  3. Execute a predicted action horizon as sequential camera-frame waypoints

Usage:
    env = StepEnv(piper_driver)
    env.start()
    obs, anchor = env.get_obs(n_obs_steps=2)    # numpy dict
    # ... run policy ...
    env.execute_actions(action_pred, anchor)
    env.stop()
"""

import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import pyindemind

from piper.env.piper_driver import PiperDriver


# ---------------------------------------------------------------------------
# Pose math
# ---------------------------------------------------------------------------


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Decode 6D rotation [r00,r10,r20, r01,r11,r21] -> (3,3) via Gram-Schmidt.
    Matches the layout in trace_dataset._relativize_poses."""
    a1 = rot6d[:3]
    a2 = rot6d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _relativize(
    positions: np.ndarray, rotvecs: np.ndarray, anchor_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relativize poses to the anchor step.

    Returns (rel_pos, rel_rot_6d, anchor_mat) matching trace_dataset convention.
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
    rel_rot_6d = np.concatenate([rel[:, :3, 0], rel[:, :3, 1]], axis=-1).astype(
        np.float32
    )

    return rel_pos, rel_rot_6d, anchor


def _preprocess_image(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Grayscale (H,W) or (H,W,1) uint8 -> (3, size, size) float32 [0,1].

    Matches trace_dataset._sample_to_data: resize, normalize, triplicate
    so a 3-channel ResNet backbone can consume it."""
    if img.ndim == 3:
        img = img[..., 0]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.repeat(img[None], 3, axis=0)  # (3, H, W)


# ---------------------------------------------------------------------------
# Observation buffer (background thread)
# ---------------------------------------------------------------------------


class _ObsBuffer:
    """Continuously captures stereo frames from INDEMIND and pairs each with
    the current Piper camera pose. Thread-safe deque for the main thread."""

    def __init__(
        self,
        driver: PiperDriver,
        resolution: str = "640x400",
        img_hz: int = 50,
        imu_hz: int = 200,
        buf_size: int = 64,
    ):
        self.driver = driver
        self.cam = pyindemind.Camera()
        ok = self.cam.start(resolution=resolution, img_hz=img_hz, imu_hz=imu_hz)
        if not ok:
            raise RuntimeError("INDEMIND camera failed to start.")
        self.buf: deque = deque(maxlen=buf_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            frame = self.cam.get_frame(timeout_s=0.1)
            if frame is None:
                continue
            ts, left, right = frame
            pos, rotvec, _ = self.driver.get_camera_pose()
            with self._lock:
                self.buf.append(
                    {
                        "timestamp": ts,
                        "cam0": left,
                        "cam1": right,
                        "cam_pos": pos.astype(np.float32),
                        "cam_rot_axis_angle": rotvec.astype(np.float32),
                    }
                )

    def get_last(self, n: int) -> Optional[List[dict]]:
        with self._lock:
            if len(self.buf) < n:
                return None
            return list(self.buf)[-n:]

    def latest_preview(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            if not self.buf:
                return None
            s = self.buf[-1]
        return s["cam0"], s["cam1"]

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cam.stop()


# ---------------------------------------------------------------------------
# StepEnv
# ---------------------------------------------------------------------------


class StepEnv:
    def __init__(
        self,
        driver: PiperDriver,
        resolution: str = "640x400",
        img_hz: int = 50,
        obs_image_size: int = 224,
        max_step_m: float = 0.05,
    ):
        self.driver = driver
        self.obs_image_size = obs_image_size
        self.max_step_m = max_step_m
        self._obs_buf = _ObsBuffer(
            driver=driver,
            resolution=resolution,
            img_hz=img_hz,
        )

    def start(self):
        self._obs_buf.start()

    def stop(self):
        self._obs_buf.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ------------------------------------------------------------------- obs

    def get_obs(
        self, n_obs_steps: int = 2
    ) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray]]:
        """Grab the last n_obs_steps buffered samples and build a policy-ready
        obs dict (numpy, unbatched).

        Returns (obs_dict, anchor_4x4) or None if the buffer isn't full yet.

        obs_dict keys match trace_dataset._sample_to_data:
            cam0:       (T, 3, H, W) float32 [0,1]
            cam1:       (T, 3, H, W) float32 [0,1]
            cam_pos:    (T, 3) float32 relativized
            cam_rot_6d: (T, 6) float32 relativized
        """
        samples = self._obs_buf.get_last(n_obs_steps)
        if samples is None:
            return None

        cam0 = np.stack(
            [_preprocess_image(s["cam0"], self.obs_image_size) for s in samples]
        )
        cam1 = np.stack(
            [_preprocess_image(s["cam1"], self.obs_image_size) for s in samples]
        )
        positions = np.stack([s["cam_pos"] for s in samples])
        rotvecs = np.stack([s["cam_rot_axis_angle"] for s in samples])

        rel_pos, rel_rot_6d, anchor = _relativize(
            positions, rotvecs, anchor_idx=n_obs_steps - 1
        )

        obs = {
            "cam0": cam0,
            "cam1": cam1,
            "cam_pos": rel_pos,
            "cam_rot_6d": rel_rot_6d,
        }
        return obs, anchor

    # ---------------------------------------------------------------- action

    def execute_actions(
        self,
        action_pred: np.ndarray,
        anchor: np.ndarray,
        n_steps: int = 0,
        step_dt: float = 0.1,
        confirm: bool = False,
    ) -> int:
        """Execute predicted actions as sequential camera-frame waypoints.

        Args:
            action_pred: (H, 9) relativized action from the policy
                         (pos 3 + rot6d 6), in the anchor's frame.
            anchor:      (4, 4) world-frame pose of obs[-1] (from get_obs).
            n_steps:     how many steps to execute; 0 = all.
            step_dt:     sleep between waypoints (seconds).
            confirm:     if True, prompt before each waypoint.

        Returns the number of steps actually executed.
        """
        if n_steps <= 0:
            n_steps = action_pred.shape[0]
        horizon = action_pred[:n_steps]

        # convert to world-frame camera poses
        world_poses = []
        for i in range(horizon.shape[0]):
            T_rel = np.eye(4)
            T_rel[:3, :3] = rot6d_to_matrix(horizon[i, 3:])
            T_rel[:3, 3] = horizon[i, :3]
            world_poses.append(anchor @ T_rel)

        # execute with safety checks
        last_pos = anchor[:3, 3]
        executed = 0
        for i, T_world in enumerate(world_poses):
            pos = T_world[:3, 3]
            delta = np.linalg.norm(pos - last_pos)
            if delta > self.max_step_m:
                print(
                    f"[abort] step {i}: delta={delta*1000:.1f} mm "
                    f"> max_step_m={self.max_step_m*1000:.1f} mm"
                )
                break

            if confirm:
                ans = (
                    input(
                        f"  step {i:2d}  pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f})m"
                        f"  delta={delta*1000:.1f}mm  [enter/s/q]: "
                    )
                    .strip()
                    .lower()
                )
                if ans == "q":
                    break
                if ans == "s":
                    last_pos = pos
                    continue

            self.driver.set_camera_pose_mat(T_world)
            time.sleep(step_dt)
            last_pos = pos
            executed += 1

        return executed

    # --------------------------------------------------------------- preview

    def get_preview(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Latest raw stereo pair for display, or None."""
        return self._obs_buf.latest_preview()


if __name__ == "__main__":
    import cv2

    driver = PiperDriver(speed_pct=30)
    env = StepEnv(driver)
    env.start()

    import time

    time.sleep(1)

    result = env.get_obs(n_obs_steps=2)
    assert result is not None, "buffer not ready"
    obs, anchor = result

    print(
        f"cam0:       {obs['cam0'].shape}  range=[{obs['cam0'].min():.3f}, {obs['cam0'].max():.3f}]"
    )
    print(f"cam1:       {obs['cam1'].shape}")
    print(f"cam_pos:    {obs['cam_pos'].shape}  last={obs['cam_pos'][-1]}")
    print(f"cam_rot_6d: {obs['cam_rot_6d'].shape}  last={obs['cam_rot_6d'][-1]}")
    print(f"anchor:\n{anchor}")

    left, right = env.get_preview()
    cv2.imshow("left", left)
    cv2.imshow("right", right)
    print("Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    env.stop()
