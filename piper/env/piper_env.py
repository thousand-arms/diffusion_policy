"""Real-time continuous control environment for the Piper + INDEMIND setup.

Thread architecture:
  - Camera/obs buffer thread: captures stereo frames at 50Hz in background.
  - Piper interpolation thread (inside PiperDriver): sends smooth poses at ~150Hz.
  - Main thread: calls get_obs() and exec_actions() — exec_actions is now
    non-blocking (queues waypoints into the interpolator).

Usage:
    driver = PiperDriver(smooth=True, smooth_hz=150, max_pos_speed=0.5, max_rot_speed=2.0)
    env = PiperEnv(driver, n_obs_steps=2, dt=0.06)
    env.start()
    obs = env.get_obs()
    env.exec_actions(actions, timestamps, anchor)   # non-blocking
    env.stop()
"""

import threading
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np

import pyindemind

from piper.env.piper_driver import PiperDriver
from piper.common.pose_util import relativize_poses, rel_action_to_world
from piper.common.image_util import preprocess_image


# Measured latencies (from piper/latency_tests/latency.yaml)
CAMERA_OBS_LATENCY_S = 0.0293   # 29.3ms — camera capture to get_frame() return
ROBOT_ACTION_LATENCY_S = 0.082  # 82ms — command sent to arm reaching target


class _TimestampedObsBuffer:
    """Background thread that continuously captures stereo frames from INDEMIND
    and pairs each with the current Piper camera pose and a monotonic timestamp.
    """

    def __init__(
        self,
        driver: PiperDriver,
        resolution: str = "640x400",
        img_hz: int = 50,
        imu_hz: int = 200,
        buf_size: int = 128,
        camera_obs_latency: float = CAMERA_OBS_LATENCY_S,
    ):
        self.camera_obs_latency = camera_obs_latency
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
            capture_time = time.monotonic() - self.camera_obs_latency
            with self._lock:
                self.buf.append(
                    {
                        "sdk_timestamp": ts,
                        "receive_time": capture_time,
                        "cam0": left,
                        "cam1": right,
                        "cam_pos": pos.astype(np.float32),
                        "cam_rot_axis_angle": rotvec.astype(np.float32),
                    }
                )

    def get_all(self):
        with self._lock:
            return list(self.buf)

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


class PiperEnv:
    """Real-time continuous control environment for Piper + INDEMIND."""

    def __init__(
        self,
        driver: PiperDriver,
        resolution: str = "640x400",
        img_hz: int = 50,
        n_obs_steps: int = 2,
        obs_image_size: int = 224,
        dt: float = 0.06,
        max_step_m: float = 0.05,
        camera_obs_latency: float = CAMERA_OBS_LATENCY_S,
        robot_action_latency: float = ROBOT_ACTION_LATENCY_S,
    ):
        self.driver = driver
        self.n_obs_steps = n_obs_steps
        self.obs_image_size = obs_image_size
        self.dt = dt
        self.max_step_m = max_step_m
        self.robot_action_latency = robot_action_latency
        self._obs_buf = _TimestampedObsBuffer(
            driver=driver,
            resolution=resolution,
            img_hz=img_hz,
            camera_obs_latency=camera_obs_latency,
        )

    def start(self):
        self._obs_buf.start()

    def stop(self):
        self._obs_buf.stop()
        self.driver.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ------------------------------------------------------------------- obs

    def get_obs(self) -> Optional[dict]:
        """Get the latest observation with timestamps."""
        min_frames = max(self.n_obs_steps, 4)
        buf = self._obs_buf.get_all()
        if len(buf) < min_frames:
            return None

        latest = buf[-1]
        latest_t = latest["receive_time"]

        target_times = [
            latest_t - i * self.dt for i in range(self.n_obs_steps - 1, -1, -1)
        ]

        selected = []
        for target_t in target_times:
            best_idx = min(
                range(len(buf)),
                key=lambda i: abs(buf[i]["receive_time"] - target_t),
            )
            selected.append(buf[best_idx])

        cam0 = np.stack(
            [preprocess_image(s["cam0"], self.obs_image_size) for s in selected]
        )
        cam1 = np.stack(
            [preprocess_image(s["cam1"], self.obs_image_size) for s in selected]
        )
        positions = np.stack([s["cam_pos"] for s in selected])
        rotvecs = np.stack([s["cam_rot_axis_angle"] for s in selected])
        timestamps = np.array([s["receive_time"] for s in selected], dtype=np.float64)

        rel_pos, rel_rot_6d, anchor = relativize_poses(
            positions, rotvecs, anchor_idx=self.n_obs_steps - 1
        )

        return {
            "cam0": cam0,
            "cam1": cam1,
            "cam_pos": rel_pos,
            "cam_rot_6d": rel_rot_6d,
            "timestamp": timestamps,
            "anchor_mat": anchor,
        }

    # ---------------------------------------------------------------- action

    def exec_actions(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        anchor_mat: np.ndarray,
        compensate_latency: bool = True,
        verbose: bool = False,
    ) -> int:
        """Non-blocking: queue camera-frame waypoints into the interp thread.

        Args:
            actions:    (N, 9) relative actions (pos3 + rot6d6).
            timestamps: (N,) monotonic timestamps — when to arrive.
            anchor_mat: (4, 4) world-frame anchor pose.
            compensate_latency: subtract robot_action_latency from target_time
                so the arm physically arrives at timestamp[i].

        Returns number of waypoints scheduled (post safety filter).
        """
        if self.driver._interp_ctrl is None:
            raise RuntimeError(
                "PiperEnv.exec_actions requires driver smooth=True")

        current_pos = self.driver.get_camera_pose_mat()[:3, 3]
        n_sent = 0
        n_skipped_far = 0

        r_latency = self.robot_action_latency if compensate_latency else 0.0

        max_delta = 0.0
        for i in range(len(actions)):
            T_world = rel_action_to_world(actions[i], anchor_mat)
            target_pos = T_world[:3, 3]
            delta = np.linalg.norm(target_pos - current_pos)
            max_delta = max(max_delta, delta)
            if delta > self.max_step_m:
                n_skipped_far += 1
                # don't break — later actions in horizon may be fine,
                # but conservatively stop to avoid teleport across obstacles
                break
            target_time = float(timestamps[i]) - r_latency
            self.driver.schedule_waypoint(T_world, target_time)
            n_sent += 1

        if verbose:
            print(
                f"[exec_actions] scheduled {n_sent}/{len(actions)}  "
                f"max_delta={max_delta*1000:.1f}mm  skipped_far={n_skipped_far}",
                flush=True,
            )
        return n_sent

    # --------------------------------------------------------------- preview

    def get_preview(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._obs_buf.latest_preview()
