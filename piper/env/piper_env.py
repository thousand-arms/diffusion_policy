"""Real-time continuous control environment for the Piper + INDEMIND setup.

Two-thread architecture designed for use with a separate inference thread:
  - Camera/obs buffer thread: captures frames at 50Hz in the background
  - Main thread calls get_obs() and send_action() at a fixed tick rate

Usage:
    env = PiperEnv(driver, n_obs_steps=2, dt=0.06)
    env.start()
    obs = env.get_obs()          # dict with timestamps
    env.send_action(action, anchor)  # single waypoint
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

    Timestamps are adjusted for camera latency: receive_time is shifted back
    by CAMERA_OBS_LATENCY_S so it approximates when the image was captured,
    not when it arrived.
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
        # consider adding ring buffer to timestamp everything. camera frames are actually from the past, where as pose is more current.
        while not self._stop.is_set():
            frame = self.cam.get_frame(timeout_s=0.1)
            if frame is None:
                continue
            ts, left, right = frame
            pos, rotvec, _ = self.driver.get_camera_pose()
            # Shift receive_time back by camera latency to approximate
            # when the image was actually captured, not when it arrived.
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

    def get_latest(self, n: int):
        """Return the last n entries, or None if not enough data."""
        with self._lock:
            if len(self.buf) < n:
                return None
            return list(self.buf)[-n:]

    def get_all(self):
        """Return all buffered entries."""
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
    """Real-time continuous control environment for Piper + INDEMIND.

    Provides get_obs() for timestamped observations and send_action() for
    commanding the robot one waypoint at a time.
    """

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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ------------------------------------------------------------------- obs

    def get_obs(self) -> Optional[dict]:
        """Get the latest observation with timestamps.

        Selects n_obs_steps frames from the buffer spaced at self.dt apart.
        Relativizes poses to the anchor (last obs step).

        Returns dict with keys:
            cam0:       (n_obs_steps, 3, H, W) float32 [0,1]
            cam1:       (n_obs_steps, 3, H, W) float32 [0,1]
            cam_pos:    (n_obs_steps, 3) float32 relativized
            cam_rot_6d: (n_obs_steps, 6) float32 relativized
            timestamp:  (n_obs_steps,) float64 monotonic receive times
            anchor_mat: (4, 4) float64 world-frame pose of last obs step
        Or None if the buffer isn't ready.
        """
        # Need enough frames to span (n_obs_steps - 1) * dt
        min_frames = max(self.n_obs_steps, 4)
        buf = self._obs_buf.get_all()
        if len(buf) < min_frames:
            return None

        # Select frames with proper dt spacing
        latest = buf[-1]
        latest_t = latest["receive_time"]

        # Target timestamps going backwards: [latest - (n-1)*dt, ..., latest]
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

        # Build observation arrays
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

    def send_action(self, action_9d: np.ndarray, anchor_mat: np.ndarray) -> bool:
        """Send a single waypoint to the robot (bypasses interpolation).

        Args:
            action_9d:  (9,) relative action in anchor frame (pos3 + rot6d6).
            anchor_mat: (4, 4) world-frame anchor pose from get_obs().

        Returns True if sent, False if safety-skipped.
        """
        T_world = rel_action_to_world(action_9d, anchor_mat)

        # Safety: check delta from current camera position
        current_pos = self.driver.get_camera_pose_mat()[:3, 3]
        target_pos = T_world[:3, 3]
        delta = np.linalg.norm(target_pos - current_pos)
        if delta > self.max_step_m:
            print(
                f"[safety] skipped: delta={delta*1000:.1f}mm "
                f"> max_step_m={self.max_step_m*1000:.1f}mm"
            )
            return False

        self.driver.set_camera_pose_mat(T_world)
        return True

    def exec_actions(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        anchor_mat: np.ndarray,
    ) -> int:
        """Execute a batch of actions, sending each waypoint at its scheduled time.

        Converts relative actions to world-frame camera poses. If the driver
        has smooth=True, sends to the interpolation controller. Otherwise,
        sends waypoints directly with precise_wait timing.

        Args:
            actions:    (N, 9) relative actions in anchor frame (pos3 + rot6d6).
            timestamps: (N,) monotonic timestamps for each action.
            anchor_mat: (4, 4) world-frame anchor pose from get_obs().

        Returns the number of waypoints sent.
        """
        from diffusion_policy.common.precise_sleep import precise_wait

        # Convert all actions to world-frame poses, stopping at first unsafe
        poses = []
        valid_times = []
        current_pos = self.driver.get_camera_pose_mat()[:3, 3]

        for i in range(len(actions)):
            T_world = rel_action_to_world(actions[i], anchor_mat)
            target_pos = T_world[:3, 3]
            delta = np.linalg.norm(target_pos - current_pos)
            if delta > self.max_step_m:
                break
            poses.append(T_world)
            valid_times.append(timestamps[i])

        if len(poses) == 0:
            return 0

        if self.driver.smooth and self.driver._interp_ctrl is not None:
            # Smooth mode: send to interpolation controller
            self.driver.schedule_waypoints(
                times=np.array(valid_times),
                poses_4x4=poses,
            )
        else:
            # Direct mode: send each waypoint at its scheduled time
            for i, (T_world, ts) in enumerate(zip(poses, valid_times)):
                self.driver.set_camera_pose_mat(T_world)
                # Wait until next waypoint's time (except after last)
                if i < len(poses) - 1:
                    precise_wait(valid_times[i + 1])

        return len(poses)

    # --------------------------------------------------------------- preview

    def get_preview(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Latest raw stereo pair for display, or None."""
        return self._obs_buf.latest_preview()
