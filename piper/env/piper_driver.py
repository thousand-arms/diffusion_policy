"""Thin wrapper around the Piper SDK for camera-frame pose control.

The policy predicts and observes in the **camera frame** (mounted on the
wrist). This driver converts between that frame and the SDK's wrist-frame
commands/feedback, handling:
  - TF_WRIST_TO_CAMERA rigid transform
  - SDK unit scaling (0.001 mm / 0.001 deg)
  - Extrinsic XYZ euler convention
  - Optional high-frequency interpolation controller for smooth motion
    (UMI-style: trim+insert trajectory, max_pos/rot_speed rate limit)

Usage:
    driver = PiperDriver(smooth=True, send_hz=150)
    pos, rotvec, T = driver.get_camera_pose()
    driver.schedule_waypoint(T_world, target_time)   # smooth=True
    driver.set_camera_pose(pos, rotvec)              # smooth=False
"""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from piper_sdk import C_PiperInterface_V2

from piper.env.pose_trajectory_interpolator import (
    PoseTrajectoryInterpolator,
    mat_to_pose6,
    pose6_to_mat,
)

# Piper SDK units
_POS_TO_M = 1e-6  # 0.001 mm -> m  (feedback)
_ROT_TO_DEG = 1e-3  # 0.001 deg -> deg (feedback)
_M_TO_POS = 1_000_000  # m -> 0.001 mm (command)
_DEG_TO_ROT = 1_000  # deg -> 0.001 deg (command)

TF_WRIST_TO_CAMERA = np.array(
    [
        [-0.0000000, 0.7071068, -0.7071068, -0.07958],
        [-0.7071068, -0.5000000, -0.5000000, 0.18364],
        [-0.7071068, 0.5000000, 0.5000000, 0.10064],
        [0, 0, 0, 1],
    ]
)


class _InterpolationController(threading.Thread):
    """High-frequency sender driven by a PoseTrajectoryInterpolator.

    Every tick: reads the interpolator at t_now, sends the pose via
    EndPoseCtrl (MOVE P). New waypoints arrive via schedule_waypoint() and
    get trim-and-inserted into the trajectory.
    """

    def __init__(
        self,
        driver: "PiperDriver",
        send_hz: int = 150,
        max_pos_speed: float = 0.5,  # m/s
        max_rot_speed: float = 2.0,  # rad/s
        verbose: bool = False,
    ):
        super().__init__(daemon=True, name="PiperInterpController")
        self._driver = driver
        self._dt = 1.0 / send_hz
        self._max_pos_speed = max_pos_speed
        self._max_rot_speed = max_rot_speed
        self._verbose = verbose

        self._lock = threading.Lock()
        self._interp: PoseTrajectoryInterpolator | None = None
        self._last_waypoint_time: float = -float("inf")
        self._stop_evt = threading.Event()

        # Stats
        self._n_ticks = 0
        self._n_waypoints = 0
        self._n_waypoints_rejected = 0

    def _init_trajectory(self):
        """Seed the trajectory with the current camera pose."""
        T_now = self._driver.get_camera_pose_mat()
        pose6 = mat_to_pose6(T_now)
        now = time.monotonic()
        self._interp = PoseTrajectoryInterpolator(
            times=np.array([now]), poses=np.array([pose6])
        )
        self._last_waypoint_time = now

    def schedule_waypoint(self, T_world_cam: np.ndarray, target_time: float):
        """Insert a single camera-frame waypoint at target_time (monotonic)."""
        pose6 = mat_to_pose6(np.asarray(T_world_cam))
        with self._lock:
            if self._interp is None:
                return  # not started yet
            curr_time = time.monotonic()
            if target_time <= curr_time:
                self._n_waypoints_rejected += 1
                return
            self._interp = self._interp.schedule_waypoint(
                pose=pose6,
                time=target_time,
                max_pos_speed=self._max_pos_speed,
                max_rot_speed=self._max_rot_speed,
                curr_time=curr_time,
                last_waypoint_time=self._last_waypoint_time,
            )
            self._last_waypoint_time = max(
                self._last_waypoint_time, target_time)
            self._n_waypoints += 1

    def run(self):
        self._init_trajectory()
        # Set MOVE P mode once; we stream EndPoseCtrl below.
        self._driver.piper.MotionCtrl_2(0x01, 0x00, self._driver.speed_pct, 0x00)

        next_t = time.monotonic()
        last_log = next_t
        while not self._stop_evt.is_set():
            now = time.monotonic()
            with self._lock:
                interp = self._interp
            if interp is not None:
                pose6 = interp(now)
                self._driver._send_camera_pose_from_pose6(pose6)
            self._n_ticks += 1

            if self._verbose and (now - last_log) > 1.0:
                elapsed = now - last_log
                print(
                    f"[interp] {self._n_ticks / elapsed:.0f} Hz  "
                    f"waypoints: +{self._n_waypoints} "
                    f"(rejected {self._n_waypoints_rejected})",
                    flush=True,
                )
                self._n_ticks = 0
                self._n_waypoints = 0
                self._n_waypoints_rejected = 0
                last_log = now

            next_t += self._dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # fell behind — reset to now to prevent runaway catch-up
                next_t = time.monotonic()

    def stop(self):
        self._stop_evt.set()


class PiperDriver:
    def __init__(
        self,
        can_port: str = "can0",
        speed_pct: int = 100,
        tf_wrist_to_camera: np.ndarray = TF_WRIST_TO_CAMERA,
        smooth: bool = False,
        smooth_hz: int = 150,
        max_pos_speed: float = 0.5,
        max_rot_speed: float = 2.0,
        verbose_interp: bool = False,
    ):
        self.speed_pct = speed_pct
        self.smooth = smooth
        self.tf_w2c = np.array(tf_wrist_to_camera, dtype=np.float64)
        self.tf_c2w = np.linalg.inv(self.tf_w2c)

        self.piper = C_PiperInterface_V2(can_port)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        # Set MOVE P mode once; _InterpolationController also sets it on start.
        self.piper.MotionCtrl_2(0x01, 0x00, self.speed_pct, 0x00)

        self._interp_ctrl: _InterpolationController | None = None
        if smooth:
            self._interp_ctrl = _InterpolationController(
                self,
                send_hz=smooth_hz,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                verbose=verbose_interp,
            )
            self._interp_ctrl.start()

    def stop(self):
        if self._interp_ctrl is not None:
            self._interp_ctrl.stop()
            self._interp_ctrl.join(timeout=2.0)

    # ------------------------------------------------------------------ read

    def get_wrist_pose_mat(self) -> np.ndarray:
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        x = ep.X_axis * _POS_TO_M
        y = ep.Y_axis * _POS_TO_M
        z = ep.Z_axis * _POS_TO_M
        rx = ep.RX_axis * _ROT_TO_DEG
        ry = ep.RY_axis * _ROT_TO_DEG
        rz = ep.RZ_axis * _ROT_TO_DEG
        T = np.eye(4)
        T[:3, :3] = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        T[:3, 3] = [x, y, z]
        return T

    def get_camera_pose_mat(self) -> np.ndarray:
        return self.get_wrist_pose_mat() @ self.tf_w2c

    def get_camera_pose(self):
        T = self.get_camera_pose_mat()
        pos = T[:3, 3]
        rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
        return pos, rotvec, T

    # ----------------------------------------------------------------- write

    def schedule_waypoint(self, T_world_cam: np.ndarray, target_time: float):
        """UMI-style: queue a camera-frame waypoint for the interp thread."""
        if self._interp_ctrl is None:
            raise RuntimeError("schedule_waypoint requires smooth=True")
        self._interp_ctrl.schedule_waypoint(T_world_cam, target_time)

    def set_camera_pose_mat(self, T_base_cam: np.ndarray):
        """Non-smooth path: send directly. Bypasses interp thread."""
        self._send_camera_pose_mat_raw(T_base_cam)

    def _send_camera_pose_from_pose6(self, pose6: np.ndarray):
        self._send_camera_pose_mat_raw(pose6_to_mat(pose6))

    def _send_camera_pose_mat_raw(self, T_base_cam: np.ndarray):
        T_base_wrist = T_base_cam @ self.tf_c2w
        self._send_wrist_pose(T_base_wrist)

    def set_camera_pose(self, pos_m: np.ndarray, rotvec_rad: np.ndarray):
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(rotvec_rad).as_matrix()
        T[:3, 3] = pos_m
        self.set_camera_pose_mat(T)

    def set_camera_pose_euler(self, pos_m: np.ndarray, euler_deg: np.ndarray):
        T = np.eye(4)
        T[:3, :3] = R.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        T[:3, 3] = pos_m
        self.set_camera_pose_mat(T)

    def set_wrist_pose_mat(self, T_base_wrist: np.ndarray):
        self._send_wrist_pose(T_base_wrist)

    def _send_wrist_pose(self, T_base_wrist: np.ndarray):
        x, y, z = T_base_wrist[:3, 3]
        rx, ry, rz = R.from_matrix(T_base_wrist[:3, :3]).as_euler("xyz", degrees=True)
        X = round(x * _M_TO_POS)
        Y = round(y * _M_TO_POS)
        Z = round(z * _M_TO_POS)
        RX = round(rx * _DEG_TO_ROT)
        RY = round(ry * _DEG_TO_ROT)
        RZ = round(rz * _DEG_TO_ROT)
        # Note: MOVE P mode is set once at init / on interp start; not per-command.
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

    # --------------------------------------------------------------- status

    def get_arm_status(self):
        return self.piper.GetArmStatus().arm_status

    def is_target_reachable(self) -> bool:
        return self.get_arm_status().arm_status == 0

    def print_status(self):
        s = self.get_arm_status()
        print(
            f"arm_status={s.arm_status}  "
            f"motion_status={s.motion_status}  "
            f"err_status={s.err_status}"
        )


if __name__ == "__main__":
    driver = PiperDriver(speed_pct=30, smooth=False)
    driver.set_camera_pose_euler([0.35, 0.11, 0.25], [0.0, 0.0, 0.0])
    driver.print_status()
