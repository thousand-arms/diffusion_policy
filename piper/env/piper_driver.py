"""Thin wrapper around the Piper SDK for camera-frame pose control.

The policy predicts and observes in the **camera frame** (mounted on the
wrist). This driver converts between that frame and the SDK's wrist-frame
commands/feedback, handling:
  - TF_WRIST_TO_CAMERA rigid transform
  - SDK unit scaling (0.001 mm / 0.001 deg)
  - Extrinsic XYZ euler convention
  - Optional trajectory interpolation for smooth motion

Usage:
    driver = PiperDriver()
    pos, rotvec, T = driver.get_camera_pose()
    driver.set_camera_pose(pos, rotvec)
"""

import threading
import time

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

from piper_sdk import C_PiperInterface_V2

# Piper SDK units
_POS_TO_M = 1e-6  # 0.001 mm -> m  (feedback)
_ROT_TO_DEG = 1e-3  # 0.001 deg -> deg (feedback)
_M_TO_POS = 1_000_000  # m -> 0.001 mm (command)
_DEG_TO_ROT = 1_000  # deg -> 0.001 deg (command)

TF_WRIST_TO_CAMERA = np.array(
    # Previous TF (before re-calibration):
    # [
    #     [-0.0000000, 0.7071068, -0.7071068, 0.01042],
    #     [-0.7071068, -0.5000000, -0.5000000, 0.18364],
    #     [-0.7071068, 0.5000000, 0.5000000, 0.10064],
    #     [0, 0, 0, 1],
    # ]
    [
        [-0.0000000, 0.7071068, -0.7071068, -0.07958],
        [-0.7071068, -0.5000000, -0.5000000, 0.18364],
        [-0.7071068, 0.5000000, 0.5000000, 0.10064],
        [0, 0, 0, 1],
    ]
)


class _TrajectoryInterpolator:
    """Piecewise-linear position + SLERP rotation interpolator.

    Accepts waypoints as (timestamp, T_cam_4x4) pairs. Interpolates
    the camera pose at any query time within the waypoint range.
    """

    def __init__(self, times, poses_4x4):
        """
        Args:
            times: (N,) monotonic timestamps.
            poses_4x4: (N, 4, 4) camera poses in world frame.
        """
        self.times = np.array(times, dtype=np.float64)
        positions = np.array([p[:3, 3] for p in poses_4x4])
        rotations = R.from_matrix([p[:3, :3] for p in poses_4x4])

        self._pos_interp = interp1d(
            self.times, positions, axis=0,
            kind="linear", fill_value="extrapolate",
        )
        self._rot_slerp = Slerp(self.times, rotations)

    @property
    def t_start(self):
        return self.times[0]

    @property
    def t_end(self):
        return self.times[-1]

    def __call__(self, t):
        """Interpolate pose at time t. Returns (4, 4) camera pose."""
        # Clamp to valid range
        t_clamped = np.clip(t, self.times[0], self.times[-1])
        pos = self._pos_interp(t_clamped)
        rot = self._rot_slerp(t_clamped)
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = pos
        return T


class _InterpolationController(threading.Thread):
    """Background thread that interpolates between waypoints and sends
    smooth commands to the Piper at a fixed high frequency.

    New waypoints are appended to the existing trajectory (not replaced),
    so the interpolation is continuous across prediction boundaries.
    """

    def __init__(self, driver, send_hz=50):
        super().__init__(daemon=True)
        self._driver = driver
        self._dt = 1.0 / send_hz
        self._lock = threading.Lock()
        self._times = []      # accumulated timestamps
        self._poses = []      # accumulated 4x4 poses
        self._interp = None
        self._stop = threading.Event()

    def schedule_waypoints(self, times, poses_4x4):
        """Append new waypoints to the trajectory.

        Only appends waypoints whose timestamp is after the current
        trajectory end to maintain monotonicity. Trims old waypoints
        that are more than 1s in the past.

        Args:
            times: (N,) monotonic timestamps.
            poses_4x4: list/array of (4, 4) world-frame camera poses.
        """
        now = time.monotonic()
        with self._lock:
            # Trim old waypoints (keep last 1s for interpolation context)
            cutoff = now - 1.0
            while self._times and self._times[0] < cutoff:
                self._times.pop(0)
                self._poses.pop(0)

            # Find the last timestamp in current trajectory
            last_t = self._times[-1] if self._times else -float("inf")

            # Append only new waypoints (after current trajectory end)
            for i in range(len(times)):
                if times[i] > last_t:
                    self._times.append(float(times[i]))
                    self._poses.append(np.array(poses_4x4[i], dtype=np.float64))

            # Rebuild interpolator if we have at least 2 points
            if len(self._times) >= 2:
                self._interp = _TrajectoryInterpolator(
                    self._times, self._poses)
            elif len(self._times) == 1:
                # Single point — hold position
                self._interp = None

    def run(self):
        next_t = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()

            with self._lock:
                interp = self._interp

            if interp is not None:
                T_cam = interp(now)
                self._driver._set_camera_pose_mat_raw(T_cam)

            next_t += self._dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()

    def stop(self):
        self._stop.set()


class PiperDriver:
    def __init__(
        self,
        can_port: str = "can0",
        speed_pct: int = 100,
        tf_wrist_to_camera: np.ndarray = TF_WRIST_TO_CAMERA,
        smooth: bool = False,
        smooth_hz: int = 50,
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
        self.piper.MotionCtrl_2(0x01, 0x00, self.speed_pct, 0x00)

        # Interpolation controller (started lazily on first schedule_waypoints)
        self._interp_ctrl = None
        if smooth:
            self._interp_ctrl = _InterpolationController(self, send_hz=smooth_hz)
            self._interp_ctrl.start()

    # ------------------------------------------------------------------ read

    def get_wrist_pose_mat(self) -> np.ndarray:
        """T_base_wrist as 4x4."""
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
        """T_base_camera as 4x4."""
        return self.get_wrist_pose_mat() @ self.tf_w2c

    def get_camera_pose(self):
        """Camera pose as (position_m, axis_angle_rad, T_base_cam).

        position_m:      (3,) float64
        axis_angle_rad:  (3,) float64  (scipy rotation vector)
        T_base_cam:      (4,4) float64
        """
        T = self.get_camera_pose_mat()
        pos = T[:3, 3]
        rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
        return pos, rotvec, T

    # ----------------------------------------------------------------- write

    def schedule_waypoints(self, times, poses_4x4):
        """Schedule a batch of future camera-frame waypoints for smooth execution.

        Only available when smooth=True. The interpolation controller will
        smoothly interpolate between them at high frequency.

        Args:
            times: (N,) monotonic timestamps for each waypoint.
            poses_4x4: list of (4, 4) world-frame camera poses.
        """
        if self._interp_ctrl is None:
            raise RuntimeError("schedule_waypoints requires smooth=True")
        self._interp_ctrl.schedule_waypoints(times, poses_4x4)

    def set_camera_pose_mat(self, T_base_cam: np.ndarray):
        """Command the arm so the camera arrives at T_base_cam.

        Sends directly (bypasses interpolation controller).
        """
        self._set_camera_pose_mat_raw(T_base_cam)

    def _set_camera_pose_mat_raw(self, T_base_cam: np.ndarray):
        """Internal: convert camera pose to wrist and send via CAN."""
        T_base_wrist = T_base_cam @ self.tf_c2w
        self._send_wrist_pose(T_base_wrist)

    def set_camera_pose(self, pos_m: np.ndarray, rotvec_rad: np.ndarray):
        """Command via position (m) + axis-angle (rad)."""
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(rotvec_rad).as_matrix()
        T[:3, 3] = pos_m
        self.set_camera_pose_mat(T)

    def set_camera_pose_euler(self, pos_m: np.ndarray, euler_deg: np.ndarray):
        """Command via position (m) + extrinsic XYZ euler (deg). Handy for
        manual testing since the Piper SDK uses the same euler convention."""
        T = np.eye(4)
        T[:3, :3] = R.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        T[:3, 3] = pos_m
        self.set_camera_pose_mat(T)

    def set_wrist_pose_mat(self, T_base_wrist: np.ndarray):
        """Command directly in the wrist frame."""
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
        self.piper.MotionCtrl_2(0x01, 0x00, self.speed_pct, 0x00)
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

    # --------------------------------------------------------------- status

    def get_arm_status(self):
        return self.piper.GetArmStatus().arm_status

    def is_target_reachable(self) -> bool:
        """Check if the arm accepted the last target. Returns False if
        arm_status reports an IK/limit error (no-sol, singularity,
        joint-limit, collision, etc.)."""
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
