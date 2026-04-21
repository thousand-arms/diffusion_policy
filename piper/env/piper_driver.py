"""Thin wrapper around the Piper SDK for camera-frame pose control.

The policy predicts and observes in the **camera frame** (mounted on the
wrist). This driver converts between that frame and the SDK's wrist-frame
commands/feedback, handling:
  - TF_WRIST_TO_CAMERA rigid transform
  - SDK unit scaling (0.001 mm / 0.001 deg)
  - Extrinsic XYZ euler convention

Usage:
    driver = PiperDriver()
    pos, rotvec, T = driver.get_camera_pose()
    driver.set_camera_pose(pos, rotvec)
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from piper_sdk import C_PiperInterface_V2

# Piper SDK units
_POS_TO_M = 1e-6  # 0.001 mm -> m  (feedback)
_ROT_TO_DEG = 1e-3  # 0.001 deg -> deg (feedback)
_M_TO_POS = 1_000_000  # m -> 0.001 mm (command)
_DEG_TO_ROT = 1_000  # deg -> 0.001 deg (command)

TF_WRIST_TO_CAMERA = np.array(
    [
        [-0.0000000, 0.7071068, -0.7071068, 0.01042],
        [-0.7071068, -0.5000000, -0.5000000, 0.18364],
        [-0.7071068, 0.5000000, 0.5000000, 0.10064],
        [0, 0, 0, 1],
    ]
)


class PiperDriver:
    def __init__(
        self,
        can_port: str = "can0",
        speed_pct: int = 50,
        tf_wrist_to_camera: np.ndarray = TF_WRIST_TO_CAMERA,
    ):
        self.speed_pct = speed_pct
        self.tf_w2c = np.array(tf_wrist_to_camera, dtype=np.float64)
        self.tf_c2w = np.linalg.inv(self.tf_w2c)

        self.piper = C_PiperInterface_V2(can_port)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        self.piper.MotionCtrl_2(0x01, 0x00, self.speed_pct, 0x00)

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

    def set_camera_pose_mat(self, T_base_cam: np.ndarray):
        """Command the arm so the camera arrives at T_base_cam."""
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
    driver = PiperDriver(speed_pct=30)
    # pos, rotvec, T = driver.get_camera_pose()
    # euler_deg = R.from_rotvec(rotvec).as_euler("xyz", degrees=True)
    # print(f"Camera pos (m):     {pos}")
    # print(f"Camera euler (deg): {euler_deg}")
    # print(f"Camera T:\n{T}")
    driver.set_camera_pose_euler([0.35, 0.11, 0.25], [0.0, 0.0, 0.0])
    driver.print_status()
