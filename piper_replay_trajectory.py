import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from piper_sdk import *

CSV_PATH = "/mnt/thousand-arms-data/data/trace/square_3d/2026-04-07_13-51-40/camera_trajectory.csv"

INDEX = 200

R_ARUCO_TO_PIPER = R.from_euler("z", 90, degrees=True)
T_ARUCO_TO_PIPER = np.array([-0.590, 0.330, 0.0])

TF_WRIST_TO_CAMERA = np.array(
    [
        [0.7071, 0.0, -0.7071, -0.0796],
        [-0.5000, 0.7071, -0.5000, 0.1836],
        [0.5000, 0.7071, 0.5000, 0.1006],
        [0.0, 0.0, 0.0, 1.0000],
    ]
)


def setup_piper(can_port="can0"):
    piper = C_PiperInterface_V2(can_port)
    piper.ConnectPort()
    while not piper.EnablePiper():
        time.sleep(0.01)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    return piper


def set_end_effector_pose(piper, pose, cam=True):
    # pose: [x, y, z (m), rx, ry, rz (deg)]
    # if cam=True, pose is for the camera; convert to wrist pose via TF_WRIST_TO_CAMERA.
    # if cam=False, pose is for the wrist directly.
    if cam:
        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = R.from_euler("xyz", pose[3:6], degrees=True).as_matrix()
        T_base_cam[:3, 3] = pose[:3]
        T_base_wrist = T_base_cam @ np.linalg.inv(TF_WRIST_TO_CAMERA)
        x, y, z = T_base_wrist[:3, 3]
        rx, ry, rz = R.from_matrix(T_base_wrist[:3, :3]).as_euler("xyz", degrees=True)
    else:
        x, y, z, rx, ry, rz = pose

    # SDK expects 0.001 mm and 0.001 deg
    pos_factor = 1_000_000
    rot_factor = 1_000
    X, Y, Z = round(x * pos_factor), round(y * pos_factor), round(z * pos_factor)
    RX, RY, RZ = round(rx * rot_factor), round(ry * rot_factor), round(rz * rot_factor)
    print(X, Y, Z, RX, RY, RZ)
    piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
    piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)


def print_arm_diagnostics(piper):
    status = piper.GetArmStatus().arm_status
    # print(f"ctrl_mode:     {status.ctrl_mode}")  # want 0x01 (CAN)
    print(
        f"arm_status:    {status.arm_status}"
    )  # want 0x00 (NORMAL); 0x02=no-sol, 0x03=singularity, 0x04=joint-limit, 0x07=collision
    # print(f"mode_feed:     {status.mode_feed}")  # want 0x00 (MOVE P)
    # print(f"motion_status: {status.motion_status}")  # 0x00 reached, 0x01 unreachable
    # print(f"err_status:    {status.err_status}")
    # print(f"enable:        {piper.GetArmEnableStatus()}")

    # low = piper.GetArmLowSpdInfoMsgs()
    # for i in range(1, 7):
    #     m = getattr(low, f"motor_{i}", None)
    #     if m is None or not hasattr(m, "foc_status"):
    #         continue
    #     f = m.foc_status
    #     print(
    #         f"motor{i} collision={f.collision_status} "
    #         f"stall={f.stall_status} overcur={f.driver_overcurrent} "
    #         f"overtemp={f.driver_overheating} err={f.driver_error_status}"
    #     )


def aruco_pose_to_piper(xyz, quat_xyzw):
    xyz_piper = R_ARUCO_TO_PIPER.apply(xyz) + T_ARUCO_TO_PIPER
    quat_piper = (R_ARUCO_TO_PIPER * R.from_quat(quat_xyzw)).as_quat()
    return xyz_piper, quat_piper


def main():
    # df = pd.read_csv(CSV_PATH)
    # print(f"shape: {df.shape}")
    # print(f"columns: {list(df.columns)}")
    # print(df.head())
    # print(df.describe())

    # row = df.iloc[INDEX]
    # xyz_aruco = np.array([row["x"], row["y"], row["z"]])
    # quat_aruco = np.array([row["q_x"], row["q_y"], row["q_z"], row["q_w"]])

    # xyz_piper, quat_piper = aruco_pose_to_piper(xyz_aruco, quat_aruco)

    # print(f"\n[index {INDEX}]")
    # print(f"aruco xyz (m):   {xyz_aruco}")
    # print(f"aruco quat xyzw: {quat_aruco}")
    # print(f"piper xyz (m):   {xyz_piper}")
    # print(f"piper quat xyzw: {quat_piper}")

    piper = setup_piper()

    # known-reachable wrist pose (matches arm's resting feedback: 57mm, 0, 215mm, 0, 85deg, 0)
    # target_pose = [0.35, 0.11, 0.25, 25.0, 85.0, 0.0]  # rx
    target_pose = [0.35, 0.11, 0.25, 0.0, 85.0, -20.0]  # rz

    # target_pose = [0.35, 0.11, 0.25, 0.0, 85.0, 0.0]  # ry
    start = time.time()
    while True:
        # cam = int((time.time() - start) / 5) % 2 == 0
        cam = True
        print(f"cam={cam}")
        print(piper.GetArmEndPoseMsgs())
        set_end_effector_pose(piper, target_pose, cam=cam)
        print_arm_diagnostics(piper)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
