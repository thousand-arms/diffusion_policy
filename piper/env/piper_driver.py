"""Thin wrapper around the Piper SDK for camera-frame pose control.

The policy predicts and observes in the camera frame (mounted on the wrist).
This driver converts between that frame and the SDK's wrist-frame commands,
handling the TF_WRIST_TO_CAMERA rigid transform, SDK unit scaling
(0.001 mm / 0.001 deg), and extrinsic XYZ euler convention.

With smooth=True, a background thread runs a PoseTrajectoryInterpolator
at smooth_hz and streams EndPoseCtrl (MOVE P) commands. Callers submit
waypoints via schedule_waypoint(); new waypoints trim-and-insert into the
trajectory so overlapping predictions smoothly override the future plan.

Usage:
    driver = PiperDriver(smooth=True, smooth_hz=150)
    batch_id = driver.new_batch()
    driver.schedule_waypoint(T_world_cam, target_time, batch_id)
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

# Piper SDK unit conversions
_POS_TO_M = 1e-6           # 0.001 mm -> m  (feedback)
_ROT_TO_DEG = 1e-3         # 0.001 deg -> deg (feedback)
_M_TO_POS = 1_000_000      # m -> 0.001 mm (command)
_DEG_TO_ROT = 1_000        # deg -> 0.001 deg (command)

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

    Every tick: evaluate the interpolator at t_now and send the pose via
    EndPoseCtrl. New waypoints arrive via schedule_waypoint() and
    trim-and-insert into the trajectory.
    """

    def __init__(
        self,
        driver: "PiperDriver",
        send_hz: int,
        max_pos_speed: float,   # m/s
        max_rot_speed: float,   # rad/s
        verbose: bool,
        record: bool,
    ):
        super().__init__(daemon=True, name="PiperInterpController")
        self._driver = driver
        self._dt = 1.0 / send_hz
        self._max_pos_speed = max_pos_speed
        self._max_rot_speed = max_rot_speed
        self._verbose = verbose
        self._record = record

        self._lock = threading.Lock()
        self._interp: PoseTrajectoryInterpolator | None = None
        self._last_waypoint_time: float = -float("inf")
        self._batch_counter: int = 0
        self._stop_evt = threading.Event()

        # per-second verbose stats
        self._n_ticks = 0
        self._n_waypoints = 0
        self._n_rejected = 0

        # recording: populated iff record=True
        self.sent_log: list = []          # [(t, pose6)]
        self.waypoint_log: list = []      # [(target_time, pose6, batch_id)]

    # -- public API --

    def new_batch(self) -> int:
        with self._lock:
            self._batch_counter += 1
            return self._batch_counter

    def schedule_waypoint(
        self, T_world_cam: np.ndarray, target_time: float, batch_id: int = -1
    ):
        pose6 = mat_to_pose6(np.asarray(T_world_cam))
        with self._lock:
            if self._interp is None:
                return
            curr_time = time.monotonic()
            if target_time <= curr_time:
                self._n_rejected += 1
                return
            self._interp = self._interp.schedule_waypoint(
                pose=pose6,
                time=target_time,
                max_pos_speed=self._max_pos_speed,
                max_rot_speed=self._max_rot_speed,
                curr_time=curr_time,
                last_waypoint_time=self._last_waypoint_time,
            )
            self._last_waypoint_time = max(self._last_waypoint_time, target_time)
            self._n_waypoints += 1
            if self._record:
                self.waypoint_log.append(
                    (float(target_time), pose6.copy(), int(batch_id))
                )

    def stop(self):
        self._stop_evt.set()

    # -- thread main --

    def run(self):
        # seed trajectory with current pose
        now = time.monotonic()
        seed = mat_to_pose6(self._driver.get_camera_pose_mat())
        self._interp = PoseTrajectoryInterpolator(
            times=np.array([now]), poses=np.array([seed])
        )
        self._last_waypoint_time = now

        # set MOVE P mode once; we stream EndPoseCtrl below
        self._driver.piper.MotionCtrl_2(0x01, 0x00, self._driver.speed_pct, 0x00)

        next_t = time.monotonic()
        last_log = next_t
        while not self._stop_evt.is_set():
            now = time.monotonic()
            with self._lock:
                interp = self._interp
            if interp is not None:
                pose6 = interp(now)
                self._driver._send_camera_pose6(pose6)
                if self._record:
                    self.sent_log.append((now, pose6.copy()))
            self._n_ticks += 1

            if self._verbose and (now - last_log) > 1.0:
                elapsed = now - last_log
                print(
                    f"[interp] {self._n_ticks / elapsed:.0f} Hz  "
                    f"waypoints: +{self._n_waypoints} (rejected {self._n_rejected})",
                    flush=True,
                )
                self._n_ticks = 0
                self._n_waypoints = 0
                self._n_rejected = 0
                last_log = now

            next_t += self._dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # fell behind; reset


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
        record_interp: bool = False,
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

        self._interp_ctrl: _InterpolationController | None = None
        if smooth:
            self._interp_ctrl = _InterpolationController(
                self,
                send_hz=smooth_hz,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                verbose=verbose_interp,
                record=record_interp,
            )
            self._interp_ctrl.start()

    def stop(self):
        if self._interp_ctrl is not None:
            self._interp_ctrl.stop()
            self._interp_ctrl.join(timeout=2.0)

    # -- smooth-mode commands --

    def new_batch(self) -> int:
        """Bump the batch counter; used to tag waypoint_log entries."""
        self._require_smooth()
        return self._interp_ctrl.new_batch()

    def schedule_waypoint(
        self, T_world_cam: np.ndarray, target_time: float, batch_id: int = -1
    ):
        """Queue a camera-frame waypoint for the interp thread."""
        self._require_smooth()
        self._interp_ctrl.schedule_waypoint(T_world_cam, target_time, batch_id)

    def get_recorded_logs(self):
        """Return (sent_log, waypoint_log) if record_interp was enabled.

        sent_log:     [(t, pose6)] — pose commanded to the arm at send_hz.
        waypoint_log: [(target_time, pose6, batch_id)] — raw scheduled waypoints.
        """
        if self._interp_ctrl is None:
            return [], []
        return list(self._interp_ctrl.sent_log), list(self._interp_ctrl.waypoint_log)

    def _require_smooth(self):
        if self._interp_ctrl is None:
            raise RuntimeError("requires smooth=True")

    # -- direct commands (bypass interp thread) --

    def set_camera_pose_mat(self, T_base_cam: np.ndarray):
        self._send_camera_pose_mat(T_base_cam)

    def set_camera_pose(self, pos_m: np.ndarray, rotvec_rad: np.ndarray):
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(rotvec_rad).as_matrix()
        T[:3, 3] = pos_m
        self._send_camera_pose_mat(T)

    def set_camera_pose_euler(self, pos_m: np.ndarray, euler_deg: np.ndarray):
        T = np.eye(4)
        T[:3, :3] = R.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        T[:3, 3] = pos_m
        self._send_camera_pose_mat(T)

    # -- feedback --

    def get_wrist_pose_mat(self) -> np.ndarray:
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        pos = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis]) * _POS_TO_M
        euler_deg = np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis]) * _ROT_TO_DEG
        T = np.eye(4)
        T[:3, :3] = R.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        T[:3, 3] = pos
        return T

    def get_camera_pose_mat(self) -> np.ndarray:
        return self.get_wrist_pose_mat() @ self.tf_w2c

    def get_camera_pose(self):
        T = self.get_camera_pose_mat()
        pos = T[:3, 3]
        rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
        return pos, rotvec, T

    def get_arm_status(self):
        """Full status message (arm_status, motion_status, err_status, ...)."""
        return self.piper.GetArmStatus().arm_status

    def get_arm_status_code(self) -> int:
        """Just the top-level arm_status byte (0 = OK, 0x04 = pos exceeds limit, ...)."""
        s = self.get_arm_status()
        return int(getattr(s, "arm_status", s))

    def print_status(self):
        s = self.get_arm_status()
        print(
            f"arm_status={s.arm_status}  "
            f"motion_status={s.motion_status}  "
            f"err_status={s.err_status}"
        )

    # -- internal send path --

    def _send_camera_pose6(self, pose6: np.ndarray):
        self._send_camera_pose_mat(pose6_to_mat(pose6))

    def _send_camera_pose_mat(self, T_base_cam: np.ndarray):
        T_base_wrist = T_base_cam @ self.tf_c2w
        xyz = T_base_wrist[:3, 3]
        rxryrz = R.from_matrix(T_base_wrist[:3, :3]).as_euler("xyz", degrees=True)
        self.piper.EndPoseCtrl(
            round(xyz[0] * _M_TO_POS),
            round(xyz[1] * _M_TO_POS),
            round(xyz[2] * _M_TO_POS),
            round(rxryrz[0] * _DEG_TO_ROT),
            round(rxryrz[1] * _DEG_TO_ROT),
            round(rxryrz[2] * _DEG_TO_ROT),
        )


if __name__ == "__main__":
    driver = PiperDriver(speed_pct=30, smooth=False)
    driver.set_camera_pose_euler([0.35, 0.11, 0.25], [0.0, 0.0, 0.0])
    driver.print_status()
