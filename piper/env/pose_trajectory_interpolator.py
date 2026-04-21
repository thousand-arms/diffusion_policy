"""Pose trajectory interpolator with smart trim-and-insert scheduling.

Ported from UMI (umi/common/pose_trajectory_interpolator.py). Maintains a
piecewise-linear + Slerp trajectory and supports scheduling new waypoints
that correctly override any overlapping future plan while staying
C0-continuous at the current time.

When a new waypoint arrives:
  - If target_time > last_waypoint_time: append (extend future plan).
  - If target_time <= last_waypoint_time: trim trajectory to curr_time
    and insert — the new waypoint replaces the overlapping portion.

max_pos_speed / max_rot_speed rate-limit the interpolator: if a waypoint
is too far for the scheduled time, duration is extended so the arm
never teleports between batches.
"""

from typing import Union
import numbers

import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()


def pose_distance(start_pose, end_pose):
    """start_pose, end_pose: (6,) = pos(3) + rotvec(3)."""
    start_pose = np.asarray(start_pose)
    end_pose = np.asarray(end_pose)
    pos_dist = np.linalg.norm(end_pose[:3] - start_pose[:3])
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist


def mat_to_pose6(T: np.ndarray) -> np.ndarray:
    """(4,4) -> (6,) = pos(3) + rotvec(3)."""
    pose = np.zeros(6)
    pose[:3] = T[:3, 3]
    pose[3:] = st.Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return pose


def pose6_to_mat(pose: np.ndarray) -> np.ndarray:
    """(6,) -> (4,4)."""
    T = np.eye(4)
    T[:3, :3] = st.Rotation.from_rotvec(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


class PoseTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        """times: (N,), poses: (N, 6) = pos(3) + rotvec(3)."""
        assert len(times) >= 1
        assert len(poses) == len(times)
        times = np.asarray(times, dtype=np.float64)
        poses = np.asarray(poses, dtype=np.float64)

        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1]), "times must be non-decreasing"
            pos = poses[:, :3]
            rot = st.Rotation.from_rotvec(poses[:, 3:])
            self.pos_interp = si.interp1d(times, pos, axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)

    @property
    def times(self) -> np.ndarray:
        return self._times if self.single_step else self.pos_interp.x

    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        n = len(self.times)
        poses = np.zeros((n, 6))
        poses[:, :3] = self.pos_interp.y
        poses[:, 3:] = self.rot_interp(self.times).as_rotvec()
        return poses

    def trim(self, start_t: float, end_t: float) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)

    def schedule_waypoint(
        self,
        pose,
        time,
        max_pos_speed=np.inf,
        max_rot_speed=np.inf,
        curr_time=None,
        last_waypoint_time=None,
    ) -> "PoseTrajectoryInterpolator":
        """Insert a new waypoint, trimming overlapping future plan."""
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # new waypoint is in the past — ignore
                return self
            start_time = max(curr_time, start_time)
            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        assert start_time <= end_time
        assert end_time <= time

        trimmed = self.trim(start_time, end_time)

        duration = time - end_time
        end_pose = trimmed(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_wp_time = end_time + duration

        times = np.append(trimmed.times, [last_wp_time], axis=0)
        poses = np.append(trimmed.poses, [np.asarray(pose)], axis=0)

        return PoseTrajectoryInterpolator(times, poses)

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        pose = np.zeros((len(t), 6))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)
            pose[:, :3] = self.pos_interp(t)
            pose[:, 3:] = self.rot_interp(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose
