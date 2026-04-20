"""Test camera frame axes by moving 1cm along each axis.

Press 'x' to move +1cm in camera X (should move right)
Press 'y' to move +1cm in camera Y (should move down)
Press 'z' to move +1cm in camera Z (should move forward)
Press 'r' to return to the starting position
Press 'q' to quit
"""

import sys
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from piper.env.piper_driver import PiperDriver

STEP_M = 0.01  # 1 cm

driver = PiperDriver(can_port="can0", speed_pct=20)
time.sleep(0.5)

# record starting camera pose
start_T = driver.get_camera_pose_mat().copy()
current_T = start_T.copy()

print(f"Starting camera pos (m): {start_T[:3, 3]}")
print(f"Starting wrist pos (m):  {driver.get_wrist_pose_mat()[:3, 3]}")
print()
print("Press x/y/z to move +1cm along that camera axis")
print("Press r to return to start")
print("Press q to quit")
print()

while True:
    key = input("> ").strip().lower()
    if key == "q":
        break
    elif key == "r":
        current_T = start_T.copy()
        driver.set_camera_pose_mat(current_T)
        time.sleep(0.5)
        actual = driver.get_camera_pose_mat()
        print(f"[reset] camera pos: {actual[:3, 3]}")
        driver.print_status()
    elif key in ("x", "y", "z"):
        axis = {"x": 0, "y": 1, "z": 2}[key]
        # Move along camera axis: the axis direction in world frame
        # is the corresponding column of the rotation matrix
        direction = current_T[:3, axis]
        new_T = current_T.copy()
        new_T[:3, 3] += direction * STEP_M
        driver.set_camera_pose_mat(new_T)
        current_T = new_T
        time.sleep(0.5)
        actual = driver.get_camera_pose_mat()
        delta = (actual[:3, 3] - start_T[:3, 3]) * 1000
        print(
            f"[+{key}] camera pos: {actual[:3, 3]}  delta from start: ({delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f}) mm"
        )
        driver.print_status()
    else:
        print("Unknown key. Use x/y/z/r/q")
