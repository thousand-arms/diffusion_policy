"""Capture a camera pose by hand-positioning the arm, save it as the eval reset pose.

Workflow:
  1. Connect to the Piper and enter drag-teach mode (gravity-comp — arm can
     be moved by hand).
  2. User positions the arm to the desired reset pose.
  3. User presses Enter; the current camera pose is written to
     piper/env/default_cam_pose.yaml as pos (m) + axis-angle (rad).

Usage:
    python -m piper.scripts.set_default_camera_pose
"""

import pathlib
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from piper.env.piper_driver import PiperDriver

DEFAULT_POSE_PATH = ROOT / "piper" / "env" / "default_cam_pose.yaml"

# MotionCtrl_1 drag-teach control values (see piper_sdk MotionCtrl_1 docstring).
_TEACH_ENTER = 0x01
_TEACH_EXIT = 0x02


def main():
    print("[driver] connecting...")
    driver = PiperDriver(smooth=False)

    print("[teach] entering drag-teach mode — arm is hand-movable now.")
    driver.piper.MotionCtrl_1(0x00, 0x00, _TEACH_ENTER)
    time.sleep(0.2)

    try:
        input("[input] move arm to desired pose, then press Enter to save ...")
        pos, rotvec, _ = driver.get_camera_pose()
    finally:
        driver.piper.MotionCtrl_1(0x00, 0x00, _TEACH_EXIT)
        time.sleep(0.2)
        print("[teach] exited drag-teach mode.")

    print(f"[pose] pos_m={pos.tolist()}")
    print(f"[pose] rot_axis_angle_rad={rotvec.tolist()}")

    DEFAULT_POSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_POSE_PATH.open("w") as f:
        yaml.safe_dump(
            {
                "pos_m": [float(x) for x in pos],
                "rot_axis_angle_rad": [float(x) for x in rotvec],
            },
            f,
            sort_keys=False,
        )
    print(f"[save] wrote {DEFAULT_POSE_PATH}")


if __name__ == "__main__":
    main()
