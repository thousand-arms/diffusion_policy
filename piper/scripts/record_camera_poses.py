"""Record a default camera pose + keyframes by hand-positioning the arm.

Workflow:
  1. Connect to the Piper and enter drag-teach mode (gravity-comp — arm
     is hand-movable).
  2. First Enter: records the DEFAULT pose.
  3. Each subsequent Enter: records the next KEYFRAME.
  4. Type 'q' (or Ctrl+C) and press Enter to save + exit.

Saves to piper/env/camera_poses.yaml as pos (m) + axis-angle (rad).

Usage:
    python -m piper.scripts.record_camera_poses
"""

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from piper.env.camera_poses import CAMERA_POSES_PATH, save_camera_poses
from piper.env.piper_driver import PiperDriver

# MotionCtrl_1 drag-teach control values (see piper_sdk docstring).
_TEACH_ENTER = 0x01
_TEACH_EXIT = 0x02
_QUIT_WORDS = {"q", "quit", "done", "exit"}


def main():
    print("[driver] connecting...")
    driver = PiperDriver(smooth=False)

    print("[teach] entering drag-teach mode — arm is hand-movable now.")
    driver.piper.MotionCtrl_1(0x00, 0x00, _TEACH_ENTER)
    time.sleep(0.2)

    default_mat = None
    keyframes = []
    try:
        while True:
            if default_mat is None:
                prompt = "[input] move arm to DEFAULT pose, press Enter to record > "
            else:
                prompt = (
                    f"[input] move arm to KEYFRAME {len(keyframes)+1}, "
                    f"press Enter to record (or 'q' + Enter to save & quit) > "
                )
            line = input(prompt)
            if line.strip().lower() in _QUIT_WORDS:
                break
            _, _, T = driver.get_camera_pose()
            if default_mat is None:
                default_mat = T
                print(f"[recorded] default: pos={T[:3, 3].tolist()}")
            else:
                keyframes.append(T)
                print(f"[recorded] keyframe {len(keyframes)}: pos={T[:3, 3].tolist()}")
    except (KeyboardInterrupt, EOFError):
        print()  # newline after ^C / ^D
    finally:
        driver.piper.MotionCtrl_1(0x00, 0x00, _TEACH_EXIT)
        time.sleep(0.2)
        print("[teach] exited drag-teach mode.")

    if default_mat is None:
        print("[save] no default pose recorded — nothing to save.")
        return

    save_camera_poses(default_mat, keyframes, CAMERA_POSES_PATH)
    print(
        f"[save] wrote {CAMERA_POSES_PATH}: "
        f"1 default + {len(keyframes)} keyframe(s)"
    )


if __name__ == "__main__":
    main()
