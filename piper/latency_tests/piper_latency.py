"""Measure Piper end-effector execution latency.

Sends a sine wave on the X axis of TCP pose (UMI uses spacemouse teleop;
a clean sine has a sharper correlation peak). Logs both the commanded
trajectory and the actual feedback. Cross-correlates the two signals to
recover lag = execution latency. Methodology follows UMI's
calibrate_robot_latency.py + latency_util.get_latency.

Architecture:
  - Main thread: command loop at COMMAND_HZ, sends EndPoseCtrl with
    X = X_base + A * sin(2*pi*f*t)
  - Worker thread: feedback loop at FEEDBACK_HZ, polls GetArmEndPoseMsgs

After the run, both buffers are passed to get_latency(). A 3-panel plot
is saved (cross-correlation curve, raw signals, aligned signals).
"""

import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_replay_trajectory import set_end_effector_pose, setup_piper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latency_util import get_latency


# Base pose [x(m), y(m), z(m), rx(deg), ry(deg), rz(deg)] -- known reachable.
# Z=0.255 m + RY=85deg keeps the wrist-mounted camera holder clear of the
# table. Sine sweeps X around this center; Z stays fixed.
BASE_POSE = [0.357, 0.0, 0.255, 0.0, 85.0, 0.0]

# Sine stimulus on X axis
AMPLITUDE_M = 0.03
FREQUENCY_HZ = 0.5

# Loop rates
COMMAND_HZ = 30
FEEDBACK_HZ = 200
DURATION_S = 10.0
SETTLE_S = 2.0  # let arm reach base pose before sweeping

PLOT_PATH = Path(__file__).resolve().parent / "piper_latency_plot.png"


class FeedbackWorker(threading.Thread):
    def __init__(self, piper, hz):
        super().__init__(daemon=True)
        self.piper = piper
        self.dt = 1.0 / hz
        self.stop_event = threading.Event()
        self.t = []
        self.x = []  # X axis in meters

    def run(self):
        import traceback
        try:
            next_t = time.time()
            while not self.stop_event.is_set():
                msg = self.piper.GetArmEndPoseMsgs()
                if msg is None:
                    continue
                # Piper SDK wraps the pose in .end_pose on some versions;
                # fall back to attribute access on the top-level msg.
                pose = getattr(msg, "end_pose", msg)
                now = time.time()
                self.x.append(pose.X_axis * 1e-6)  # 0.001 mm -> m
                self.t.append(now)
                next_t += self.dt
                sleep = next_t - time.time()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.time()
        except Exception:
            traceback.print_exc()


def main():
    piper = setup_piper()

    print(f"Moving to base pose {BASE_POSE} ...")
    set_end_effector_pose(piper, BASE_POSE, cam=False)
    time.sleep(SETTLE_S)

    fb = FeedbackWorker(piper, FEEDBACK_HZ)
    fb.start()

    cmd_t, cmd_x = [], []
    print(
        f"Sweep: A={AMPLITUDE_M * 1000:.0f}mm  f={FREQUENCY_HZ}Hz  "
        f"dur={DURATION_S}s  cmd@{COMMAND_HZ}Hz  fb@{FEEDBACK_HZ}Hz"
    )

    cmd_dt = 1.0 / COMMAND_HZ
    t0 = time.time()
    next_cmd = t0
    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed > DURATION_S:
            break
        x = BASE_POSE[0] + AMPLITUDE_M * np.sin(2 * np.pi * FREQUENCY_HZ * elapsed)
        pose = [x, *BASE_POSE[1:]]
        set_end_effector_pose(piper, pose, cam=False)
        cmd_t.append(now)
        cmd_x.append(x)
        next_cmd += cmd_dt
        sleep = next_cmd - time.time()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_cmd = time.time()

    set_end_effector_pose(piper, BASE_POSE, cam=False)
    fb.stop_event.set()
    fb.join(timeout=1.0)

    cmd_t = np.array(cmd_t)
    cmd_x = np.array(cmd_x)
    fb_t = np.array(fb.t)
    fb_x = np.array(fb.x)
    print(f"Logged: {len(cmd_t)} commands, {len(fb_t)} feedback samples")

    latency, info = get_latency(cmd_x, cmd_t, fb_x, fb_t, force_positive=True)
    print(f"\nPiper execution latency: {latency * 1000:.1f} ms")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(info["lags"] * 1000, info["correlation"])
    axes[0].axvline(
        latency * 1000,
        color="r",
        linestyle="--",
        label=f"{latency * 1000:.1f} ms",
    )
    axes[0].set_xlabel("lag (ms)")
    axes[0].set_ylabel("correlation")
    axes[0].set_title("Cross-correlation")
    axes[0].legend()

    axes[1].plot(cmd_t - cmd_t[0], cmd_x, label="commanded")
    axes[1].plot(fb_t - cmd_t[0], fb_x, label="actual")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("X (m)")
    axes[1].set_title("Raw signals")
    axes[1].legend()

    axes[2].plot(cmd_t - cmd_t[0], cmd_x, label="commanded")
    axes[2].plot(
        fb_t - cmd_t[0] - latency,
        fb_x,
        label=f"actual shifted -{latency * 1000:.1f} ms",
    )
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("X (m)")
    axes[2].set_title("Aligned")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"Plot saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
