"""Measure Piper end-effector execution latency.

Sends a sine wave on the X axis of the camera TCP pose. Logs both the
commanded trajectory and the actual feedback. Cross-correlates to recover
the execution latency.

Usage:
    1. Manually move the arm to a safe position
    2. Run: python piper/latency_tests/piper_latency.py
    The script uses the current arm position as the base pose.
"""

import sys
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from piper.env.piper_driver import PiperDriver

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latency_util import get_latency

# Sine stimulus on camera X axis
AMPLITUDE_M = 0.03       # 30mm
FREQUENCY_HZ = 0.5       # slow enough for the arm to track
COMMAND_HZ = 30           # command send rate
FEEDBACK_HZ = 200         # feedback poll rate
DURATION_S = 10.0
SETTLE_S = 2.0            # settle time before sweeping
SPEED_PCT = 100

PLOT_PATH = Path(__file__).resolve().parent / "piper_latency_plot.png"


class FeedbackWorker(threading.Thread):
    """Poll camera-frame X position at high rate."""

    def __init__(self, driver: PiperDriver, hz: int):
        super().__init__(daemon=True)
        self.driver = driver
        self.dt = 1.0 / hz
        self.stop_event = threading.Event()
        self.t = []
        self.x = []  # camera X in meters

    def run(self):
        next_t = time.time()
        while not self.stop_event.is_set():
            pos = self.driver.get_camera_pose_mat()[:3, 3]
            now = time.time()
            self.x.append(pos[0])  # camera X
            self.t.append(now)
            next_t += self.dt
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()


def main():
    print(f"[init] Connecting Piper (speed_pct={SPEED_PCT})...")
    driver = PiperDriver(can_port="can0", speed_pct=SPEED_PCT)
    time.sleep(0.5)

    # Read current camera pose as base
    base_T = driver.get_camera_pose_mat().copy()
    base_pos = base_T[:3, 3]
    print(f"[init] Current camera pos (m): x={base_pos[0]:.4f} y={base_pos[1]:.4f} z={base_pos[2]:.4f}")
    print(f"[init] Sine sweep on camera X: A={AMPLITUDE_M*1000:.0f}mm  f={FREQUENCY_HZ}Hz")
    print(f"[init] Settling for {SETTLE_S}s...")
    time.sleep(SETTLE_S)

    # Start feedback polling
    fb = FeedbackWorker(driver, FEEDBACK_HZ)
    fb.start()

    # Command loop: sine on camera X
    cmd_t, cmd_x = [], []
    cmd_dt = 1.0 / COMMAND_HZ
    t0 = time.time()
    next_cmd = t0

    print(f"[sweep] Running for {DURATION_S}s at cmd@{COMMAND_HZ}Hz fb@{FEEDBACK_HZ}Hz...")
    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed > DURATION_S:
            break

        x_offset = AMPLITUDE_M * np.sin(2 * np.pi * FREQUENCY_HZ * elapsed)
        target_T = base_T.copy()
        target_T[0, 3] = base_pos[0] + x_offset  # only modify camera X position

        driver.set_camera_pose_mat(target_T)
        cmd_t.append(now)
        cmd_x.append(base_pos[0] + x_offset)

        next_cmd += cmd_dt
        sleep = next_cmd - time.time()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_cmd = time.time()

    # Return to base and stop
    driver.set_camera_pose_mat(base_T)
    fb.stop_event.set()
    fb.join(timeout=1.0)

    cmd_t = np.array(cmd_t)
    cmd_x = np.array(cmd_x)
    fb_t = np.array(fb.t)
    fb_x = np.array(fb.x)
    print(f"[data] {len(cmd_t)} commands, {len(fb_t)} feedback samples")

    # Trim first 2 seconds (warm-up: arm may not yet be tracking smoothly)
    trim_s = 2.0
    cmd_mask = cmd_t >= (cmd_t[0] + trim_s)
    fb_mask = fb_t >= (fb_t[0] + trim_s)
    cmd_t_trim = cmd_t[cmd_mask]
    cmd_x_trim = cmd_x[cmd_mask]
    fb_t_trim = fb_t[fb_mask]
    fb_x_trim = fb_x[fb_mask]
    print(f"[data] After {trim_s}s trim: {len(cmd_t_trim)} commands, {len(fb_t_trim)} feedback")

    # Cross-correlation
    latency, info = get_latency(cmd_x_trim, cmd_t_trim, fb_x_trim, fb_t_trim, force_positive=True)
    print(f"\n>>> Round-trip latency (cmd→fb): {latency * 1000:.1f} ms <<<")
    print(f"    (This includes CAN feedback delay. One-way execution"
          f" latency is likely ~{latency * 1000 - 10:.0f}-{latency * 1000 - 5:.0f} ms)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(info["lags"] * 1000, info["correlation"])
    axes[0].axvline(latency * 1000, color="r", linestyle="--",
                    label=f"{latency * 1000:.1f} ms")
    axes[0].set_xlabel("lag (ms)")
    axes[0].set_ylabel("correlation")
    axes[0].set_title("Cross-correlation")
    axes[0].legend()

    axes[1].plot(cmd_t_trim - cmd_t_trim[0], cmd_x_trim * 1000, label="commanded")
    axes[1].plot(fb_t_trim - cmd_t_trim[0], fb_x_trim * 1000, label="actual")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("camera X (mm)")
    axes[1].set_title("Raw signals (trimmed)")
    axes[1].legend()

    axes[2].plot(cmd_t_trim - cmd_t_trim[0], cmd_x_trim * 1000, label="commanded")
    axes[2].plot(fb_t_trim - cmd_t_trim[0] - latency, fb_x_trim * 1000,
                label=f"actual shifted -{latency * 1000:.1f}ms")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("camera X (mm)")
    axes[2].set_title("Aligned")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"[plot] Saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
