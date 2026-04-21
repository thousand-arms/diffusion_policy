"""Post-run visualization for the eval_real pipeline.

Plots raw scheduled waypoints (what the policy proposed) against the
smoothed poses the interpolation controller actually sent to the arm.
"""

import os
import time

import numpy as np


def plot_trajectory(sent_log, waypoint_log, t_start, out_root):
    """Plot and save raw waypoints vs smoothed sent poses.

    Args:
        sent_log:     [(t_monotonic, pose6)] from the interp thread at send_hz.
        waypoint_log: [(target_time, pose6, batch_id)] from the main loop.
        t_start:      monotonic time the policy started (for relative x-axis).
        out_root:     directory to write the PNG/NPZ into.
    """
    import matplotlib
    matplotlib.use("TkAgg", force=False)
    import matplotlib.pyplot as plt

    if not sent_log and not waypoint_log:
        print("[viz] no data recorded.")
        return

    sent_t = np.array([s[0] for s in sent_log]) - t_start
    sent_xyz = np.array([s[1][:3] for s in sent_log]) if sent_log else np.zeros((0, 3))

    wp_t = np.array([w[0] for w in waypoint_log]) - t_start
    wp_xyz = np.array([w[1][:3] for w in waypoint_log]) if waypoint_log else np.zeros((0, 3))
    wp_batch = np.array([w[2] for w in waypoint_log]) if waypoint_log else np.array([])

    os.makedirs(out_root, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    npz_path = os.path.join(out_root, f"{stamp}.npz")
    np.savez(
        npz_path,
        sent_t=sent_t, sent_xyz=sent_xyz,
        wp_t=wp_t, wp_xyz=wp_xyz, wp_batch=wp_batch,
    )
    print(f"[viz] saved raw data to {npz_path}")

    fig = plt.figure(figsize=(14, 9))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    if len(sent_xyz):
        ax3d.plot(
            sent_xyz[:, 0] * 1000, sent_xyz[:, 1] * 1000, sent_xyz[:, 2] * 1000,
            color="#2196F3", linewidth=1.2, label="sent (interp, 150Hz)",
        )
    if len(wp_xyz):
        sc = ax3d.scatter(
            wp_xyz[:, 0] * 1000, wp_xyz[:, 1] * 1000, wp_xyz[:, 2] * 1000,
            c=wp_batch, cmap="viridis", s=12, alpha=0.8, label="scheduled",
        )
        plt.colorbar(sc, ax=ax3d, shrink=0.6, label="batch id")
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.set_title("3D camera trajectory (world frame)")
    ax3d.legend(loc="best", fontsize=8)

    for i, lbl in enumerate(["X (mm)", "Y (mm)", "Z (mm)"]):
        ax = fig.add_subplot(2, 2, i + 2)
        if len(sent_t):
            ax.plot(sent_t, sent_xyz[:, i] * 1000,
                    color="#2196F3", linewidth=1.0, label="sent (interp)")
        if len(wp_t):
            ax.scatter(wp_t, wp_xyz[:, i] * 1000, c=wp_batch, cmap="viridis",
                       s=10, alpha=0.7, label="scheduled")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(lbl)
        ax.set_title(lbl)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    png_path = os.path.join(out_root, f"{stamp}.png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"[viz] saved plot to {png_path}")
    try:
        plt.show()
    except Exception as e:
        print(f"[viz] plt.show failed ({e}); use the saved PNG instead.")
