"""Combined QR display + INDEMIND camera latency measurement.

Follows UMI's calibrate_uvc_camera_latency.py methodology:

    t_sample = time.time()  # before QR generation
    # build QR encoding str(t_sample), then imshow
    t_show   = time.time()  # after imshow returns
    qr_overhead = t_show - t_sample           # script overhead per iter
    raw_lat     = t_recv - t_sample_decoded   # measured per detected QR
    avg_latency = mean(raw_lat) - mean(qr_overhead)

The avg_qr_overhead correction removes the QR-gen + imshow time so
t_show (imshow-return) is treated as the effective display time.

Architecture: display loop on main thread (refreshes as fast as possible).
Capture loop on worker thread drains camera frames at the camera's
natural rate so stale frames don't pile up in the SDK buffer.

Note: avg_latency still includes monitor pixel response + refresh wait
(your 240 Hz panel adds ~5-10 ms). Subtract that for camera-only.
"""

import threading
import time
from collections import deque

import cv2
import numpy as np
import pyindemind
import qrcode

QR_PX = 500
PANEL_H = 500
QR_OVERHEAD_MAXLEN = 500  # rolling window for mean qr_overhead


def make_qr(text: str, size: int) -> np.ndarray:
    q = qrcode.QRCode(version=3, box_size=10, border=2)
    q.add_data(text)
    q.make(fit=True)
    img = np.array(q.make_image(fill_color="black", back_color="white").convert("L"))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class CaptureWorker(threading.Thread):
    def __init__(self, cam: pyindemind.Camera, qr_overhead_deque: deque):
        super().__init__(daemon=True)
        self.cam = cam
        self.qr_overhead_deque = qr_overhead_deque  # shared, written by main
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self._latest_vis: "np.ndarray | None" = None
        self.latencies: list[float] = []  # raw t_recv - t_sample_decoded
        self.detector = cv2.QRCodeDetector()

    def run(self):
        while not self.stop_event.is_set():
            frame = self.cam.get_frame(timeout_s=0.1)
            if frame is None:
                continue
            _, left, _ = frame
            receive_time = time.time()

            text, pts, _ = self.detector.detectAndDecode(left)
            latency_ms = None
            if text:
                try:
                    display_time = float(text)
                    latency = receive_time - display_time
                    self.latencies.append(latency)
                    latency_ms = latency * 1000
                    if len(self.latencies) % 30 == 0:
                        recent = np.array(self.latencies[-100:]) * 1000
                        qr_oh = (
                            np.mean(self.qr_overhead_deque) * 1000
                            if self.qr_overhead_deque
                            else 0.0
                        )
                        corrected = recent - qr_oh
                        print(
                            f"n={len(self.latencies):4d}  "
                            f"raw_mean={recent.mean():5.1f}ms  "
                            f"qr_oh={qr_oh:4.1f}ms  "
                            f"corrected={corrected.mean():5.1f}ms  "
                            f"std={corrected.std():4.1f}ms"
                        )
                except ValueError:
                    pass

            vis = left if left.ndim == 3 else cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            if pts is not None:
                pts_int = np.asarray(pts).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(vis, [pts_int], (0, 0, 0))
            h, w = vis.shape[:2]
            scale = PANEL_H / h
            vis = cv2.resize(vis, (int(w * scale), PANEL_H))
            if latency_ms is not None:
                cv2.putText(
                    vis,
                    f"{latency_ms:.1f} ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            with self.lock:
                self._latest_vis = vis

    def get_latest_vis(self) -> "np.ndarray | None":
        with self.lock:
            return self._latest_vis

    def stop(self):
        self.stop_event.set()


def main():
    cam = pyindemind.Camera()
    if not cam.start(resolution="640x400", img_hz=50, imu_hz=1000):
        raise SystemExit(
            "Failed to start INDEMIND camera. Check USB permissions and that "
            "the device is plugged in."
        )

    qr_overhead_deque: deque = deque(maxlen=QR_OVERHEAD_MAXLEN)
    worker = CaptureWorker(cam, qr_overhead_deque)
    worker.start()

    placeholder = np.zeros((PANEL_H, 640, 3), dtype=np.uint8)

    try:
        while True:
            t_sample = time.time()
            qr_panel = make_qr(f"{t_sample:.6f}", QR_PX)
            vis = worker.get_latest_vis()
            if vis is None:
                vis = placeholder
            combined = np.hstack([qr_panel, vis])
            cv2.imshow("latency", combined)
            t_show = time.time()
            qr_overhead_deque.append(t_show - t_sample)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        worker.stop()
        worker.join(timeout=1.0)
        cam.stop()
        cv2.destroyAllWindows()
        if worker.latencies:
            raw = np.array(worker.latencies) * 1000
            qr_oh = np.mean(qr_overhead_deque) * 1000 if qr_overhead_deque else 0.0
            corrected = raw - qr_oh
            print(
                f"\nFinal: n={len(raw)}  qr_overhead={qr_oh:.1f}ms"
                f"\n  raw       mean={raw.mean():.1f}ms  std={raw.std():.1f}ms"
                f"  p50={np.percentile(raw, 50):.1f}ms"
                f"  p95={np.percentile(raw, 95):.1f}ms"
                f"\n  corrected mean={corrected.mean():.1f}ms  std={corrected.std():.1f}ms"
                f"  p50={np.percentile(corrected, 50):.1f}ms"
                f"  p95={np.percentile(corrected, 95):.1f}ms"
                f"\n  (subtract another ~5-10ms for monitor lag to get camera-only)"
            )


if __name__ == "__main__":
    main()
