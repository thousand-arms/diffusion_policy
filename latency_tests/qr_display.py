"""Render a QR code encoding time.time() on screen as fast as possible.

Run this on the monitor the INDEMIND camera is pointed at, then run
camera_latency.py from another terminal. Press q or ESC to quit.
"""

import time

import cv2
import numpy as np
import qrcode

WINDOW = "qr_latency"
QR_PX = 600  # on-screen QR size in pixels


def make_qr(text: str) -> np.ndarray:
    q = qrcode.QRCode(version=3, box_size=10, border=2)
    q.add_data(text)
    q.make(fit=True)
    img = np.array(q.make_image(fill_color="black", back_color="white").convert("L"))
    return cv2.resize(img, (QR_PX, QR_PX), interpolation=cv2.INTER_NEAREST)


def main():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, QR_PX, QR_PX)
    while True:
        img = make_qr(f"{time.time():.6f}")
        cv2.imshow(WINDOW, img)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
