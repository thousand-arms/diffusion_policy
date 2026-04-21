"""Image preprocessing utilities for the Piper pipeline."""

import cv2
import numpy as np


def preprocess_image(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Grayscale (H,W) or (H,W,1) uint8 -> (3, size, size) float32 [0,1].

    Matches trace_dataset._sample_to_data: resize, normalize, triplicate
    so a 3-channel ResNet backbone can consume it.
    """
    if img.ndim == 3:
        img = img[..., 0]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.repeat(img[None], 3, axis=0)  # (3, H, W)
