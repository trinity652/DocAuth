"""
Modern edge detection utilities for document forensics.

All functions accept a PIL Image or file path and return a dict of
named result arrays — no hardcoded filenames, no GUI calls.

Detectors implemented:
  - Canny  (adaptive thresholding with Otsu)
  - Sobel  (gradient magnitude)
  - Laplacian of Gaussian (LoG)
  - Prewitt (custom kernel)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _load_gray(source: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
    if isinstance(source, Image.Image):
        return np.array(source.convert("L"))
    return cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)


def detect_all(
    source: str | Path | Image.Image | np.ndarray,
    blur_ksize: int = 3,
) -> dict[str, np.ndarray]:
    """Run all edge detectors and return results as a dict of uint8 arrays.

    Args:
        source: Input image (path, PIL Image, or numpy array).
        blur_ksize: Gaussian blur kernel size applied before detection.

    Returns:
        dict with keys: ``"canny"``, ``"sobel"``, ``"laplacian"``, ``"prewitt_x"``, ``"prewitt_y"``.
    """
    gray = _load_gray(source)
    if gray is None:
        raise ValueError(f"Cannot load image: {source}")

    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # ── Canny ─────────────────────────────────────────────────────────────────
    # Use Otsu to estimate thresholds automatically
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny = cv2.Canny(blurred, otsu_thresh * 0.5, otsu_thresh)

    # ── Sobel ─────────────────────────────────────────────────────────────────
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── Laplacian of Gaussian ─────────────────────────────────────────────────
    lap_raw = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.normalize(np.abs(lap_raw), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── Prewitt ───────────────────────────────────────────────────────────────
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    prewitt_x_raw = cv2.filter2D(blurred.astype(np.float32), -1, kernel_x)
    prewitt_y_raw = cv2.filter2D(blurred.astype(np.float32), -1, kernel_y)
    prewitt_x = cv2.normalize(np.abs(prewitt_x_raw), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    prewitt_y = cv2.normalize(np.abs(prewitt_y_raw), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {
        "canny": canny,
        "sobel": sobel,
        "laplacian": laplacian,
        "prewitt_x": prewitt_x,
        "prewitt_y": prewitt_y,
    }
