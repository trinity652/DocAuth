"""
Visualization utilities for copy-move forgery detection results.

Produces:
  - Heatmap overlays (COLORMAP_JET)
  - Annotated images with detected regions highlighted
  - Side-by-side comparison arrays (suitable for Streamlit st.image)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def mask_to_heatmap(mask: np.ndarray) -> np.ndarray:
    """Convert a single-channel detection mask to a colour heatmap (RGB)."""
    if mask.dtype != np.uint8:
        mask = (mask * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    image: np.ndarray | Image.Image,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a forgery mask as a semi-transparent heatmap on the original image.

    Args:
        image: Original BGR ndarray or PIL Image (RGB).
        mask: Single-channel detection mask.
        alpha: Transparency of the heatmap layer (0 = invisible, 1 = opaque).

    Returns:
        RGB ndarray with the heatmap blended in.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    heatmap_bgr = cv2.applyColorMap(
        cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0])),
        cv2.COLORMAP_JET,
    )
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_bgr, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def annotate_regions(
    image: np.ndarray | Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    min_area: int = 100,
) -> np.ndarray:
    """Draw bounding rectangles around detected forgery regions.

    Args:
        image: Original image (BGR ndarray or PIL Image).
        mask: Binary mask (uint8, 0 or 255).
        color: BGR colour for bounding boxes.
        min_area: Ignore contours smaller than this (px²).

    Returns:
        RGB ndarray with annotated regions.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image = image.copy()

    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def side_by_side(
    original: np.ndarray | Image.Image,
    result: np.ndarray,
    label_left: str = "Original",
    label_right: str = "Forgery Detection",
    font_scale: float = 0.7,
) -> np.ndarray:
    """Create a horizontal comparison image with labels."""
    if isinstance(original, Image.Image):
        original = np.array(original.convert("RGB"))
    if isinstance(result, Image.Image):
        result = np.array(result.convert("RGB"))

    h = max(original.shape[0], result.shape[0])
    orig_resized = cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h))
    res_resized = cv2.resize(result, (int(result.shape[1] * h / result.shape[0]), h))

    # Add label bars
    def _add_label(img: np.ndarray, text: str) -> np.ndarray:
        bar = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        return np.vstack([bar, img])

    orig_labelled = _add_label(orig_resized, label_left)
    res_labelled = _add_label(res_resized, label_right)

    return np.hstack([orig_labelled, res_labelled])
