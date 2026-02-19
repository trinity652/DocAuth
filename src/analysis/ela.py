"""
Error Level Analysis (ELA) for document forgery detection.

ELA works by re-saving a JPEG image at a fixed quality level and computing
the absolute difference between the original and the re-compressed version.
Regions that were previously manipulated (and thus already compressed)
show lower error levels than untouched regions, making them visually distinct.

References:
  - Farid, H. (2009). "Image forgery detection." IEEE Signal Processing Magazine.
  - PMC 11323046 (2024): ELA + ResNet50 + CBAM, 96.21% accuracy on CASIA v2.
  - IJARCCE 2025: ELA + CNN achieving 96.21% on CASIA v2.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def generate_ela(
    source: str | Path | Image.Image,
    quality: int = 95,
    scale: int = 15,
) -> Image.Image:
    """Generate an Error Level Analysis (ELA) map.

    Args:
        source: Path to image or a PIL Image object.
        quality: JPEG re-compression quality (default 95).
                 Lower values amplify compression differences.
        scale: Multiplier applied to the absolute pixel differences so
               subtle changes become visible (default 15).

    Returns:
        PIL Image (RGB) showing the ELA map. Bright regions indicate
        potential manipulation.
    """
    if isinstance(source, (str, Path)):
        original = Image.open(source).convert("RGB")
    else:
        original = source.convert("RGB")

    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    orig_arr = np.array(original, dtype=np.float32)
    comp_arr = np.array(compressed, dtype=np.float32)

    ela_arr = np.abs(orig_arr - comp_arr) * scale
    ela_arr = ela_arr.clip(0, 255).astype(np.uint8)

    return Image.fromarray(ela_arr)


def ela_score(ela_image: Image.Image) -> float:
    """Compute a scalar suspicion score from an ELA image.

    Higher values indicate more potential manipulation.

    Returns:
        Float in [0, 1] representing overall ELA intensity.
    """
    arr = np.array(ela_image, dtype=np.float32)
    max_possible = 255.0 * arr.shape[0] * arr.shape[1] * arr.shape[2]
    return float(arr.sum() / max_possible)


def ela_heatmap(ela_image: Image.Image) -> np.ndarray:
    """Convert an ELA image to a single-channel intensity map (grayscale).

    Useful for thresholding or feeding into downstream CNN classifiers.
    """
    gray = np.array(ela_image.convert("L"), dtype=np.float32)
    return gray / 255.0
