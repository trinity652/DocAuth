"""
Wavelet-based texture analysis for document forensics.

Modernised from the original heat.py — no hardcoded file paths,
returns numpy arrays instead of writing files, supports any wavelet
and decomposition level supported by PyWavelets.

References:
  - PyWavelets: https://pywavelets.readthedocs.io
  - Haar & Daubechies wavelets for image forensics (standard baseline)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import pywt
except ImportError as e:
    raise ImportError("Install PyWavelets: pip install PyWavelets>=1.7.0") from e


def decompose(
    source: str | Path | Image.Image | np.ndarray,
    wavelet: str = "db1",
    level: int = 3,
) -> dict[str, np.ndarray]:
    """Perform multi-level wavelet decomposition and return detail sub-bands.

    The approximation coefficients are zeroed out so only high-frequency
    detail components (edges, noise, tampering artefacts) remain.

    Args:
        source: Input image (path, PIL Image, or numpy array).
        wavelet: PyWavelets wavelet name (e.g. ``"haar"``, ``"db1"``, ``"sym4"``).
        level:   Number of decomposition levels.

    Returns:
        dict with keys:
          - ``"reconstructed"`` (ndarray, float32): Detail-only reconstruction.
          - ``"heatmap"``       (ndarray, uint8 RGB): Jet-coloured intensity map.
          - ``"detail_bands"``  (list): Raw detail coefficient arrays per level.
    """
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    elif isinstance(source, Image.Image):
        img = np.array(source.convert("L"))
    elif isinstance(source, np.ndarray):
        img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    img_float = img.astype(np.float32) / 255.0

    # Wavelet decomposition
    coeffs = pywt.wavedec2(img_float, wavelet=wavelet, level=level)

    # Zero approximation, keep all detail bands
    coeffs_detail = list(coeffs)
    coeffs_detail[0] = np.zeros_like(coeffs[0])

    # Reconstruct using only detail coefficients
    reconstructed = pywt.waverec2(coeffs_detail, wavelet=wavelet)

    # Crop to original size (waverec2 may pad by 1 pixel)
    reconstructed = reconstructed[: img.shape[0], : img.shape[1]]

    # Normalise to [0, 255]
    recon_norm = cv2.normalize(
        np.abs(reconstructed), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(recon_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    detail_bands = [coeffs[i] for i in range(1, len(coeffs))]

    return {
        "reconstructed": recon_norm,
        "heatmap": heatmap_rgb,
        "detail_bands": detail_bands,
    }
