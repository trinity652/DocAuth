"""Tests for the document analysis module (ELA, edge detection, OCR, wavelet)."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def jpeg_image(tmp_path):
    """JPEG image saved at full quality — ELA should show near-zero error."""
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    path = tmp_path / "test.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture()
def png_image(tmp_path):
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    path = tmp_path / "test.png"
    img.save(path)
    return path


# ── ELA ───────────────────────────────────────────────────────────────────────

class TestELA:
    def test_generate_ela_returns_pil_image(self, jpeg_image):
        from src.analysis.ela import generate_ela

        result = generate_ela(jpeg_image)
        assert isinstance(result, Image.Image)

    def test_ela_shape_matches_input(self, jpeg_image):
        from src.analysis.ela import generate_ela

        original = Image.open(jpeg_image).convert("RGB")
        ela = generate_ela(original)
        assert ela.size == original.size

    def test_ela_score_in_range(self, jpeg_image):
        from src.analysis.ela import generate_ela, ela_score

        ela = generate_ela(jpeg_image)
        score = ela_score(ela)
        assert 0.0 <= score <= 1.0

    def test_ela_heatmap_normalised(self, jpeg_image):
        from src.analysis.ela import generate_ela, ela_heatmap

        ela = generate_ela(jpeg_image)
        h = ela_heatmap(ela)
        assert h.min() >= 0.0
        assert h.max() <= 1.0

    def test_ela_accepts_pil_input(self, png_image):
        from src.analysis.ela import generate_ela

        img = Image.open(png_image)
        result = generate_ela(img)
        assert isinstance(result, Image.Image)


# ── Edge Detection ─────────────────────────────────────────────────────────────

class TestEdgeDetection:
    def test_returns_all_detectors(self, png_image):
        from src.analysis.edge_detection import detect_all

        result = detect_all(png_image)
        for key in ("canny", "sobel", "laplacian", "prewitt_x", "prewitt_y"):
            assert key in result

    def test_outputs_are_uint8(self, png_image):
        from src.analysis.edge_detection import detect_all

        result = detect_all(png_image)
        for name, arr in result.items():
            assert arr.dtype == np.uint8, f"{name} is not uint8"

    def test_output_shape_matches_input(self, png_image):
        from src.analysis.edge_detection import detect_all

        gray = np.array(Image.open(png_image).convert("L"))
        result = detect_all(png_image)
        for name, arr in result.items():
            assert arr.shape == gray.shape, f"{name} shape mismatch"


# ── Wavelet ────────────────────────────────────────────────────────────────────

class TestWavelet:
    def test_decompose_returns_keys(self, png_image):
        from src.analysis.wavelet import decompose

        result = decompose(png_image)
        assert "reconstructed" in result
        assert "heatmap" in result
        assert "detail_bands" in result

    def test_heatmap_is_rgb(self, png_image):
        from src.analysis.wavelet import decompose

        result = decompose(png_image)
        assert result["heatmap"].shape[-1] == 3

    def test_different_wavelets(self, png_image):
        from src.analysis.wavelet import decompose

        for w in ("haar", "db4"):
            result = decompose(png_image, wavelet=w, level=2)
            assert result["reconstructed"] is not None
