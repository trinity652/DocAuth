"""Tests for the copy-move forgery detection module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


TEST_IMAGES = Path(__file__).parent.parent / "Copy Move Forgery" / "CopyMoveDetection" / "test_images"
HAS_TEST_IMAGES = TEST_IMAGES.exists() and any(TEST_IMAGES.glob("*.png"))


@pytest.fixture()
def sample_image(tmp_path):
    """Create a small synthetic image with a duplicated region."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # Duplicate a 64×64 region
    arr[100:164, 150:214] = arr[10:74, 10:74]
    img = Image.fromarray(arr)
    path = tmp_path / "copy_move_test.png"
    img.save(path)
    return path


@pytest.fixture()
def uniform_image(tmp_path):
    """A uniform grey image — should return low forgery score."""
    arr = np.full((256, 256, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr)
    path = tmp_path / "uniform.png"
    img.save(path)
    return path


class TestDetector:
    def test_returns_dict_keys(self, sample_image):
        from src.copy_move.detector import detect_copy_move

        result = detect_copy_move(sample_image)
        assert "score" in result
        assert "verdict" in result
        assert "mask" in result
        assert "method" in result

    def test_score_in_range(self, sample_image):
        from src.copy_move.detector import detect_copy_move

        result = detect_copy_move(sample_image)
        assert 0.0 <= result["score"] <= 1.0

    def test_verdict_values(self, sample_image):
        from src.copy_move.detector import detect_copy_move

        result = detect_copy_move(sample_image)
        assert result["verdict"] in ("Authentic", "Suspicious", "Forged")

    def test_mask_shape_matches_image(self, sample_image):
        import cv2
        from src.copy_move.detector import detect_copy_move

        result = detect_copy_move(sample_image)
        img = cv2.imread(str(sample_image), cv2.IMREAD_GRAYSCALE)
        assert result["mask"].shape == img.shape

    def test_invalid_path_raises(self, tmp_path):
        from src.copy_move.detector import detect_copy_move

        with pytest.raises(ValueError, match="Cannot read image"):
            detect_copy_move(tmp_path / "nonexistent.png")


class TestVisualizer:
    def test_mask_to_heatmap_shape(self, sample_image):
        import cv2
        from src.copy_move.visualizer import mask_to_heatmap

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:100, 50:100] = 255
        heatmap = mask_to_heatmap(mask)
        assert heatmap.shape == (256, 256, 3)

    def test_overlay_returns_rgb(self, sample_image):
        from src.copy_move.visualizer import overlay_heatmap

        img = np.array(Image.open(sample_image))
        mask = np.zeros((256, 256), dtype=np.uint8)
        result = overlay_heatmap(img, mask)
        assert result.shape[-1] == 3  # RGB


@pytest.mark.skipif(not HAS_TEST_IMAGES, reason="Copy-move test images not found")
class TestOnRealImages:
    def test_cheque(self):
        from src.copy_move.detector import detect_copy_move

        cheque = TEST_IMAGES / "cheque22.png"
        if cheque.exists():
            result = detect_copy_move(cheque)
            assert 0.0 <= result["score"] <= 1.0
