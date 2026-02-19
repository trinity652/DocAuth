"""Tests for the signature verification module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


DATA_DIR = Path(__file__).parent.parent / "Signature Detection and Analysis" / "data" / "test" / "021"
HAS_DATA = DATA_DIR.exists() and any(DATA_DIR.glob("*.png"))


@pytest.fixture()
def genuine_images():
    return sorted(DATA_DIR.glob("genuine-*.png"))[:2]


@pytest.fixture()
def forged_images():
    return sorted(DATA_DIR.glob("forged-*.png"))[:1]


@pytest.fixture()
def sample_sig_image(tmp_path):
    """Create a minimal dummy signature image for unit tests."""
    img = Image.fromarray(np.random.randint(200, 255, (64, 200, 3), dtype=np.uint8))
    path = tmp_path / "sig.png"
    img.save(path)
    return path


class TestSiameseModel:
    def test_model_instantiation(self):
        from src.signature.model import SiameseNet
        model = SiameseNet(backbone="efficientnet_b0", embed_dim=128, pretrained=False)
        assert model is not None

    def test_forward_pass_shape(self):
        import torch
        from src.signature.model import SiameseNet

        model = SiameseNet(backbone="efficientnet_b0", embed_dim=128, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        e1, e2 = model(x, x)
        assert e1.shape == (2, 128)
        assert e2.shape == (2, 128)

    def test_embeddings_are_normalised(self):
        import torch
        from src.signature.model import SiameseNet

        model = SiameseNet(backbone="efficientnet_b0", embed_dim=128, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        e1, _ = model(x, x)
        norm = e1.norm(dim=1).item()
        assert abs(norm - 1.0) < 1e-5, f"Embedding not normalised: norm={norm}"

    def test_similarity_same_input(self):
        import torch
        from src.signature.model import SiameseNet

        model = SiameseNet(backbone="efficientnet_b0", embed_dim=128, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        sim = model.similarity(x, x).item()
        assert sim > 0.99, f"Same-input similarity should be ~1.0, got {sim}"


class TestPreprocessing:
    def test_pil_to_tensor_shape(self, sample_sig_image):
        from src.signature.dataset import _pil_to_tensor
        from PIL import Image

        img = Image.open(sample_sig_image)
        t = _pil_to_tensor(img)
        assert t.shape == (3, 224, 224)

    def test_pil_to_tensor_dtype(self, sample_sig_image):
        import torch
        from src.signature.dataset import _pil_to_tensor
        from PIL import Image

        img = Image.open(sample_sig_image)
        t = _pil_to_tensor(img)
        assert t.dtype == torch.float32


@pytest.mark.skipif(not HAS_DATA, reason="Signature test data not found")
class TestDataset:
    def test_dataset_loads(self):
        from src.signature.dataset import SignaturePairDataset

        ds = SignaturePairDataset(DATA_DIR.parent, train=False)
        assert len(ds) > 0

    def test_dataset_item_shapes(self):
        from src.signature.dataset import SignaturePairDataset

        ds = SignaturePairDataset(DATA_DIR.parent, train=False)
        x1, x2, label = ds[0]
        assert x1.shape[0] == 3
        assert x2.shape[0] == 3
        assert label.item() in (0.0, 1.0)
