"""
Inference API for Siamese signature verification.

Usage:
    from src.signature.inference import verify

    result = verify("reference_signature.png", "query_signature.png")
    # {"match": True, "confidence": 0.87, "distance": 0.13, "verdict": "Genuine"}
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import _pil_to_tensor
from .model import SiameseNet

# Similarity threshold: cosine_similarity > THRESHOLD → genuine
SIMILARITY_THRESHOLD = 0.50

_model_cache: dict[str, SiameseNet] = {}


def _load_model(
    weights_path: str | Path,
    backbone: str = "efficientnet_b0",
    embed_dim: int = 256,
    device: str = "cpu",
) -> SiameseNet:
    key = str(weights_path)
    if key not in _model_cache:
        model = SiameseNet(backbone=backbone, embed_dim=embed_dim, pretrained=False)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        _model_cache[key] = model
    return _model_cache[key]


def _preprocess(path: str | Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import numpy as np
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        return transform(image=np.array(img))["image"]
    except ImportError:
        return _pil_to_tensor(img)


def verify(
    reference_path: str | Path,
    query_path: str | Path,
    weights: str | Path = "weights/siamese_best.pt",
    threshold: float = SIMILARITY_THRESHOLD,
    backbone: str = "efficientnet_b0",
    embed_dim: int = 256,
    device: str | None = None,
) -> dict:
    """Compare two signature images and return a verification result.

    Args:
        reference_path: Path to the reference (enrolled) signature.
        query_path: Path to the query (candidate) signature.
        weights: Path to trained model weights (.pt file).
        threshold: Cosine similarity threshold (default 0.5).
        backbone: timm backbone used during training.
        embed_dim: Embedding dimension used during training.
        device: ``"cuda"`` or ``"cpu"``; auto-detected if None.

    Returns:
        dict with keys:
          - ``match`` (bool): True if signatures are deemed genuine.
          - ``confidence`` (float): Probability-like score in [0, 1].
          - ``distance`` (float): Cosine distance (0 = identical).
          - ``verdict`` (str): ``"Genuine"`` or ``"Forged"``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{weights}'. "
            "Run `python -m src.signature.train` first or download pretrained weights."
        )

    model = _load_model(weights, backbone=backbone, embed_dim=embed_dim, device=device)

    t1 = _preprocess(reference_path).unsqueeze(0).to(device)
    t2 = _preprocess(query_path).unsqueeze(0).to(device)

    with torch.no_grad():
        e1, e2 = model(t1, t2)
        sim: float = F.cosine_similarity(e1, e2).item()

    distance = 1.0 - sim
    # Map similarity [−1, 1] → confidence [0, 1]
    confidence = (sim + 1.0) / 2.0
    match = sim >= threshold

    return {
        "match": match,
        "confidence": round(confidence, 4),
        "distance": round(distance, 4),
        "verdict": "Genuine" if match else "Forged",
    }
