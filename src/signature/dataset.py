"""
Dataset loader for offline signature verification.

Supports the directory structure used by the CEDAR dataset and the
existing DocAuth test data:

    data/
      training/
        <writer_id>/
          genuine-01.png
          ...
          forged-01.png
          ...
      test/
        <writer_id>/
          genuine-01.png
          ...
          forged-01.png
          ...

Images whose filename starts with "genuine" are labelled 1 (authentic).
All others are labelled 0 (forged).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False

# ── Default transforms ────────────────────────────────────────────────────────

def _make_transforms(train: bool = True) -> "A.Compose | None":
    if not _HAS_ALBUMENTATIONS:
        return None
    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=10, p=0.4),
            A.GaussNoise(var_limit=(5, 25), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Fallback when albumentations is not installed."""
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


# ── Dataset ───────────────────────────────────────────────────────────────────

class SignaturePairDataset(Dataset):
    """Pairs of (reference, query) images with a match label (1=genuine, 0=forged).

    Each sample is a contrastive pair drawn from the same writer:
    - Positive pair (label=1): two genuine signatures
    - Negative pair (label=0): one genuine + one forged signature
    """

    def __init__(self, root: str | Path, train: bool = True) -> None:
        self.root = Path(root)
        self.train = train
        self.transforms = _make_transforms(train)
        self.pairs: list[tuple[Path, Path, int]] = []
        self._build_pairs()

    def _build_pairs(self) -> None:
        for writer_dir in sorted(self.root.iterdir()):
            if not writer_dir.is_dir():
                continue
            genuine = sorted(p for p in writer_dir.iterdir() if "genuine" in p.stem and not "another" in p.stem)
            forged = sorted(p for p in writer_dir.iterdir() if p.stem.startswith("forged"))
            # Positive pairs
            for i in range(len(genuine) - 1):
                self.pairs.append((genuine[i], genuine[i + 1], 1))
            # Negative pairs (genuine vs forged)
            for g in genuine:
                for f in forged:
                    self.pairs.append((g, f, 0))
        random.shuffle(self.pairs)

    def _load(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            arr = np.array(img)
            return self.transforms(image=arr)["image"]
        return _pil_to_tensor(img)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1, p2, label = self.pairs[idx]
        return self._load(p1), self._load(p2), torch.tensor(label, dtype=torch.float32)
