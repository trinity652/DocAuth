"""
Training script for the Siamese signature verification network.

Usage:
    python -m src.signature.train \\
        --data-dir "Signature Detection and Analysis/data" \\
        --epochs 30 \\
        --batch-size 32 \\
        --backbone efficientnet_b0 \\
        --embed-dim 256 \\
        --lr 3e-4 \\
        --output weights/siamese_best.pt

The script trains with contrastive loss (pytorch-metric-learning) and
saves the best checkpoint by validation loss.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import SignaturePairDataset
from .model import SiameseNet

try:
    from pytorch_metric_learning.losses import ContrastiveLoss
    _HAS_PML = True
except ImportError:
    _HAS_PML = False


def _contrastive_loss(
    e1: torch.Tensor, e2: torch.Tensor, labels: torch.Tensor, margin: float = 0.5
) -> torch.Tensor:
    """Manual contrastive loss fallback when pytorch-metric-learning is absent."""
    dist = 1.0 - torch.sum(e1 * e2, dim=1)  # cosine distance in [0, 2]
    pos = labels * dist.pow(2)
    neg = (1 - labels) * torch.clamp(margin - dist, min=0).pow(2)
    return (pos + neg).mean()


def train(
    data_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 32,
    backbone: str = "efficientnet_b0",
    embed_dim: int = 256,
    lr: float = 3e-4,
    val_split: float = 0.15,
    output: str | Path = "weights/siamese_best.pt",
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dir = Path(data_dir) / "training"
    if not train_dir.exists():
        train_dir = Path(data_dir)  # flat layout fallback

    dataset = SignaturePairDataset(train_dir, train=True)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SiameseNet(backbone=backbone, embed_dim=embed_dim, pretrained=True).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    if _HAS_PML:
        loss_fn = ContrastiveLoss(pos_margin=0, neg_margin=0.5)
    else:
        loss_fn = None

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x1, x2, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            e1, e2 = model(x1, x2)
            if loss_fn is not None:
                embeddings = torch.cat([e1, e2])
                ref_labels = torch.cat([labels, labels])
                loss = loss_fn(embeddings, ref_labels.long())
            else:
                loss = _contrastive_loss(e1, e2, labels)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item()

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x1, x2, labels in val_loader:
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                e1, e2 = model(x1, x2)
                if loss_fn is not None:
                    embeddings = torch.cat([e1, e2])
                    ref_labels = torch.cat([labels, labels])
                    loss = loss_fn(embeddings, ref_labels.long())
                else:
                    loss = _contrastive_loss(e1, e2, labels)
                val_loss += loss.item()
                sim = model.similarity(x1, x2)
                pred = (sim > 0.5).float()
                correct += (pred == labels).sum().item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        acc = correct / len(val_ds)
        print(f"Epoch {epoch:>3}/{epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  val_acc={acc:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), output)
            print(f"  ↳ Saved best weights → {output}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Siamese signature verification network")
    parser.add_argument("--data-dir", default="Signature Detection and Analysis/data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--output", default="weights/siamese_best.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
