"""
OCR module for document forensics.

Two engines:
  1. EasyOCR  — best for multi-language printed documents (fast, pip-installable).
  2. TrOCR    — transformer-based OCR (HuggingFace); superior for handwritten text
                and degraded documents. Model: microsoft/trocr-base-printed or
                microsoft/trocr-base-handwritten.

The module auto-selects based on the ``handwritten`` flag and availability
of each library.

References:
  - EasyOCR: github.com/JaidedAI/EasyOCR
  - TrOCR: huggingface.co/docs/transformers/model_doc/trocr
  - PaddleOCR (PP-OCRv5, 2025): 94.5% on OmniDocBench v1.5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


# ── EasyOCR ───────────────────────────────────────────────────────────────────

_easyocr_reader_cache: dict[tuple, object] = {}


def _easyocr_extract(image_path: str | Path, languages: list[str], gpu: bool) -> dict:
    try:
        import easyocr
    except ImportError as e:
        raise ImportError("Install EasyOCR: pip install easyocr>=1.7.1") from e

    key = (tuple(languages), gpu)
    if key not in _easyocr_reader_cache:
        _easyocr_reader_cache[key] = easyocr.Reader(languages, gpu=gpu)
    reader = _easyocr_reader_cache[key]

    results = reader.readtext(str(image_path))
    words = []
    for bbox, text, conf in results:
        words.append({
            "text": text,
            "confidence": round(float(conf), 4),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
        })
    full_text = " ".join(w["text"] for w in words)
    avg_conf = float(np.mean([w["confidence"] for w in words])) if words else 0.0
    return {"full_text": full_text, "words": words, "avg_confidence": round(avg_conf, 4), "engine": "easyocr"}


# ── TrOCR ─────────────────────────────────────────────────────────────────────

_trocr_model_cache: dict[str, tuple] = {}


def _trocr_extract(image_path: str | Path, handwritten: bool) -> dict:
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as e:
        raise ImportError("Install transformers: pip install transformers>=4.45.0") from e

    import torch

    model_id = (
        "microsoft/trocr-base-handwritten" if handwritten else "microsoft/trocr-base-printed"
    )

    if model_id not in _trocr_model_cache:
        processor = TrOCRProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        model.eval()
        _trocr_model_cache[model_id] = (processor, model)

    processor, model = _trocr_model_cache[model_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {
        "full_text": text,
        "words": [{"text": text, "confidence": 1.0, "bbox": []}],
        "avg_confidence": 1.0,
        "engine": "trocr",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def extract_text(
    image_path: str | Path,
    handwritten: bool = False,
    languages: list[str] | None = None,
    gpu: bool = False,
    engine: str = "auto",
) -> dict:
    """Extract text from a document image.

    Args:
        image_path: Path to the input image.
        handwritten: If True, use TrOCR handwriting model instead of EasyOCR.
        languages: Language codes for EasyOCR (default: ``["en"]``).
        gpu: Enable GPU for EasyOCR (requires CUDA).
        engine: ``"auto"`` | ``"easyocr"`` | ``"trocr"``.
                ``"auto"`` selects TrOCR for handwritten, EasyOCR otherwise.

    Returns:
        dict with keys:
          - ``full_text``      (str): Concatenated recognised text.
          - ``words``          (list): Per-word dicts with text, confidence, bbox.
          - ``avg_confidence`` (float): Average recognition confidence.
          - ``engine``         (str): Which OCR engine was used.
    """
    if languages is None:
        languages = ["en"]

    if engine == "auto":
        engine = "trocr" if handwritten else "easyocr"

    if engine == "trocr":
        return _trocr_extract(image_path, handwritten=handwritten)
    else:
        return _easyocr_extract(image_path, languages=languages, gpu=gpu)
