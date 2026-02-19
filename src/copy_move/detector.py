"""
Copy-Move Forgery Detector.

Two-tier approach:
  1. Primary  — photoholmes library (NoiseSniffer / Splicebuster) when available.
  2. Fallback — Classical ORB feature matching + RANSAC homography estimation.
               Interpretable, fast, and dependency-free beyond OpenCV.

References:
  - CMFDFormer (arXiv 2311.13263): MiT transformer backbone for CMFD
  - PhotoHolmes (arXiv 2412.14969 / Springer 2025): pip install photoholmes
  - MVSS-Net++ (T-PAMI): multi-view multi-scale supervision
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ── Photoholmes (optional primary detector) ───────────────────────────────────

def _try_photoholmes(image: np.ndarray) -> dict | None:
    """Run photoholmes Splicebuster if the library is installed."""
    try:
        from photoholmes.methods.splicebuster import Splicebuster
        from photoholmes.preprocessing.image import ImagePreprocessing

        method = Splicebuster()
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = method.predict(img_pil)
        heatmap = np.array(result.heatmap) if result.heatmap is not None else None
        score = float(result.score) if result.score is not None else 0.0
        return {"score": score, "heatmap": heatmap, "method": "photoholmes/splicebuster"}
    except Exception:
        return None


# ── ORB + RANSAC fallback ─────────────────────────────────────────────────────

def _orb_ransac(gray: np.ndarray, nfeatures: int = 5000, min_match_count: int = 10) -> dict:
    """Detect copy-move forgery using ORB keypoints and RANSAC homography.

    Matches are filtered to exclude trivially close keypoints (same region).
    Returns detected duplicate region pairs and a binary forgery mask.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(gray, None)

    if des is None or len(kp) < 2:
        return {"score": 0.0, "mask": np.zeros_like(gray), "matches": [], "method": "orb_ransac"}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # knnMatch with k=2 → ratio test; k=3 so we can skip self-match (dist=0)
    raw_matches = bf.knnMatch(des, des, k=3)

    MIN_SPATIAL_DIST = 20  # pixels — ignore same-region matches

    good_matches = []
    for m_list in raw_matches:
        for m in m_list[1:]:  # skip self-match (index 0, distance ≈ 0)
            pt1 = np.array(kp[m.queryIdx].pt)
            pt2 = np.array(kp[m.trainIdx].pt)
            if np.linalg.norm(pt1 - pt2) > MIN_SPATIAL_DIST:
                good_matches.append(m)
                break

    mask_out = np.zeros(gray.shape, dtype=np.uint8)
    score = 0.0

    if len(good_matches) >= min_match_count:
        src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        _, ransac_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if ransac_mask is not None:
            inliers = ransac_mask.ravel().tolist()
            inlier_count = sum(inliers)
            score = min(1.0, inlier_count / max(1, len(good_matches)))

            # Draw detected duplicate regions on mask
            for i, (m, inlier) in enumerate(zip(good_matches, inliers)):
                if inlier:
                    pt1 = tuple(map(int, kp[m.queryIdx].pt))
                    pt2 = tuple(map(int, kp[m.trainIdx].pt))
                    cv2.circle(mask_out, pt1, 5, 255, -1)
                    cv2.circle(mask_out, pt2, 5, 255, -1)
                    cv2.line(mask_out, pt1, pt2, 128, 1)

    return {
        "score": round(score, 4),
        "mask": mask_out,
        "matches": good_matches,
        "method": "orb_ransac",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def detect_copy_move(image_path: str | Path) -> dict:
    """Detect copy-move forgery in a document image.

    Args:
        image_path: Path to the input image.

    Returns:
        dict with keys:
          - ``score``   (float):  Forgery confidence in [0, 1].
          - ``verdict`` (str):    ``"Authentic"``, ``"Suspicious"``, or ``"Forged"``.
          - ``heatmap`` (ndarray | None): Per-pixel suspicion map (if available).
          - ``mask``    (ndarray): Binary detection mask from ORB+RANSAC.
          - ``method``  (str):    Which detector was used.
    """
    image_path = Path(image_path)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Try photoholmes first
    ph_result = _try_photoholmes(bgr)
    orb_result = _orb_ransac(gray)

    if ph_result is not None:
        score = max(ph_result["score"], orb_result["score"])
        heatmap = ph_result.get("heatmap")
        method = ph_result["method"] + " + orb_ransac"
    else:
        score = orb_result["score"]
        heatmap = None
        method = orb_result["method"]

    if score < 0.10:
        verdict = "Authentic"
    elif score < 0.55:
        verdict = "Suspicious"
    else:
        verdict = "Forged"

    return {
        "score": round(score, 4),
        "verdict": verdict,
        "heatmap": heatmap,
        "mask": orb_result["mask"],
        "method": method,
    }
