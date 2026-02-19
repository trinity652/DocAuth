"""
DocAuth — Document Forgery Detection and Analysis
Streamlit multi-tab application.

Run with:
    streamlit run app.py

Tabs:
  1. Signature Verification    — Siamese network pair comparison
  2. Copy-Move Detection       — ORB+RANSAC / photoholmes
  3. Document Analysis         — ELA, edge detection, OCR, wavelet
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocAuth",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("DocAuth — Document Forgery Detection")
st.caption("Powered by PyTorch Siamese networks · ORB+RANSAC · ELA · EasyOCR · PyWavelets")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "✍️  Signature Verification",
    "🔍  Copy-Move Detection",
    "📄  Document Analysis",
])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_upload(uploaded) -> Path:
    """Save a Streamlit UploadedFile to a temp file and return the path."""
    suffix = Path(uploaded.name).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    return Path(tmp.name)


def _verdict_badge(verdict: str) -> None:
    colours = {"Authentic": "🟢", "Genuine": "🟢", "Suspicious": "🟡", "Forged": "🔴"}
    icon = colours.get(verdict, "⚪")
    st.markdown(f"## {icon} {verdict}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Signature Verification
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Offline Signature Verification")
    st.markdown(
        "Upload a **reference** (enrolled) signature and a **query** (candidate) signature. "
        "The Siamese network compares their embeddings and determines if they match."
    )

    col_ref, col_qry = st.columns(2)
    with col_ref:
        ref_file = st.file_uploader(
            "Reference signature", type=["png", "jpg", "jpeg"], key="sig_ref"
        )
        if ref_file:
            st.image(ref_file, caption="Reference", use_container_width=True)

    with col_qry:
        qry_file = st.file_uploader(
            "Query signature", type=["png", "jpg", "jpeg"], key="sig_qry"
        )
        if qry_file:
            st.image(qry_file, caption="Query", use_container_width=True)

    weights_path = st.text_input(
        "Model weights path", value="weights/siamese_best.pt",
        help="Run `python -m src.signature.train` to generate weights."
    )

    if st.button("🔎 Verify Signatures", disabled=not (ref_file and qry_file)):
        ref_path = _save_upload(ref_file)
        qry_path = _save_upload(qry_file)
        weights = Path(weights_path)

        if not weights.exists():
            st.warning(
                f"Weights file `{weights_path}` not found. "
                "Train the model first with:\n"
                "```\npython -m src.signature.train\n```"
            )
        else:
            with st.spinner("Running Siamese network..."):
                from src.signature.inference import verify
                result = verify(ref_path, qry_path, weights=weights)

            _verdict_badge(result["verdict"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{result['confidence']:.1%}")
            m2.metric("Cosine Distance", f"{result['distance']:.4f}")
            m3.metric("Match", "Yes ✓" if result["match"] else "No ✗")

    st.divider()
    with st.expander("ℹ️  About the model"):
        st.markdown("""
**Architecture**: Siamese Network with shared EfficientNet-B0 backbone (timm) +
projection head (Linear → BN → ReLU → Dropout → Linear).

**Training**: Contrastive loss (pytorch-metric-learning), AdamW optimiser,
CosineAnnealingLR scheduler. Default 30 epochs on CEDAR-style paired data.

**References**:
- HTCSigNet (Pattern Recognition, 2025) — Hybrid Transformer-Conv signature network
- Multi-Scale CNN-CrossViT (Complex & Intelligent Systems, 2025) — 98.85% on CEDAR
- TransOSV (Pattern Recognition, 2023) — First ViT-based writer-independent verification
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Copy-Move Detection
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Copy-Move Forgery Detection")
    st.markdown(
        "Upload a document image. The detector identifies regions that have been "
        "copied and pasted within the same image using ORB keypoint matching and "
        "RANSAC geometric verification."
    )

    img_file = st.file_uploader(
        "Document image", type=["png", "jpg", "jpeg", "tiff"], key="cm_img"
    )

    if img_file:
        img_pil = Image.open(img_file).convert("RGB")
        st.image(img_pil, caption="Uploaded image", use_container_width=True)

        if st.button("🔎 Detect Copy-Move"):
            img_path = _save_upload(img_file)
            with st.spinner("Running copy-move detector..."):
                from src.copy_move.detector import detect_copy_move
                from src.copy_move.visualizer import overlay_heatmap, annotate_regions

                result = detect_copy_move(img_path)

            _verdict_badge(result["verdict"])
            m1, m2 = st.columns(2)
            m1.metric("Forgery Score", f"{result['score']:.1%}")
            m2.metric("Detection Method", result["method"])

            st.subheader("Detection Results")
            c1, c2 = st.columns(2)
            with c1:
                mask = result["mask"]
                if mask.any():
                    overlay = overlay_heatmap(np.array(img_pil), mask, alpha=0.4)
                    st.image(overlay, caption="Heatmap overlay", use_container_width=True)
                else:
                    st.info("No significant copy-move regions detected.")

            with c2:
                if result["heatmap"] is not None:
                    st.image(result["heatmap"], caption="Photoholmes heatmap", use_container_width=True)
                elif mask.any():
                    annotated = annotate_regions(np.array(img_pil), mask)
                    st.image(annotated, caption="Annotated regions", use_container_width=True)

    st.divider()
    with st.expander("ℹ️  About the detector"):
        st.markdown("""
**Primary**: [PhotoHolmes](https://github.com/photoholmes/photoholmes) (Splicebuster) when installed.

**Fallback**: ORB feature extraction → BFMatcher → RANSAC homography estimation.
Inlier ratio determines the forgery confidence score.

**References**:
- CMFDFormer (arXiv 2311.13263, 2023): MiT transformer backbone for CMFD
- PhotoHolmes (arXiv 2412.14969, Springer 2025): unified forensics library
- MVSS-Net++ (T-PAMI): multi-view multi-scale supervision
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Document Analysis (ELA + Edge + OCR + Wavelet)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Document Analysis")
    st.markdown("Upload a document to run Error Level Analysis, edge detection, OCR, and wavelet decomposition.")

    doc_file = st.file_uploader(
        "Document image", type=["png", "jpg", "jpeg", "tiff", "bmp"], key="doc_img"
    )

    if doc_file:
        doc_pil = Image.open(doc_file).convert("RGB")
        st.image(doc_pil, caption="Uploaded document", use_container_width=True)

        analysis_options = st.multiselect(
            "Select analyses to run",
            ["Error Level Analysis (ELA)", "Edge Detection", "OCR", "Wavelet Decomposition"],
            default=["Error Level Analysis (ELA)", "Edge Detection"],
        )

        if st.button("▶ Run Analysis"):
            doc_path = _save_upload(doc_file)

            # ── ELA ───────────────────────────────────────────────────────────
            if "Error Level Analysis (ELA)" in analysis_options:
                st.subheader("Error Level Analysis")
                with st.spinner("Generating ELA map..."):
                    from src.analysis.ela import generate_ela, ela_score
                    ela_quality = st.session_state.get("ela_quality", 95)
                    ela_img = generate_ela(doc_pil, quality=ela_quality, scale=15)
                    score = ela_score(ela_img)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(doc_pil, caption="Original", use_container_width=True)
                with c2:
                    st.image(ela_img, caption="ELA Map", use_container_width=True)

                verdict = "Forged" if score > 0.08 else ("Suspicious" if score > 0.03 else "Authentic")
                _verdict_badge(verdict)
                st.metric("ELA Intensity Score", f"{score:.4f}")
                st.caption(
                    "Bright regions in the ELA map indicate areas that may have been "
                    "digitally manipulated. Uniform texture suggests an authentic image."
                )

            # ── Edge Detection ─────────────────────────────────────────────────
            if "Edge Detection" in analysis_options:
                st.subheader("Edge Detection")
                detector = st.selectbox(
                    "Detector", ["canny", "sobel", "laplacian", "prewitt_x", "prewitt_y"],
                    key="edge_det",
                )
                with st.spinner("Running edge detection..."):
                    from src.analysis.edge_detection import detect_all
                    edges = detect_all(doc_pil)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(doc_pil, caption="Original", use_container_width=True)
                with c2:
                    st.image(edges[detector], caption=f"{detector.capitalize()} edges", use_container_width=True)

            # ── OCR ───────────────────────────────────────────────────────────
            if "OCR" in analysis_options:
                st.subheader("Optical Character Recognition")
                handwritten = st.toggle("Handwritten text mode (uses TrOCR)", value=False, key="ocr_hw")
                with st.spinner("Extracting text..."):
                    from src.analysis.ocr import extract_text
                    ocr_result = extract_text(doc_path, handwritten=handwritten)

                st.text_area("Extracted text", ocr_result["full_text"], height=200)
                m1, m2 = st.columns(2)
                m1.metric("Avg. Confidence", f"{ocr_result['avg_confidence']:.1%}")
                m2.metric("Engine", ocr_result["engine"])

                if ocr_result["words"]:
                    with st.expander("Word-level results"):
                        import pandas as pd
                        df = pd.DataFrame([
                            {"Text": w["text"], "Confidence": f"{w['confidence']:.1%}"}
                            for w in ocr_result["words"]
                        ])
                        st.dataframe(df, use_container_width=True)

            # ── Wavelet ────────────────────────────────────────────────────────
            if "Wavelet Decomposition" in analysis_options:
                st.subheader("Wavelet Decomposition")
                col_w, col_l = st.columns(2)
                wavelet = col_w.selectbox("Wavelet", ["haar", "db1", "db4", "sym4"], key="wav_name")
                level = col_l.slider("Decomposition level", 1, 6, 3, key="wav_level")

                with st.spinner("Running wavelet decomposition..."):
                    from src.analysis.wavelet import decompose
                    wav_result = decompose(doc_pil, wavelet=wavelet, level=level)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(doc_pil, caption="Original", use_container_width=True)
                with c2:
                    st.image(wav_result["heatmap"], caption=f"{wavelet} detail heatmap (level {level})", use_container_width=True)

    st.divider()
    with st.expander("ℹ️  ELA quality setting"):
        quality = st.slider(
            "JPEG re-compression quality", min_value=70, max_value=99, value=95,
            help="Lower quality amplifies differences in manipulated regions.",
            key="ela_quality",
        )
