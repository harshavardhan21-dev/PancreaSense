import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st
from src.predict import run_pipeline
from datetime import datetime

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────

st.set_page_config(
    page_title="PancreaSense — Pancreas AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: #080D1A;
        color: #D6E4F0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0D1526;
        border-right: 1px solid #1E2D47;
    }
    [data-testid="stSidebar"] * { color: #B0C4D8 !important; }

    /* ── Header strip ── */
    .header-strip {
        background: linear-gradient(135deg, #0A1628 0%, #0F2040 100%);
        border: 1px solid #1E3A5C;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .header-logo  { font-size: 40px; line-height: 1; }
    .header-title { font-size: 28px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px; margin: 0; }
    .header-subtitle { font-size: 13px; color: #6A8CAE; margin: 4px 0 0 0; font-weight: 400; letter-spacing: 0.3px; }
    .header-badge {
        margin-left: auto;
        background: #00C6BE22;
        border: 1px solid #00C6BE55;
        color: #00C6BE;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 1px;
        text-transform: uppercase;
        white-space: nowrap;
    }

    /* ── Section heading ── */
    .section-heading {
        font-size: 11px;
        font-weight: 600;
        color: #4A7AAA;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 28px 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #1A2D45;
    }

    /* ── Metric cards ── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 8px;
    }
    .metric-card {
        background: #0D1A2E;
        border: 1px solid #1A2D45;
        border-radius: 10px;
        padding: 18px 20px;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #2A4A6E; }
    .metric-label {
        font-size: 10px;
        font-weight: 600;
        color: #4A7AAA;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }
    .metric-value.detected     { color: #00C6BE; }
    .metric-value.not-detected { color: #F0883E; }
    .metric-sub { font-size: 11px; color: #4A7AAA; margin-top: 4px; }

    /* ── Status pills ── */
    .status-detected, .status-not-detected {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-detected     { background: #00C6BE18; border: 1px solid #00C6BE44; color: #00C6BE; }
    .status-not-detected { background: #F0883E18; border: 1px solid #F0883E44; color: #F0883E; }

    /* ── Image panels ── */
    .img-panel {
        background: #0A1220;
        border: 1px solid #1A2D45;
        border-radius: 10px;
        padding: 14px;
    }
    .img-panel-label {
        font-size: 10px;
        font-weight: 600;
        color: #4A7AAA;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #162030;
    }

    /* ── Findings box ── */
    .findings-box {
        background: #0A1A2A;
        border-left: 3px solid #00C6BE;
        border-radius: 0 8px 8px 0;
        padding: 18px 22px;
        margin: 4px 0;
    }
    .findings-box.warning { border-left-color: #F0883E; }
    .findings-row {
        display: flex;
        align-items: baseline;
        gap: 10px;
        margin: 8px 0;
        font-size: 14px;
        color: #C0D4E8;
    }
    .findings-key {
        color: #6A8CAE;
        font-size: 12px;
        font-weight: 500;
        min-width: 140px;
    }
    .findings-val {
        color: #E8F4FD;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #0D1A2E;
        border: 2px dashed #1E3A5C;
        border-radius: 10px;
        padding: 8px;
    }
    [data-testid="stFileUploader"]:hover { border-color: #00C6BE66; }

    /* ── Confidence bar ── */
    .conf-bar-bg {
        background: #0D1A2E;
        border-radius: 4px;
        height: 6px;
        margin-top: 8px;
        overflow: hidden;
    }
    .conf-bar-fill { height: 100%; border-radius: 4px; }

    /* ── Sidebar metadata ── */
    .meta-item { padding: 10px 0; border-bottom: 1px solid #1E2D47; }
    .meta-key  { font-size: 10px; color: #3A6A9A !important; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .meta-val  { font-size: 13px; color: #C0D4E8 !important; margin-top: 2px; font-weight: 500; }

    hr { border-color: #1A2D45 !important; }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# Sidebar — scan metadata
# ─────────────────────────────────────────

with st.sidebar:

    st.markdown(
        """
        <div style='padding: 8px 0 20px 0;'>
            <div style='font-size:20px; font-weight:700; color:#FFFFFF; letter-spacing:-0.3px;'>
                🩺 PancreaSense
            </div>
            <div style='font-size:11px; color:#3A6A9A; margin-top:4px;'>
                Radiology AI Platform v1.0
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-heading' style='margin-top:0;'>Scan Session</div>",
        unsafe_allow_html=True,
    )

    scan_date = datetime.now().strftime("%d %b %Y")
    scan_time = datetime.now().strftime("%H:%M:%S")

    for label, value in [
        ("Date",       scan_date),
        ("Time",       scan_time),
        ("Modality",   "CT — Abdominal"),
        ("Model",      "CNN + U-Net"),
        ("Resolution", "256 × 256 px"),
        ("Framework",  "TensorFlow 2.x"),
    ]:
        st.markdown(
            f"""
            <div class='meta-item'>
                <div class='meta-key'>{label}</div>
                <div class='meta-val'>{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='background:#0A1220; border:1px solid #1A2D45; border-radius:8px;
                    padding:14px; font-size:11px; color:#4A7AAA; line-height:1.6;'>
            ⚠️ <b style='color:#6A8CAE;'>Clinical Disclaimer</b><br><br>
            PancreaSense is an AI-assisted screening tool.
            All outputs must be reviewed by a qualified
            radiologist before clinical decision-making.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────

st.markdown(
    """
    <div class='header-strip'>
        <div class='header-logo'>🩺</div>
        <div>
            <div class='header-title'>PancreaSense</div>
            <div class='header-subtitle'>
                AI-Powered Pancreas Detection &amp; Segmentation · CNN Classification · U-Net Segmentation
            </div>
        </div>
        <div class='header-badge'>🟢 Model Ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# Upload
# ─────────────────────────────────────────

st.markdown("<div class='section-heading'>Upload CT Scan</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop an abdominal CT image (PNG / JPG)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if not uploaded_file:
    st.markdown(
        """
        <div style='background:#0A1220; border:1px solid #1A2D45; border-radius:10px;
                    padding:32px; text-align:center; color:#3A6A9A; font-size:13px;'>
            Upload a CT scan above to begin AI analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────

if uploaded_file:

    with st.spinner("Running CNN classification and U-Net segmentation…"):
        result = run_pipeline(uploaded_file)

    confidence     = result["score"] * 100
    region_percent = (result["area"] / (256 * 256) * 100) if result["area"] else 0
    status_text    = "Detected" if result["detected"] else "Not Detected"
    status_class   = "detected" if result["detected"] else "not-detected"
    status_icon    = "✓" if result["detected"] else "✗"
    bar_color      = "#00C6BE" if result["detected"] else "#F0883E"

    # ── Metrics ──────────────────────────

    st.markdown("<div class='section-heading'>Quantitative Findings</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='metric-grid'>
            <div class='metric-card'>
                <div class='metric-label'>Detection Status</div>
                <div class='metric-value {status_class}'>{status_icon}</div>
                <div class='metric-sub'>{status_text}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Confidence Score</div>
                <div class='metric-value'>{confidence:.1f}<span style='font-size:14px;color:#4A7AAA;'>%</span></div>
                <div class='conf-bar-bg'>
                    <div class='conf-bar-fill' style='width:{confidence:.1f}%; background:linear-gradient(90deg,{bar_color},{bar_color}AA);'></div>
                </div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Segmented Area</div>
                <div class='metric-value'>{result["area"] if result["area"] else 0}</div>
                <div class='metric-sub'>pixels</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Region Coverage</div>
                <div class='metric-value'>{region_percent:.2f}<span style='font-size:14px;color:#4A7AAA;'>%</span></div>
                <div class='metric-sub'>of scan frame</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Image Grid ───────────────────────

    st.markdown("<div class='section-heading'>CT Analysis Imagery</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("<div class='img-panel'><div class='img-panel-label'>Original CT Scan</div>", unsafe_allow_html=True)
        st.image(result["original"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='img-panel'><div class='img-panel-label'>Segmentation Mask</div>", unsafe_allow_html=True)
        if result["mask"] is not None:
            st.image((result["mask"] * 255).astype("uint8"), use_container_width=True)
        else:
            st.markdown("<div style='color:#3A6A9A;font-size:12px;padding:20px 0;'>No mask generated</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='img-panel'><div class='img-panel-label'>Overlay — Pancreas Region</div>", unsafe_allow_html=True)
        if result["overlay"] is not None:
            st.image(result["overlay"], use_container_width=True)
        else:
            st.markdown("<div style='color:#3A6A9A;font-size:12px;padding:20px 0;'>No overlay available</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Findings ─────────────────────────

    st.markdown("<div class='section-heading'>AI Clinical Findings</div>", unsafe_allow_html=True)

    if result["detected"]:
        pill    = "<span class='status-detected'>✓ &nbsp;Pancreas Detected</span>"
        summary = "The AI model has identified a pancreas region in the uploaded CT scan with the confidence and segmentation metrics detailed below."
    else:
        pill    = "<span class='status-not-detected'>✗ &nbsp;Not Detected</span>"
        summary = "No significant pancreas region was identified. The CT image may not contain a clearly visible pancreas, or may differ significantly from the training distribution."

    st.markdown(
        f"""
        <div class='findings-box {"" if result["detected"] else "warning"}'>
            {pill}
            <div style='margin-top:12px; font-size:13px; color:#8AAAC4; line-height:1.6;'>{summary}</div>
            <div style='margin-top:16px;'>
                <div class='findings-row'><span class='findings-key'>Detection Status</span><span class='findings-val'>{status_text}</span></div>
                <div class='findings-row'><span class='findings-key'>Confidence Score</span><span class='findings-val'>{confidence:.2f}%</span></div>
                <div class='findings-row'><span class='findings-key'>Segmented Area</span><span class='findings-val'>{result["area"] if result["area"] else 0} px</span></div>
                <div class='findings-row'><span class='findings-key'>Region Coverage</span><span class='findings-val'>{region_percent:.2f}%</span></div>
                <div class='findings-row'><span class='findings-key'>Analysis Model</span><span class='findings-val'>CNN (cls) + U-Net (seg)</span></div>
                <div class='findings-row'><span class='findings-key'>Scan Date / Time</span><span class='findings-val'>{scan_date} &nbsp;{scan_time}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────

st.markdown(
    """
    <div style='margin-top:48px; padding-top:16px; border-top:1px solid #1A2D45;
                display:flex; justify-content:space-between; align-items:center;
                font-size:11px; color:#2A4A6A;'>
        <span>PancreaSense v1.0 &nbsp;·&nbsp; CNN Classification &nbsp;·&nbsp; U-Net Segmentation &nbsp;·&nbsp; TensorFlow</span>
        <span>AI-Assisted Screening Tool — Not a Substitute for Clinical Diagnosis</span>
    </div>
    """,
    unsafe_allow_html=True,
)