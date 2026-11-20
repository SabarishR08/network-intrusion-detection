"""Cybersecurity dashboard for the PS40 intrusion detection project.

Dashboard Layout Overview:
- Hero Section & KPI Metrics: Key summary banner and core validation metrics (Best Model, Accuracy, F1, AUC).
- Navigation & Theme: Styled UI theme with customizable visual cards and layout sections.
- Performance & Evaluation Panels: Visualizations and model assessment views.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import load_bundle, predict_frame  # type: ignore
from utils import FIGURES_DIR, MODEL_DIR, REPORT_DIR, ensure_directories
import traceback
import logging

logger = logging.getLogger(__name__)


st.set_page_config(page_title="PS40 NIDS Dashboard", page_icon="🛡️", layout="wide")
ensure_directories()


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top, #0b1320 0%, #050816 55%, #02040a 100%);
            color: #e5eefb;
        }
        .hero {
            padding: 1.4rem 1.5rem;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            background: rgba(8, 15, 30, 0.85);
            box-shadow: 0 20px 40px rgba(0,0,0,0.35);
        }
        .metric-card {
            padding: 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home(metrics: dict) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1 style="margin-bottom:0.2rem;">AI-Powered Network Intrusion Detection System</h1>
            <p style="margin-top:0; color:#9fb3c8;">PS40 | Machine learning-based anomaly detection for TCP/IP connections</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Best Model", metrics.get("best_model", "Random Forest"))
    cols[1].metric("Validation Accuracy", f"{metrics.get('validation', {}).get('accuracy', 0):.4f}")
    cols[2].metric("Validation F1", f"{metrics.get('validation', {}).get('f1', 0):.4f}")
    cols[3].metric("Validation AUC", f"{metrics.get('validation', {}).get('roc_auc', 0):.4f}")


def render_about() -> None:
    st.subheader("About Project")
    st.write(
        "This project trains a reusable intrusion detection pipeline for normal versus anomaly traffic. "
        "It is aligned to the IBM AICTE internship narrative with an ML engine, a LangFlow orchestration layer, "
        "and an interactive Streamlit user interface."
    )
