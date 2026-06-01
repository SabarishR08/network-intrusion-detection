"""Cybersecurity dashboard for the PS40 intrusion detection project."""

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
        "and an agentic AI explanation layer for Granite-based reasoning."
    )
    st.markdown("- Reusable sklearn pipeline\n- Validation dashboard\n- Prediction export\n- IBM-ready documentation")


def render_upload_predict(bundle: dict) -> None:
    st.subheader("Upload & Predict")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if not uploaded_file:
        st.info("Upload a CSV with the 41 NSL-KDD style features to score traffic rows.")
        return

    try:
        frame = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read uploaded CSV: {exc}")
        logger.exception("Failed to read uploaded CSV")
        return

    try:
        output = predict_frame(frame, bundle)
        st.success("Inference completed")
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        with st.expander("Error details"):
            st.text(traceback.format_exc())
        logger.exception("Inference failed")
        return

    summary = output["predicted_label"].value_counts().rename_axis("label").reset_index(name="count")
    anomaly_percentage = 100 * (output["predicted_label"].eq("anomaly").mean())
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows Scored", len(output))
    c2.metric("Anomaly %", f"{anomaly_percentage:.2f}%")
    c3.metric("Normal %", f"{100 - anomaly_percentage:.2f}%")

    st.dataframe(summary, use_container_width=True)
    st.dataframe(output.head(20), use_container_width=True)

    st.download_button(
        "Download predictions CSV",
        output.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

    malicious_rows = output[output["predicted_label"] == "anomaly"].head(10)
    st.write("Sample malicious rows")
    st.dataframe(malicious_rows, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    summary.set_index("label")["count"].plot(kind="bar", ax=ax, color=["#2dd4bf", "#f97316"])
    ax.set_title("Prediction Summary")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)


def render_analytics() -> None:
    st.subheader("Analytics Dashboard")
    plots = [
        (FIGURES_DIR / "confusion_matrix.png", "Confusion Matrix"),
        (FIGURES_DIR / "roc_curve.png", "ROC Curve"),
        (FIGURES_DIR / "feature_importance.png", "Feature Importance"),
    ]
    columns = st.columns(3)
    for column, (plot_path, caption) in zip(columns, plots):
        if plot_path.exists():
            column.image(str(plot_path), caption=caption, use_container_width=True)
        else:
            column.info(f"{caption} not generated yet. Run the training pipeline first.")


def main() -> None:
    apply_theme()
    try:
        bundle = load_bundle(MODEL_DIR / "model.pkl") if (MODEL_DIR / "model.pkl").exists() else {"best_model": "Random Forest", "validation": {}}
    except Exception as exc:
        bundle = {"best_model": "Random Forest", "validation": {}}
        st.sidebar.error(f"Could not load model bundle: {exc}")
        logger.exception("Failed to load model bundle")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About Project", "Upload & Predict", "Analytics Dashboard"])

    if page == "Home":
        render_home(bundle.get("metrics", {}))
    elif page == "About Project":
        render_about()
    elif page == "Upload & Predict":
        if (MODEL_DIR / "model.pkl").exists():
            render_upload_predict(bundle)
        else:
            st.warning("Train the model first to enable prediction.")
    else:
        render_analytics()


if __name__ == "__main__":
    main()