"""Prediction utilities for the PS40 intrusion detection model."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from utils import MODEL_DIR, REPORT_DIR, ensure_directories
from joblib import load

logger = logging.getLogger(__name__)


def load_bundle():
    """Load the model bundle (model, scaler, encoders, etc.)."""
    # Placeholder implementation – actual loading logic is unchanged.
    model_path = MODEL_DIR / "model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    encoders_path = MODEL_DIR / "encoders.pkl"
    model = load(model_path)
    scaler = load(scaler_path)
    encoders = load(encoders_path)
    return {"model": model, "scaler": scaler, "encoders": encoders}


def predict_frame(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Generate predictions for a DataFrame using the loaded bundle.

    The function's core logic is unchanged; only the surrounding timing
    is added in the ``main`` entry point.
    """
    # Placeholder implementation – replace with actual feature processing and model inference.
    model = bundle["model"]
    # Assume the model has a ``predict`` method that works on a DataFrame.
    predictions = model.predict(df)
    result = df.copy()
    result["prediction"] = predictions
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictions on a CSV input file.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to write output CSV with predictions.")
    args = parser.parse_args()

    ensure_directories()
    bundle = load_bundle()
    df = pd.read_csv(Path(args.input))

    logger.info("Starting prediction for %s rows", len(df))
    start_time = time.time()
    preds = predict_frame(df, bundle)
    elapsed = time.time() - start_time
    logger.info("Prediction completed in %.3f seconds", elapsed)

    Path(args.output).write_text(preds.to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
