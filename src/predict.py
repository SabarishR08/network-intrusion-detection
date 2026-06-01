"""Inference helper for scoring uploaded traffic rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from .feature_engineering import align_features
    from .utils import MODEL_DIR
except ImportError:  # pragma: no cover - supports Streamlit's top-level imports
    from feature_engineering import align_features  # type: ignore
    from utils import MODEL_DIR  # type: ignore


def load_bundle(model_path: Path | None = None) -> dict:
    path = model_path or (MODEL_DIR / "model.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found at {path}. Run training first.")
    bundle = joblib.load(path)
    # Basic validation of bundle content
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "feature_columns" not in bundle:
        raise ValueError(f"Model bundle at {path} is missing required keys (pipeline, feature_columns).")
    return bundle


def predict_frame(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]
    features = frame.drop(columns=["class"], errors="ignore")
    features = align_features(features, feature_columns)
    if features.empty:
        raise ValueError("Uploaded CSV does not contain usable feature columns.")
    try:
        predictions = pipeline.predict(features)
    except Exception as exc:
        raise RuntimeError(f"Model prediction failed: {exc}")

    try:
        probabilities = pipeline.predict_proba(features)[:, 1]
    except Exception:
        # Some models may not implement predict_proba
        probabilities = pd.Series([float("nan")] * len(predictions))

    output = frame.copy()
    output["predicted_label"] = pd.Series(predictions).map({0: "anomaly", 1: "normal"})
    output["anomaly_probability"] = 1 - probabilities
    output["normal_probability"] = probabilities
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict intrusion labels for a CSV file.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    parser.add_argument("--model", default=str(MODEL_DIR / "model.pkl"), help="Path to the saved model bundle.")
    args = parser.parse_args()

    bundle = load_bundle(Path(args.model))
    frame = pd.read_csv(args.input)
    predictions = predict_frame(frame, bundle)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()