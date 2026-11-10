"""Prediction functionality for the PS40 Network Intrusion Detection system."""

from __future__ import annotations

import time
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from utils import MODEL_DIR

# Module level logger
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    label: str
    confidence: float
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_bundle(model_dir: Path | str | None = None) -> dict[str, Any]:
    path = Path(model_dir) if model_dir else MODEL_DIR
    model = joblib.load(path / "model.pkl")
    scaler = joblib.load(path / "scaler.pkl")
    encoders = joblib.load(path / "encoders.pkl")
    return {"model": model, "scaler": scaler, "encoders": encoders}


def predict_single(sample: dict[str, Any] | pd.Series, bundle: dict[str, Any]) -> PredictionResult:
    start_time = time.perf_counter()
    if isinstance(sample, dict):
        df = pd.DataFrame([sample])
    else:
        df = pd.DataFrame([sample.to_dict()])

    preds, probs = predict_frame(df, bundle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    label = str(preds[0])
    confidence = float(probs[0]) if probs is not None and len(probs) > 0 else 1.0

    return PredictionResult(label=label, confidence=confidence, elapsed_ms=round(elapsed_ms, 2))


def predict_frame(df: pd.DataFrame, bundle: dict[str, Any]) -> tuple[list[str], list[float]]:
    model = bundle["model"]
    scaler = bundle.get("scaler")
    encoders = bundle.get("encoders", {})

    data = df.copy()

    for col, enc in encoders.items():
        if col in data.columns:
            try:
                data[col] = enc.transform(data[col].astype(str))
            except Exception:
                pass

    if scaler is not None and hasattr(scaler, "transform"):
        try:
            features = scaler.transform(data)
        except Exception:
            features = data
    else:
        features = data

    preds = model.predict(features)

    if hasattr(model, "predict_proba"):
        probs_array = model.predict_proba(features)
        probs = [float(max(p)) for p in probs_array]
    else:
        probs = [1.0] * len(preds)

    return list(preds), probs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict using trained NIDS model")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV")
    args = parser.parse_args()

    logger.info("Loading model bundle")
    bundle = load_bundle()

    logger.info("Reading input data from %s", args.input)
    df_input = pd.read_csv(args.input)

    logger.info("Running predictions")
    preds, probs = predict_frame(df_input, bundle)

    df_input["prediction"] = preds
    df_input["confidence"] = probs

    logger.info("Writing predictions to %s", args.output)
    df_input.to_csv(args.output, index=False)

    logger.info("Prediction completed")

if __name__ == "__main__":
    main()
