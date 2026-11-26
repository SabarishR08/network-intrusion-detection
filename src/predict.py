"""Prediction functionality for the PS40 Network Intrusion Detection system.

This module provides functions to load a trained model bundle, make predictions on
single samples or entire DataFrames, and includes a command‑line entry point
(`main`) that can be invoked to run predictions from the terminal.
"""

from __future__ import annotations

import time
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    from .utils import MODEL_DIR
except ImportError:
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

    parser = argparse.ArgumentParser(description="Predict using")
    # Remaining implementation unchanged
    ...
