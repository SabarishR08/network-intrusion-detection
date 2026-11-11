"""Shared utility functions and path definitions for the project.

Provides helper routines and directory paths used across training, evaluation,
and prediction workflows:
- Path constants: PROJECT_ROOT, MODEL_DIR, REPORT_DIR, FIGURES_DIR
- ensure_directories: Creates required output directory structures
- save_model_metadata: Serializes model run metrics and timestamps to JSON
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"


def ensure_directories() -> None:
    """Ensure essential project output directories exist."""
    for directory in (MODEL_DIR, REPORT_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_model_metadata(
    model_name: str,
    accuracy: float,
    output_path: str | Path | None = None,
    date: str | None = None,
) -> Path:
    """Save model metadata (model name, date, and accuracy) to a JSON file."""
    ensure_directories()
    if output_path is None:
        output_path = REPORT_DIR / "model_metadata.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    metadata = {
        "model_name": model_name,
        "date": date,
        "accuracy": float(accuracy),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return output_path
