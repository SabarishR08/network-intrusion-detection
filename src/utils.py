"""Utility functions and directory management for the intrusion detection pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

LABEL_MAPPING = {"normal": 0, "anomaly": 1}


def ensure_directories() -> None:
    """Ensure all required project directories exist."""
    for directory in [DATA_DIR, MODEL_DIR, REPORT_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def normalise_labels(y: Any) -> Any:
    """Map raw label strings or values to integer class indices.

    Args:
        y: Series, array-like, or list of raw labels (e.g. 'normal', 'anomaly').

    Returns:
        Mapped integer indices as pd.Series or np.ndarray matching input structure.
    """
    is_series = isinstance(y, pd.Series)
    s = y if is_series else pd.Series(y)

    s_str = s.astype(str).str.lower().str.strip()
    if s_str.isin(LABEL_MAPPING.keys()).any():
        mapped = s_str.map(LABEL_MAPPING)
        res = mapped.fillna(1).astype(int)
    elif pd.api.types.is_numeric_dtype(s):
        res = s.astype(int)
    else:
        codes, _ = pd.factorize(s)
        res = pd.Series(codes, index=s.index, dtype=int)

    return res if is_series else res.to_numpy()
