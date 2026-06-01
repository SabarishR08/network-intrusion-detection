"""Feature engineering helpers for the intrusion detection pipeline."""

from __future__ import annotations

import pandas as pd


def align_features(frame: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    aligned = frame.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned[expected_columns]
