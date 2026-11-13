'''Feature engineering helpers for the intrusion detection pipeline.

This module contains utility functions to process and align feature data for the intrusion detection system.
The align_features function ensures that a DataFrame contains all expected columns, adding missing ones filled with NA values.'''
from __future__ import annotations

import logging
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)


def align_features(frame: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
    aligned = frame.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            logger.debug("Adding missing column: %s", column)
            aligned[column] = pd.NA
    return aligned[expected_columns]
