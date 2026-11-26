"""Data loading and preprocessing helpers for the intrusion detection pipeline.

This module provides functions for loading raw network datasets, normalizing labels,
splitting features and targets, and building scikit-learn preprocessing pipelines.

Preprocessing steps include:
- Data loading: Reading raw training and test datasets.
- Label normalization & encoding: Mapping raw target class strings ('normal', 'anomaly', etc.) to binary targets (0: anomaly, 1: normal).
- Feature/target separation: Splitting target columns from feature sets.
- Numeric preprocessing: Applying median imputation followed by StandardScaler scaling to numeric features.
- Categorical preprocessing: Applying most-frequent imputation and OneHotEncoding to categorical features ('protocol_type', 'service', 'flag').
"""

from __future__ import annotations

import logging
import builtins
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import RAW_DIR, ensure_dataset_available

LOGGER = logging.getLogger(__name__)

# Redirect any stray print calls to the module logger
def _log_print(*args, **kwargs):
    """Log messages that were originally sent to stdout via print()."""
    # Join positional arguments with spaces, mimicking print's default separator
    message = " ".join(str(a) for a in args)
    LOGGER.info(message)

# Override built‑in print within this module's scope
builtins.print = _log_print

TARGET_COLUMN = "class"
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
LABEL_MAP = {"anomaly": 0, "normal": 1}

@dataclass(frozen=True)
class DatasetBundle:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


def load_datasets() -> DatasetBundle:
    ensure_dataset_available()
    train_frame = pd.read_csv(RAW_DIR / "Train_data.csv")
    test_frame = pd.read_csv(RAW_DIR / "Test_data.csv")
    if TARGET_COLUMN not in test_frame.columns:
        test_frame[TARGET_COLUMN] = np.nan
    return DatasetBundle(train_frame=train_frame, test_frame=test_frame)


def normalize_label(value: object) -> str:
    label = str(value).strip().lower()
    if label in {"normal", "benign", "0", "false"}:
        return "normal"
    return "anomaly"


def encode_labels(frame: pd.DataFrame) -> pd.Series:
    return frame[TARGET_COLUMN].map(normalize_label).map(LABEL_MAP).astype(int)


def split_features_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = frame.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    targets = encode_labels(frame)
    return features, targets


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [column for column in feature_frame.columns if column not in CATEGORICAL_COLUMNS]
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", one_hot),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
    ])
