"""Data loading and preprocessing helpers for the intrusion detection pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import RAW_DIR, ensure_dataset_available

LOGGER = logging.getLogger(__name__)
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

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", one_hot),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
