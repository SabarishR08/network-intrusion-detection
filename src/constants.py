"""
Constants used across the project for data preprocessing, feature engineering,
model training, and prediction.
"""

from typing import Dict, List

# Target Column
TARGET_COLUMN: str = "target"

# Feature Column Names
NUMERICAL_FEATURES: List[str] = [
    "age",
    "tenure",
    "balance",
    "num_of_products",
    "estimated_salary",
]

CATEGORICAL_FEATURES: List[str] = [
    "geography",
    "gender",
    "has_cr_card",
    "is_active_member",
]

FEATURE_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Label Mappings
LABEL_MAPPING: Dict[int, str] = {
    0: "No",
    1: "Yes",
}

REVERSE_LABEL_MAPPING: Dict[str, int] = {
    label: id_ for id_, label in LABEL_MAPPING.items()
}

# Default Classification Threshold
DEFAULT_THRESHOLD: float = 0.5
