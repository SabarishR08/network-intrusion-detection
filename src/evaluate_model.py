"""Evaluation module for the intrusion detection pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class TrainingMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_metrics(y_true: Any, y_pred: Any) -> TrainingMetrics:
    """Calculate evaluation metrics for model predictions."""
    return TrainingMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )
