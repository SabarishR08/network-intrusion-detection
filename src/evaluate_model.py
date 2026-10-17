"""Model evaluation module for logging metrics and confusion matrix summaries."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(y_true: Any, y_pred: Any, y_prob: Any | None = None) -> dict[str, Any]:
    """Evaluate model predictions and log confusion matrix summary."""
    cm = confusion_matrix(y_true, y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    logger.info("Confusion matrix summary:\n%s", cm)
    logger.info("Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f", acc, prec, rec, f1)

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    return metrics
