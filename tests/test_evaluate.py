"""Tests for model evaluation logging and metrics."""

from __future__ import annotations

import logging
import numpy as np
from src.evaluate_model import evaluate_model


def test_evaluate_model_logs_confusion_matrix(caplog) -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])

    with caplog.at_level(logging.INFO):
        metrics = evaluate_model(y_true, y_pred)

    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert "Confusion matrix summary:" in caplog.text
