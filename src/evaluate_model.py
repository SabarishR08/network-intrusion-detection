"""Evaluation helpers for the PS40 intrusion detection project."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def export_evaluation_report(y_true, y_pred, labels: list[str], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=labels, zero_division=0, output_dict=True)
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    payload = {"classification_report": report, "confusion_matrix": confusion}
    (output_dir / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_summary_table(summary: dict, output_path: Path) -> None:
    frame = pd.DataFrame([summary])
    frame.to_csv(output_path, index=False)
