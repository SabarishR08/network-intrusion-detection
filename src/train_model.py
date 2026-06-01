"""Training entry point for the PS40 intrusion detection project."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_preprocessing import LABEL_MAP, build_preprocessor, load_datasets, split_features_targets
from .feature_engineering import align_features
from .utils import FIGURES_DIR, MODEL_DIR, REPORT_DIR, configure_logging, ensure_directories, write_json

matplotlib.use("Agg")
LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42


def build_models() -> dict[str, object]:
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.08,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    except Exception:
        LOGGER.info("XGBoost not available; skipping optional model.")
    return models


def safe_auc(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")


def plot_confusion_matrix(y_true, y_pred, output_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Anomaly", "Normal"])
    ax.set_yticks([0, 1], ["Anomaly", "Normal"])
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, cm[row, col], ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_roc_curve(y_true, y_score, output_path: Path, label: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#00B8D9", linewidth=2.2, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9AA4B2", linewidth=1, label="Random")
    ax.fill_between(fpr, tpr, color="#00B8D9", alpha=0.08)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(pipeline: Pipeline, output_path: Path) -> None:
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    if not hasattr(classifier, "feature_importances_"):
        return
    feature_names = preprocessor.get_feature_names_out()
    importances = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    importances.sort_values().plot(kind="barh", ax=ax, color="#2563EB")
    ax.set_title("Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train() -> dict:
    ensure_directories()
    configure_logging()
    datasets = load_datasets()
    train_frame, test_frame = datasets.train_frame.copy(), datasets.test_frame.copy()

    train_features, train_targets = split_features_targets(train_frame)
    test_features = test_frame.drop(columns=["class"], errors="ignore").copy()
    train_features, val_features, train_targets, val_targets = train_test_split(
        train_features,
        train_targets,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_targets,
    )

    models = build_models()
    results: dict[str, dict] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    shared_preprocessor = build_preprocessor(train_features)
    train_matrix = shared_preprocessor.fit_transform(train_features)
    val_matrix = shared_preprocessor.transform(val_features)

    LOGGER.info("Training candidate models")
    for name, classifier in models.items():
        fitted_classifier = clone(classifier)
        fitted_classifier.fit(train_matrix, train_targets)
        predictions = fitted_classifier.predict(val_matrix)
        probabilities = fitted_classifier.predict_proba(val_matrix)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(val_targets, predictions)),
            "precision": float(precision_score(val_targets, predictions, zero_division=0)),
            "recall": float(recall_score(val_targets, predictions, zero_division=0)),
            "f1": float(f1_score(val_targets, predictions, zero_division=0)),
            "roc_auc": float(safe_auc(val_targets, probabilities)),
            "classification_report": classification_report(
                val_targets,
                predictions,
                labels=[0, 1],
                target_names=["anomaly", "normal"],
                zero_division=0,
            ),
        }
        fitted_pipelines[name] = Pipeline(
            steps=[
                ("preprocessor", shared_preprocessor),
                ("classifier", fitted_classifier),
            ]
        )
        LOGGER.info(
            "%s | accuracy=%.4f f1=%.4f roc_auc=%.4f",
            name,
            results[name]["accuracy"],
            results[name]["f1"],
            results[name]["roc_auc"],
        )

    best_name = max(results, key=lambda key: results[key]["f1"])
    best_pipeline = fitted_pipelines[best_name]

    full_features, full_targets = split_features_targets(train_frame)
    final_preprocessor = build_preprocessor(full_features)
    full_matrix = final_preprocessor.fit_transform(full_features)
    final_classifier = clone(models[best_name])
    final_classifier.fit(full_matrix, full_targets)
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", final_preprocessor),
            ("classifier", final_classifier),
        ]
    )

    val_predictions = fitted_pipelines[best_name].named_steps["classifier"].predict(val_matrix)
    val_probabilities = fitted_pipelines[best_name].named_steps["classifier"].predict_proba(val_matrix)[:, 1]

    plot_confusion_matrix(
        val_targets,
        val_predictions,
        FIGURES_DIR / "confusion_matrix.png",
        f"Validation Confusion Matrix - {best_name}",
    )
    plot_roc_curve(
        val_targets,
        val_probabilities,
        FIGURES_DIR / "roc_curve.png",
        f"{best_name} (AUC={results[best_name]['roc_auc']:.4f})",
    )
    plot_feature_importance(best_pipeline, FIGURES_DIR / "feature_importance.png")

    test_predictions = final_pipeline.predict(test_features)
    test_probabilities = final_pipeline.predict_proba(test_features)[:, 1]
    test_frame_output = test_frame.copy()
    test_frame_output["predicted_label"] = pd.Series(test_predictions).map({0: "anomaly", 1: "normal"})
    test_frame_output["anomaly_probability"] = 1 - test_probabilities
    test_frame_output["normal_probability"] = test_probabilities

    train_report = {
        "problem_statement": "PS40 - Network Intrusion Detection",
        "best_model": best_name,
        "model_comparison": results,
        "validation": {
            "accuracy": float(accuracy_score(val_targets, val_predictions)),
            "precision": float(precision_score(val_targets, val_predictions, zero_division=0)),
            "recall": float(recall_score(val_targets, val_predictions, zero_division=0)),
            "f1": float(f1_score(val_targets, val_predictions, zero_division=0)),
            "roc_auc": float(safe_auc(val_targets, val_probabilities)),
        },
        "prediction_summary": test_frame_output["predicted_label"].value_counts().to_dict(),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "label_map": LABEL_MAP,
    }

    model_payload = {
        "pipeline": final_pipeline,
        "best_model": best_name,
        "feature_columns": list(full_features.columns),
        "metrics": train_report,
    }

    joblib.dump(model_payload, MODEL_DIR / "model.pkl")
    joblib.dump(final_pipeline.named_steps["preprocessor"], MODEL_DIR / "encoders.pkl")
    try:
        joblib.dump(final_pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"], MODEL_DIR / "scaler.pkl")
    except Exception:
        joblib.dump(None, MODEL_DIR / "scaler.pkl")

    test_frame_output.to_csv(REPORT_DIR / "predictions.csv", index=False)
    write_json(REPORT_DIR / "metrics.json", train_report)

    return {
        "model_payload": model_payload,
        "test_predictions": test_frame_output,
        "metrics": train_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PS40 network intrusion detection models.")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1]), help="Project root.")
    parser.parse_args()
    result = train()
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()