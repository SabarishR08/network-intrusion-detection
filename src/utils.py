"""Shared utilities for the PS40 Network Intrusion Detection project."""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SOURCE_ZIP = PROJECT_ROOT.parent / "Network Intrusion Detection.zip"
ROOT_ZIP = PROJECT_ROOT / "Network Intrusion Detection.zip"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, FIGURES_DIR, SCREENSHOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def candidate_dataset_paths() -> list[Path]:
    return [
        RAW_DIR / "Train_data.csv",
        RAW_DIR / "Test_data.csv",
        PROJECT_ROOT / "data" / "Train_data.csv",
        PROJECT_ROOT / "data" / "Test_data.csv",
        PROJECT_ROOT.parent / "data" / "Train_data.csv",
        PROJECT_ROOT.parent / "data" / "Test_data.csv",
    ]


def ensure_dataset_available() -> None:
    ensure_directories()
    train_exists = (RAW_DIR / "Train_data.csv").exists()
    test_exists = (RAW_DIR / "Test_data.csv").exists()
    if train_exists and test_exists:
        return

    for fallback in [SOURCE_ZIP, ROOT_ZIP]:
        if fallback.exists():
            with zipfile.ZipFile(fallback) as archive:
                archive.extractall(RAW_DIR)
            return

    source_dir = PROJECT_ROOT.parent / "Network Intrusion Detection"
    if source_dir.exists():
        for name in ["Train_data.csv", "Test_data.csv"]:
            source_file = source_dir / name
            if source_file.exists():
                (RAW_DIR / name).write_bytes(source_file.read_bytes())
        if (RAW_DIR / "Train_data.csv").exists() and (RAW_DIR / "Test_data.csv").exists():
            return

    raise FileNotFoundError("Could not locate Train_data.csv and Test_data.csv.")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def flatten(items: Iterable[Iterable[str]]) -> list[str]:
    return [value for group in items for value in group]
