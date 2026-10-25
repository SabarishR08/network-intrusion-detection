from __future__ import annotations

import logging
import pickle
from pathlib import Path

# Existing constants (preserved)
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def ensure_directories() -> None:
    """Create essential project directories if they do not exist."""
    for d in (FIGURES_DIR, MODEL_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_model_with_fallback(model_path: str | Path) -> object | None:
    """Load a pickled model, returning ``None`` and logging a warning if the file is missing.

    Parameters
    ----------
    model_path: str | Path
        Path to the model file (typically a ``.pkl`` file).

    Returns
    -------
    object | None
        The deserialized model object, or ``None`` when the file does not exist.
    """
    path = Path(model_path)
    logger = logging.getLogger(__name__)
    if not path.is_file():
        logger.warning("Model file %s not found; returning None.", path)
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load model from %s: %s", path, exc)
        return None
