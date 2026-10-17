"""Utility functions, retry helpers, and path management for PS40 Network Intrusion Detection."""

from __future__ import annotations

import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

logger = logging.getLogger(__name__)


def ensure_directories() -> None:
    """Ensure essential project directories exist."""
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


RT = TypeVar("RT")


def retry_file_io(
    fn: Callable[..., Any] | None = None,
    *,
    max_retries: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (FileNotFoundError, OSError, IOError),
) -> Any:
    """Retry wrapper/decorator for file I/O operations that may fail on missing or busy datasets."""

    def decorator(func: Callable[..., RT]) -> Callable[..., RT]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> RT:
            current_delay = delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as err:
                    if attempt == max_retries:
                        logger.error(
                            "File I/O operation %s failed after %d attempts: %s",
                            getattr(func, "__name__", str(func)),
                            max_retries,
                            err,
                        )
                        raise
                    logger.warning(
                        "Attempt %d/%d failed for %s (%s). Retrying in %.2fs...",
                        attempt,
                        max_retries,
                        getattr(func, "__name__", str(func)),
                        err,
                        current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            return func(*args, **kwargs)

        return wrapper

    if fn is not None and callable(fn):
        return decorator(fn)
    return decorator


def read_file_with_retry(
    reader_fn: Callable[..., RT],
    *args: Any,
    max_retries: int = 3,
    delay: float = 0.1,
    **kwargs: Any,
) -> RT:
    """Execute a file reading function (e.g. pd.read_csv) with automatic retry logic."""
    wrapped = retry_file_io(reader_fn, max_retries=max_retries, delay=delay)
    return wrapped(*args, **kwargs)
