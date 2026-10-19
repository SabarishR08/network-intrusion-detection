"""Tests for utility helper functions."""

import numpy as np
import pandas as pd
from src.utils import ensure_directories, normalise_labels


def test_ensure_directories():
    ensure_directories()


def test_normalise_labels():
    s = pd.Series(["normal", "anomaly", "normal"])
    res = normalise_labels(s)
    assert isinstance(res, pd.Series)
    assert list(res) == [0, 1, 0]

    raw_list = ["normal", "anomaly"]
    res_arr = normalise_labels(raw_list)
    assert isinstance(res_arr, np.ndarray)
    assert list(res_arr) == [0, 1]
