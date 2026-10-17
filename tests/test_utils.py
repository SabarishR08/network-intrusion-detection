import pytest
import pandas as pd
from src.utils import validate_dataframe


def test_validate_dataframe_valid():
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    assert validate_dataframe(df, ["col1", "col2"]) is True


def test_validate_dataframe_missing_column():
    df = pd.DataFrame({"col1": [1, 2]})
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_dataframe(df, ["col1", "missing_col"])


def test_validate_dataframe_invalid_type():
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        validate_dataframe(None, ["col1"])
