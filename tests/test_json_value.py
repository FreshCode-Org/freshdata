import numpy as np
import pandas as pd

from freshdata.duplicate_defense import _json_value
from freshdata.plan import _is_missing_scalar
from freshdata.plan import _json_value as plan_json_value


def test_json_value_handles_numpy_array_and_series():
    """Regression: duplicate_defense._json_value must not raise on arrays."""
    assert _json_value(np.array([])) == []
    assert _json_value(np.array([1, None, 3])) == [1, None, 3]
    assert _json_value(pd.Series([None, "a", 2])) == [None, "a", 2]


def test_plan_json_value_handles_numpy_array_and_series():
    """Regression: plan._json_value must not raise on arrays."""
    assert plan_json_value(np.array([])) == []
    assert plan_json_value(np.array([1, None, 3])) == [1, None, 3]
    assert plan_json_value(pd.Series([None, "a", 2])) == [None, "a", 2]


def test_is_missing_scalar_rejects_array_like():
    """_is_missing_scalar should return False for containers, not raise."""
    assert _is_missing_scalar(np.array([])) is False
    assert _is_missing_scalar(np.array([1, 2])) is False
    assert _is_missing_scalar(pd.Series([1])) is False
    assert _is_missing_scalar([]) is False
    assert _is_missing_scalar({}) is False
