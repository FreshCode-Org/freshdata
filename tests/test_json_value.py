import numpy as np
import pandas as pd

from freshdata.duplicate_defense import _json_value


def test_json_value_handles_numpy_array_and_series():
    assert _json_value(np.array([])) == []
    assert _json_value(np.array([1, None, 3])) == [1, None, 3]
    assert _json_value(pd.Series([None, "a", 2])) == [None, "a", 2]
