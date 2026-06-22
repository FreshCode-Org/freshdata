import pandas as pd

from ..steps.outliers import detection_bounds

def _has_outliers(s: pd.Series) -> bool:
    bounds = detection_bounds(s, "iqr", 1.5)
    if bounds is None:
        return False
    return bool(((s < bounds[0]) | (s > bounds[1])).any())
