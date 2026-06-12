import numpy as np
import pandas as pd

import freshdata as fd


def is_string(dtype) -> bool:
    return pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)


def test_dtypes_untouched_by_default():
    df = pd.DataFrame({"i": np.arange(100, dtype="int64"),
                       "f": np.linspace(0, 1, 100),
                       "c": ["a", "b"] * 50})
    out = fd.clean(df)
    assert out["i"].dtype == "int64"
    assert out["f"].dtype == "float64"
    assert is_string(out["c"].dtype)


def test_optimize_downcasts_and_categorizes():
    df = pd.DataFrame({"i": np.arange(100, dtype="int64"),
                       "f": np.linspace(0, 1, 100).astype("float64"),
                       "c": ["a", "b"] * 50})
    out, report = fd.clean(df, optimize_memory=True, report=True)
    assert out["i"].dtype == "int8"
    assert out["f"].dtype == "float32"
    assert str(out["c"].dtype) == "category"
    assert any(a.step == "optimize_memory" for a in report)


def test_nullable_int_downcast():
    df = pd.DataFrame({"v": pd.array([1, None, 120], dtype="Int64")})
    out = fd.clean(df, optimize_memory=True, drop_empty_rows=False)
    assert out["v"].dtype == "Int8"
    assert out["v"].isna().sum() == 1


def test_high_cardinality_text_stays_object():
    df = pd.DataFrame({"id": [f"user_{i}" for i in range(100)]})
    out = fd.clean(df, optimize_memory=True)
    assert is_string(out["id"].dtype)


def test_category_threshold_configurable():
    df = pd.DataFrame({"c": [f"v{i % 30}" for i in range(100)]})  # ratio 0.3
    as_cat = fd.clean(df, optimize_memory=True, drop_duplicates=False)
    assert str(as_cat["c"].dtype) == "category"
    kept = fd.clean(df, optimize_memory=True, category_threshold=0.1,
                    drop_duplicates=False)
    assert is_string(kept["c"].dtype)


def test_memory_reported_smaller():
    df = pd.DataFrame({"i": np.arange(10_000, dtype="int64"),
                       "c": ["x", "y"] * 5_000})
    _, report = fd.clean(df, optimize_memory=True, report=True)
    assert report.memory_after < report.memory_before
