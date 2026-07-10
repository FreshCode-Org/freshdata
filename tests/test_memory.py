import numpy as np
import pandas as pd

import freshdata as fd
from freshdata import _util


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
    out, report = fd.clean(df, optimize_memory=True, return_report=True)
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
    _, report = fd.clean(df, optimize_memory=True, return_report=True)
    assert report.memory_after < report.memory_before


def test_memory_bytes_sampling_excludes_index_overhead(monkeypatch):
    """Regression for #35: the sampled extrapolation must not re-count the
    index payload once per string-like column, and must count it once."""
    monkeypatch.setattr(_util, "_MEMORY_SAMPLE_THRESHOLD", 100)
    monkeypatch.setattr(_util, "_MEMORY_SAMPLE_SIZE", 50)

    n = 200
    index = pd.Index([f"row-{i:04d}-{'x' * 40}" for i in range(n)])
    df = pd.DataFrame(
        {"a": ["ab"] * n, "b": ["cd"] * n, "c": ["ef"] * n},
        index=index,
    )
    exact = int(df.memory_usage(deep=True).sum())
    estimate = _util.memory_bytes(df)
    # Constant-width values mean sampling should land within a few percent of
    # the exact measurement.  Before the fix the heavy string index leaked
    # into every column's payload delta, inflating the estimate ~1.7x.
    assert exact * 0.9 <= estimate <= exact * 1.1


def test_memory_bytes_sampling_matches_exact_for_range_index(monkeypatch):
    """The common RangeIndex case must stay accurate after the fix."""
    monkeypatch.setattr(_util, "_MEMORY_SAMPLE_THRESHOLD", 100)
    monkeypatch.setattr(_util, "_MEMORY_SAMPLE_SIZE", 50)

    n = 200
    df = pd.DataFrame({"a": ["ab"] * n, "b": np.arange(n, dtype="int64")})
    exact = int(df.memory_usage(deep=True).sum())
    estimate = _util.memory_bytes(df)
    assert exact * 0.9 <= estimate <= exact * 1.1
