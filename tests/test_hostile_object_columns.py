# tests/test_hostile_object_columns.py
from __future__ import annotations

from decimal import Decimal
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import KNNImputer, SimpleImputer

import freshdata as fd
from freshdata.config import CleanConfig


def test_hostile_object_column_never_raises_type_error() -> None:
    """Fixes #507: Object columns mixing complex, dict, list, set, bytes, Decimal, NA, and strings must not crash."""
    df = pd.DataFrame({
        "mixed_bag": pd.Series(
            [1 + 2j, {"k": 1}, [1, 2], {3, 4}, b"raw_bytes", Decimal("19.99"), np.nan, pd.NA, None, " 42.0 "],
            dtype="object",
        ),
        "clean_num": pd.Series([10.0, 20.0, 30.0, np.nan, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], dtype="Float64"),
    })
    cfg = CleanConfig(strategy="balanced", fix_dtypes=True, impute_missing=True, handle_outliers=True)
    cleaned, report = fd.clean(df, config=cfg, return_report=True)

    assert len(cleaned) == len(df)
    assert cleaned["mixed_bag"].dtype == object
    assert cleaned["mixed_bag"].iloc[9] == "42.0"  # string element stripped safely
    assert cleaned["clean_num"].isna().sum() == 0


def test_knn_fill_nullable_int_and_zero_variance_safety() -> None:
    """Fixes #470 / PR #487: _knn_fill on nullable Int64, boolean, and zero-variance columns must not crash or yield inf."""
    df = pd.DataFrame({
        "feature_a": pd.Series([1, 2, 3, 4, pd.NA, 6], dtype="Int64"),
        "feature_b": pd.Series([10, 20, 30, 40, 50, 60], dtype="Int64"),
        "constant_col": pd.Series([5, 5, 5, 5, 5, 5], dtype="Int64"),
        "all_null_col": pd.Series([pd.NA] * 6, dtype="Float64"),
        "flag": pd.Series([True, False, True, False, True, pd.NA], dtype="boolean"),
    })
    cfg = CleanConfig(strategy="balanced", impute_missing=True, imputer="knn", fix_dtypes=False)
    cleaned, report = fd.clean(df, config=cfg, return_report=True)

    # Compare against sklearn KNNImputer baseline on the valid numeric slice
    knn = KNNImputer(n_neighbors=2)
    baseline_ab = knn.fit_transform(df[["feature_a", "feature_b"]].to_numpy(dtype=np.float64, na_value=np.nan))

    assert cleaned["feature_a"].isna().sum() == 0
    assert np.isclose(float(cleaned["feature_a"].iloc[4]), baseline_ab[4, 0], atol=1.0)
    assert not np.isinf(cleaned["feature_a"].to_numpy(dtype=np.float64)).any()


def test_never_imputes_integral_float_or_high_cardinality_identifier() -> None:
    """Fixes #488: Never impute integral-float or monotonic/unique identifier columns even without 'id' in the name."""
    df = pd.DataFrame({
        "record_seq": [1001.0, 1002.0, np.nan, 1004.0, 1005.0, 1006.0, 1007.0, 1008.0],
        "account_num": pd.Series([90001, 90002, 90003, pd.NA, 90005, 90006, 90007, 90008], dtype="Int64"),
        "metric": [12.5, 13.0, np.nan, 12.8, 13.1, 12.9, 13.2, 12.7],
    })
    cfg = CleanConfig(strategy="balanced", impute_missing=True, fix_dtypes=False)
    cleaned, report = fd.clean(df, config=cfg, return_report=True)

    # Identifiers must keep their exact missingness
    assert pd.isna(cleaned["record_seq"].iloc[2])
    assert pd.isna(cleaned["account_num"].iloc[3])
    # Genuine continuous metric is imputed
    assert cleaned["metric"].isna().sum() == 0

    skipped_reasons = {entry["column"]: entry["reason"] for entry in report.skipped_columns}
    assert "record_seq" in skipped_reasons
    assert "skipped_identifier_imputation" in skipped_reasons["record_seq"]
    assert "account_num" in skipped_reasons
    assert "skipped_identifier_imputation" in skipped_reasons["account_num"]


def test_hostile_object_column_engine_fallbacks() -> None:
    """Non-pandas engines must gracefully fall back to pandas when encountering unconvertible python objects."""
    df = pd.DataFrame({
        "hostile": pd.Series([{"a": 1}, [2, 3], {4, 5}, b"raw"], dtype="object"),
        "val": pd.Series([1, 2, 3, 4], dtype="Int64"),
    })
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False)

    for eng in ["polars", "duckdb"]:
        cleaned, report = fd.clean(df, config=cfg, engine=eng, return_report=True)
        assert len(cleaned) == len(df)
        assert any(ev["reason"] == "hostile_python_objects_require_pandas_object_handling" for ev in report.fallback_events)


def test_outlier_clipping_and_categorical_sentinels() -> None:
    """Verify outlier clipping and categorical sentinels normalization branches."""
    df = pd.DataFrame({
        "num_outliers": pd.Series([10.0, 11.0, 10.5, 11.2, 1000.0], dtype="Float64"),
        "int_outliers": pd.Series([5, 6, 5, 6, 500], dtype="Int64"),
        "cat_col": pd.Series(pd.Categorical([" good ", " N/A ", " bad ", " good "])),
        "order_codes": pd.Series(["ORD-001", "ORD-002", "ORD-003", pd.NA, "ORD-005"], dtype="string"),
    })
    cfg = CleanConfig(
        strategy="balanced",
        handle_outliers=True,
        normalize_sentinels=True,
        strip_whitespace=True,
        impute_missing=True,
    )
    cleaned, report = fd.clean(df, config=cfg, return_report=True)

    assert report.outliers_clipped >= 2
    assert "order_codes" in [sc["column"] for sc in report.skipped_columns]


def test_outlier_and_duplicate_warning_and_knn_edge_cases() -> None:
    """Cover outlier target skipping, duplicate warnings, and knn edge cases."""
    df_dup = pd.DataFrame({"a": [1, 1, 2], "target": [10.0, 10.0, 1000.0]})
    cfg = CleanConfig(drop_duplicates=False, handle_outliers=True, target_column="target")
    _, rep = fd.clean(df_dup, config=cfg, return_report=True)
    assert any("duplicate" in w for w in rep.warnings)

    from freshdata.imputation import _knn_fill
    s_full, c1 = _knn_fill(pd.DataFrame({"x": [1, 2, 3]}), "x")
    assert c1 == 0
    s_missing, c2 = _knn_fill(pd.DataFrame({"str_col": ["a", None]}), "str_col")
    assert c2 == 0

