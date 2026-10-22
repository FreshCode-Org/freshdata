# tests/test_engine_extension_dtypes.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig


@pytest.mark.parametrize("include_na", [False, True])
def test_nullable_extension_dtypes_roundtrip_across_engines(include_na: bool) -> None:
    """Fixes #465: All pandas nullable extension dtypes must round-trip identically across engines."""
    na = pd.NA if include_na else None

    data = {
        "int8_col": pd.Series([1, 2, 1 if not include_na else na, 4], dtype="Int8"),
        "int16_col": pd.Series([10, 20, 10 if not include_na else na, 40], dtype="Int16"),
        "int32_col": pd.Series([100, 200, 100 if not include_na else na, 400], dtype="Int32"),
        "int64_col": pd.Series([1000, 2000, 1000 if not include_na else na, 4000], dtype="Int64"),
        "uint8_col": pd.Series([1, 2, 1 if not include_na else na, 4], dtype="UInt8"),
        "uint16_col": pd.Series([10, 20, 10 if not include_na else na, 40], dtype="UInt16"),
        "uint32_col": pd.Series([100, 200, 100 if not include_na else na, 400], dtype="UInt32"),
        "uint64_col": pd.Series([1000, 2000, 1000 if not include_na else na, 4000], dtype="UInt64"),
        "float32_col": pd.Series([1.5, 2.5, 1.5 if not include_na else na, 4.5], dtype="Float32"),
        "float64_col": pd.Series([10.5, 20.5, 10.5 if not include_na else na, 40.5], dtype="Float64"),
        "string_col": pd.Series(["alpha", "beta", "alpha" if not include_na else na, "delta"], dtype="string"),
        "bool_col": pd.Series([True, False, True if not include_na else na, False], dtype="boolean"),
        "cat_col": pd.Series(
            pd.Categorical(
                ["low", "med", "low" if not include_na else None, "high"],
                categories=["low", "med", "high"],
                ordered=True,
            )
        ),
        "dt_tz_col": pd.Series(
            pd.date_range("2026-01-01", periods=4, tz="UTC")
        ),
    }
    if include_na:
        data["dt_tz_col"].iloc[2] = pd.NaT

    df = pd.DataFrame(data)
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False, drop_duplicates=False)

    res_pd, rep_pd = fd.clean(df, config=cfg, engine="pandas", return_report=True)
    res_pl, rep_pl = fd.clean(df, config=cfg, engine="polars", return_report=True)
    res_ddb, rep_ddb = fd.clean(df, config=cfg, engine="duckdb", return_report=True)

    assert res_pd.dtypes.to_dict() == df.dtypes.to_dict()
    assert res_pd.dtypes.to_dict() == res_pl.dtypes.to_dict() == res_ddb.dtypes.to_dict()

    pd.testing.assert_frame_equal(res_pd, res_pl, check_dtype=True, check_categorical=True)
    pd.testing.assert_frame_equal(res_pd, res_ddb, check_dtype=True, check_categorical=True)
    assert rep_pd.dtype_changes == {}
    assert rep_pl.dtype_changes == {}
    assert rep_ddb.dtype_changes == {}


def test_fix_dtypes_promotion_logs_and_uses_nullable_extension_dtypes() -> None:
    """When fix_dtypes=True promotes dirty string numbers with nulls, promote to nullable extension types and log."""
    df = pd.DataFrame({
        "dirty_int": pd.Series([" 10 ", "20", "N/A", "40"], dtype="object"),
        "dirty_bool": pd.Series(["true", "false", "null", "true"], dtype="object"),
    })
    cfg = CleanConfig(strategy="conservative", fix_dtypes=True, impute_missing=False)

    res_pd, rep_pd = fd.clean(df, config=cfg, engine="pandas", return_report=True)
    res_pl, rep_pl = fd.clean(df, config=cfg, engine="polars", return_report=True)
    res_ddb, rep_ddb = fd.clean(df, config=cfg, engine="duckdb", return_report=True)

    assert isinstance(res_pd["dirty_int"].dtype, pd.Int64Dtype)
    assert isinstance(res_pd["dirty_bool"].dtype, pd.BooleanDtype)
    assert "dirty_int" in rep_pd.dtype_changes
    assert "dirty_bool" in rep_pd.dtype_changes
    pd.testing.assert_frame_equal(res_pd, res_pl, check_dtype=True)
    pd.testing.assert_frame_equal(res_pd, res_ddb, check_dtype=True)
