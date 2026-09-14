"""Median imputation on nullable integer columns.

pandas computes a masked-integer median by filling masked slots with ``iNaT``;
on newer numpy that overflows Int8/Int16/Int32 and unsigned dtypes
(``OverflowError: Python integer ... out of bounds for int8``). Every median
imputation path must survive those columns and keep Int64/float behaviour.
"""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata._util import exact_int_stat, exceeds_float64_exact, fill_na_exact, safe_median
from freshdata.engine import missing, model_select
from freshdata.engine.utils import _has_outliers

NARROW = ["Int8", "Int16", "Int32", "UInt8", "UInt64"]
KEEP_ROWS = {"drop_empty_rows": False, "drop_duplicates": False, "verbose": False}


def _series(dtype: str, n: int = 40) -> pd.Series:
    # 37 non-missing values -> the median is an observed (integer) value.
    s = pd.Series(pd.array([i % 7 for i in range(n)], dtype=dtype))
    s.iloc[[3, 11, 25]] = pd.NA
    return s


def _expected_median(s: pd.Series) -> float:
    return float(np.median(s.dropna().to_numpy(dtype="float64")))


@pytest.mark.parametrize("dtype", [*NARROW, "Int64", "Float32", "float64"])
def test_safe_median_matches_float_median(dtype):
    s = _series(dtype)
    assert safe_median(s) == _expected_median(s)


def test_safe_median_leaves_non_integer_dtypes_untouched():
    s = _series("Float32")
    value = safe_median(s)
    assert type(value) is type(s.median())
    assert value == s.median()


@pytest.mark.parametrize("strategy", ["median", "auto"])
@pytest.mark.parametrize("dtype", [*NARROW, "Int64"])
def test_explicit_impute_median_on_nullable_int(dtype, strategy):
    s = _series(dtype)
    df = pd.DataFrame({"v": s})
    out, report = fd.clean(
        df, impute=strategy, strategy="conservative", return_report=True, **KEEP_ROWS
    )
    assert out["v"].isna().sum() == 0
    assert out["v"].iloc[3] == _expected_median(s)
    assert any(a.step == "impute" and strategy in a.description for a in report)


@pytest.mark.parametrize("dtype", [*NARROW, "Int64"])
def test_fill_missing_median_on_nullable_int(dtype):
    s = _series(dtype)
    out = fd.fill_missing(pd.DataFrame({"v": s}), method="median")
    assert out["v"].isna().sum() == 0
    assert out["v"].iloc[3] == _expected_median(s)


@pytest.mark.parametrize("dtype", NARROW)
def test_default_engine_survives_nullable_int_with_outliers(dtype):
    rng = np.random.default_rng(0)
    values = rng.integers(10, 60, 100)
    values[:4] = 120  # outliers -> the engine's low band picks median
    s = pd.Series(pd.array(values, dtype=dtype))
    s.iloc[[10, 20, 30]] = pd.NA
    df = pd.DataFrame({"v": s, "x": rng.normal(0, 1, 100)})
    out, report = fd.clean(df, return_report=True, **KEEP_ROWS)
    filled = [a for a in report if a.step == "missing" and a.column == "v"]
    assert filled, "the engine must record a decision for the column"
    if "median" in filled[0].description:
        assert out["v"].isna().sum() == 0


@pytest.mark.parametrize("dtype", ["Int16", "UInt8"])
def test_seasonal_imputation_global_median_on_nullable_int(dtype):
    idx = pd.date_range("2024-01-01", periods=96, freq="h")
    s = pd.Series(pd.array([10 + (i % 5) for i in range(96)], dtype=dtype))
    s.iloc[[5, 40, 72]] = pd.NA
    df = pd.DataFrame({"t": idx, "v": s})
    out = fd.clean_timeseries(
        df, timestamp_column="t", max_interpolation_gap=0,
        seasonal_period="hour", seasonal_imputation_enabled=True,
    )
    assert out["v"].isna().sum() == 0


BIG = 2**53 + 1  # float64 rounds this to 2**53


def test_exact_int_stat_and_detection():
    s = pd.Series(pd.array([BIG, 0, 1, None], dtype="Int64"))
    assert exceeds_float64_exact(s)
    assert not exceeds_float64_exact(pd.Series(pd.array([2**53, None], dtype="Int64")))
    assert not exceeds_float64_exact(pd.Series([float(BIG), np.nan]))
    assert exact_int_stat(s, "mean") == round(Fraction(BIG + 1, 3))
    assert exact_int_stat(s, "median") == 1
    even = pd.Series(pd.array([BIG, BIG + 2, None, BIG + 5, BIG + 7], dtype="Int64"))
    # (BIG+2 + BIG+5) / 2 == 2**53 + 4.5, which rounds half-to-even to 2**53 + 4
    assert exact_int_stat(even, "median") == BIG + 3


def test_fill_na_exact_keeps_small_int_behaviour():
    s = pd.Series(pd.array([1, None, 2], dtype="Int64"))
    filled, note = fill_na_exact(s, 1.5)
    assert filled.tolist() == [1.0, 1.5, 2.0]
    assert note == ", column cast to float64"
    filled, note = fill_na_exact(s, 1)
    assert str(filled.dtype) == "Int64" and note == ""


@pytest.mark.parametrize("impute", ["mean", "median", "auto"])
def test_explicit_impute_keeps_int64_beyond_2_53_exact(impute):
    df = pd.DataFrame({"x": pd.array([BIG, 0, 1, None], dtype="Int64"), "y": [1.0, 2.0, 3.0, 4.0]})
    out, report = fd.clean(
        df, impute=impute, strategy="conservative", return_report=True, **KEEP_ROWS
    )
    assert str(out["x"].dtype) == "Int64"
    assert out["x"].iloc[:3].tolist() == [BIG, 0, 1]  # present values untouched
    expected = round(Fraction(BIG + 1, 3)) if impute == "mean" else 1
    assert out["x"].iloc[3] == expected
    notes = [a.description for a in report if a.step == "impute" and a.column == "x"]
    assert notes and "2**53" in notes[0]


def test_default_engine_keeps_int64_beyond_2_53_exact():
    base = 2**60
    values = pd.array([base + (i % 7) for i in range(60)], dtype="Int64")
    s = pd.Series(values)
    s.iloc[[3, 17]] = pd.NA
    df = pd.DataFrame({"v": s, "x": np.random.default_rng(0).normal(0, 1, 60)})
    out, report = fd.clean(df, return_report=True, **KEEP_ROWS)
    assert str(out["v"].dtype) == "Int64"
    present = s.notna()
    assert out["v"][present].tolist() == s[present].tolist()
    filled = [a for a in report if a.step == "missing" and a.column == "v"]
    if filled and "filled" in filled[0].description:
        assert out["v"].isna().sum() == 0
        assert base <= int(out["v"].iloc[3]) <= base + 6
        assert "2**53" in filled[0].description


def test_has_outliers_single_definition():
    assert missing._has_outliers is _has_outliers
    assert model_select._has_outliers is _has_outliers
    assert _has_outliers(pd.Series([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert not _has_outliers(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
