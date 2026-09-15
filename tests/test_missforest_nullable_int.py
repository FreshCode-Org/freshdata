"""MissForest keeps integer dtypes by rounding imputed values (issue #263)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd

pytest.importorskip("sklearn")

ISOLATE = {
    "drop_duplicates": False,
    "drop_empty_rows": False,
    "drop_empty_columns": False,
    "fix_dtypes": False,
    "verbose": False,
}


def _action(report: fd.CleanReport, column: str):
    actions = [a for a in report if a.column == column and a.model_id.startswith("missforest")]
    assert actions, f"no missforest action for {column!r}"
    return actions[-1]


def _int_frame(n: int = 80, dtype: str = "Int64") -> pd.DataFrame:
    x = np.random.default_rng(0).normal(size=n)
    return pd.DataFrame({"a": x, "b": pd.array(np.round(x * 10).astype(int), dtype=dtype)})


def _assert_python_ints(values: list) -> None:
    for v in values:
        assert isinstance(v, (int, np.integer)) and not isinstance(v, bool), v


@pytest.mark.parametrize("dtype", ["Int64", "Int32", "Int16"])
def test_nullable_int_column_keeps_dtype_with_integer_predictions(dtype: str) -> None:
    df = _int_frame(dtype=dtype)
    df.loc[[1, 2, 5], "b"] = pd.NA
    before = df["b"].copy()

    out, report = fd.clean(df, impute="missforest", return_report=True, **ISOLATE)

    assert str(out["b"].dtype) == dtype
    assert out["b"].isna().sum() == 0
    _assert_python_ints(out.loc[[1, 2, 5], "b"].tolist())
    observed = before.notna()
    pd.testing.assert_series_equal(out.loc[observed, "b"], before[observed])

    action = _action(report, "b")
    assert action.model_id == "missforest_regressor"
    assert action.metadata["rounded_to_integer"] is True
    assert "rounded to the nearest integer" in action.rationale
    assert dtype in action.rationale


def test_nullable_int_predictions_are_rounded_regressor_output() -> None:
    df = _int_frame()
    df.loc[[1, 2, 5], "b"] = pd.NA

    out = fd.clean(df, impute="missforest", **ISOLATE)
    as_float = fd.clean(df.astype({"b": "float64"}), impute="missforest", **ISOLATE)

    expected = [int(round(v)) for v in as_float.loc[[1, 2, 5], "b"]]
    assert out.loc[[1, 2, 5], "b"].tolist() == expected


def test_float_column_keeps_fractional_predictions() -> None:
    df = _int_frame().astype({"b": "float64"})
    df.loc[[1, 2, 5], "b"] = np.nan

    out, report = fd.clean(df, impute="missforest", return_report=True, **ISOLATE)

    assert out["b"].dtype == np.float64
    assert any(v != round(v) for v in out.loc[[1, 2, 5], "b"])
    action = _action(report, "b")
    assert action.metadata["rounded_to_integer"] is False
    assert "rounded" not in action.rationale


def test_numpy_int_column_has_nothing_to_impute() -> None:
    # A numpy int64 column cannot hold NaN, so MissForest never touches it;
    # it must come back unchanged while a sibling column is imputed.
    df = _int_frame().astype({"b": "int64"})
    df.loc[[3, 4], "a"] = np.nan

    out, report = fd.clean(df, impute="missforest", return_report=True, **ISOLATE)

    assert out["b"].dtype == np.int64
    pd.testing.assert_series_equal(out["b"], df["b"])
    assert _action(report, "a").metadata["rounded_to_integer"] is False


def test_fallback_on_small_frame_keeps_int_dtype_and_reports_rounding() -> None:
    # Fewer rows than missforest_min_rows_for_model routes to the simple
    # median fallback, which reports the integer rounding.
    df = pd.DataFrame(
        {
            "a": np.arange(6, dtype=float),
            "b": pd.array([1, 2, None, 4, 7, None], dtype="Int64"),
        }
    )

    out, report = fd.clean(df, impute="missforest", return_report=True, **ISOLATE)

    assert out["b"].dtype == "Int64"
    # median of [1, 2, 4, 7] is 3.0
    assert out["b"].tolist() == [1, 2, 3, 4, 7, 3]
    action = _action(report, "b")
    assert action.model_id == "missforest_fallback"
    assert action.metadata["fallback_reason"]
    assert action.metadata["rounded_to_integer"] is True
    assert "rounded to the nearest integer" in action.rationale


def test_fallback_rounds_half_median_to_even_integer() -> None:
    df = pd.DataFrame(
        {
            "a": np.arange(5, dtype=float),
            "b": pd.array([1, 2, None, 5, 8], dtype="Int64"),
        }
    )

    out = fd.clean(df, impute="missforest", **ISOLATE)

    # median of [1, 2, 5, 8] is 3.5 -> half-to-even 4
    assert out["b"].dtype == "Int64"
    assert out["b"].tolist() == [1, 2, 4, 5, 8]
    _assert_python_ints(out["b"].tolist())


def test_fallback_keeps_values_beyond_float64_exact_range() -> None:
    big = 2**60
    df = pd.DataFrame(
        {
            "a": np.arange(5, dtype=float),
            "b": pd.array([big + 1, big + 3, None, big + 6, big + 8], dtype="Int64"),
        }
    )

    out = fd.clean(df, impute="missforest", **ISOLATE)

    assert out["b"].dtype == "Int64"
    # exact median of the present values is big + 4.5 -> half-to-even big + 4;
    # a float64 median would have lost the low digits entirely
    assert out["b"].tolist() == [big + 1, big + 3, big + 4, big + 6, big + 8]


def test_regressor_keeps_present_values_beyond_float64_exact_range() -> None:
    big = 2**60
    df = _int_frame()
    df["b"] = pd.array([big + 2 * int(v) + 1 for v in df["b"]], dtype="Int64")
    before = df["b"].copy()
    df.loc[[1, 2, 5], "b"] = pd.NA

    out = fd.clean(df, impute="missforest", **ISOLATE)

    assert out["b"].dtype == "Int64"
    observed = df["b"].notna()
    assert out.loc[observed, "b"].tolist() == before[observed].tolist()
    _assert_python_ints(out.loc[[1, 2, 5], "b"].tolist())
