"""Polars adapter round-trip tests."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from expectations import ALL_ONLINE_TIER1, load_online_fixture

pytest.importorskip("polars")
import polars as pl  # noqa: E402

from freshdata.adapters.polars import to_pandas


@pytest.mark.parametrize("name", ALL_ONLINE_TIER1[:5])
def test_polars_round_trip_parity(name):
    pdf = load_online_fixture(name)
    pl_df = pl.from_pandas(pdf)
    out_pd = fd.clean(pdf, strategy="balanced", verbose=False)
    out_pl = fd.clean(pl_df, strategy="balanced", verbose=False)
    out_pl_pd = out_pl.to_pandas()
    assert out_pd.shape == out_pl_pd.shape
    assert list(out_pd.columns) == list(out_pl_pd.columns)


def test_polars_return_type():
    pdf = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    pl_df = pl.from_pandas(pdf)
    result = fd.clean(pl_df, verbose=False)
    assert isinstance(result, pl.DataFrame)


def test_infer_roles_accepts_polars():
    pdf = pd.DataFrame({"customer_id": [1, 2, 3], "amount": [1.0, 2.0, 3.0]})
    pl_df = pl.from_pandas(pdf)
    roles = fd.infer_roles(pl_df)
    assert len(roles) == 2


def test_to_pandas_names_pyarrow_when_missing(monkeypatch):
    """The polars→pandas interchange needs pyarrow; a bare `[polars]`-era
    install crashed with polars' internal ModuleNotFoundError. The adapter
    must name the fix instead."""
    def boom(self, *a, **k):
        raise ModuleNotFoundError("No module named 'pyarrow'")

    monkeypatch.setattr(pl.DataFrame, "to_pandas", boom)
    with pytest.raises(ModuleNotFoundError, match=r"freshdata-cleaner\[polars\]"):
        to_pandas(pl.DataFrame({"a": [1]}))


def test_lazy_frame_round_trips_through_default_clean():
    lf = pl.DataFrame({"a": [1.0, None, 3.0], "b": ["x", " y", "z"]}).lazy()
    out = fd.clean(lf, verbose=False)
    assert isinstance(out, pl.LazyFrame)  # LazyFrame in, LazyFrame out
    assert out.collect()["b"].to_list() == ["x", "y", "z"]


# -- #444 integer columns holding nulls must not go through float64 ------------


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        ([2**53 + 1, None, 7], pl.Int64, "Int64"),
        ([2**63 + 1, None, 7], pl.UInt64, "UInt64"),
        ([5, None, 7], pl.Int32, "Int32"),
    ],
)
def test_to_pandas_keeps_integers_exact_when_the_column_has_nulls(values, dtype, expected):
    """pl.to_pandas() renders these as float64, rounding past 2**53 (#444)."""
    out = to_pandas(pl.DataFrame({"v": pl.Series(values, dtype=dtype)}))
    assert str(out["v"].dtype) == expected
    assert out["v"].tolist()[0] == values[0]
    assert out["v"].isna().tolist() == [False, True, False]


def test_to_pandas_leaves_integer_columns_without_nulls_alone():
    out = to_pandas(pl.DataFrame({"v": pl.Series([2**53 + 1, 3], dtype=pl.Int64)}))
    assert str(out["v"].dtype) == "int64"
    assert out["v"].tolist() == [2**53 + 1, 3]


def test_clean_does_not_round_large_integers_from_a_polars_frame():
    """The default engine reads a polars source through the same adapter (#444)."""
    df = pl.DataFrame({"v": pl.Series([2**53 + 1, None, 7], dtype=pl.Int64), "k": [1.0, 2.0, 3.0]})
    out = fd.clean(df, verbose=False)  # polars in, polars out
    assert out.schema["v"] == pl.Int64
    assert out["v"].to_list() == [2**53 + 1, None, 7]


def test_native_polars_engine_returns_exact_integers():
    """The native path converts its result with the same adapter (#444)."""
    pytest.importorskip("polars")
    df = pd.DataFrame({"v": pd.array([2**53 + 1, None, 7], dtype="Int64"), "k": [1.0, 2.0, 3.0]})
    out, report = fd.clean(
        df,
        strategy="conservative",
        fix_dtypes=False,
        verbose=False,
        engine="polars",
        return_report=True,
    )
    assert report.backend == "polars"
    assert str(out["v"].dtype) == "Int64"
    assert out["v"].tolist()[0] == 2**53 + 1
