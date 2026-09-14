"""Polars backend behaviour: ingestion, streaming, pushdown, fallback."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.execution import EngineConfig
from freshdata.execution._config import FallbackError

pl = pytest.importorskip("polars")


def test_accepts_lazy_frame(small_df, native_config):
    lf = pl.from_pandas(small_df).lazy()
    out = fd.clean(lf, config=native_config, engine="polars", output_format="polars")
    assert isinstance(out, pl.DataFrame)


def test_accepts_polars_frame_returns_pandas_by_default(small_df, native_config):
    pf = pl.from_pandas(small_df)
    out = fd.clean(pf, config=native_config, engine="polars")
    assert isinstance(out, pd.DataFrame)  # output_format defaults to pandas


def test_accepts_parquet_path(parquet_10k, native_config):
    out = fd.clean(parquet_10k, config=native_config, engine="polars")
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0


def test_streaming_true_and_false(small_df, native_config):
    pf = pl.from_pandas(small_df)
    for streaming in (True, False):
        ec = EngineConfig(engine="polars", streaming=streaming)
        out = fd.clean(pf, config=native_config, engine_config=ec)
        assert isinstance(out, (pd.DataFrame,))


def test_strip_whitespace_and_sentinels(native_config):
    # id column keeps the sentinel row from becoming all-null (and thus dropped)
    df = pd.DataFrame({"id": [1, 2, 3, 4], "name": [" alice", "bob ", "N/A", "carol"]})
    out = fd.clean(df, config=native_config, engine="polars", output_format="polars")
    vals = out.sort("id")["name"].to_list()
    assert vals[0] == "alice" and vals[1] == "bob"
    assert vals[2] is None  # sentinel -> null


def test_drop_empty_column(native_config):
    df = pd.DataFrame({"a": [1, 2, 3], "empty": [None, None, None]})
    out = fd.clean(df, config=native_config, engine="polars", output_format="polars")
    assert "empty" not in out.columns


def test_fallback_on_knn_like_config_warns(small_df, caplog):
    """A config needing the decision engine falls back to pandas with a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="freshdata.execution.polars"):
        out = fd.clean(small_df.copy(), engine="polars")  # default balanced -> fallback
    assert isinstance(out, pd.DataFrame)
    assert any("falling back to pandas" in r.message for r in caplog.records)


def test_thread_config_does_not_raise(small_df, native_config):
    ec = EngineConfig(engine="polars", polars_n_threads=2)
    out = fd.clean(small_df.copy(), config=native_config, engine_config=ec)
    assert isinstance(out, pd.DataFrame)


def test_projection_pushdown_drops_empty_before_collect(native_config):
    """Empty columns are removed; result has fewer columns than the input."""
    df = pd.DataFrame({"keep": [1, 2], "gone": [None, None]})
    out = fd.clean(df, config=native_config, engine="polars", output_format="polars")
    assert list(out.columns) == ["keep"]


def _native_nan_source():
    return pl.DataFrame({
        "a": [1.0, float("nan"), 3.0],
        "b": [float("nan")] * 3,
        "s": ["x", None, "y"],
    })


def _steps(report):
    return [(a.step, a.count) for a in report.actions]


@pytest.mark.parametrize(
    "make_source",
    [lambda df: df, lambda df: df.lazy(), lambda df: df.to_arrow()],
    ids=["dataframe", "lazyframe", "arrow"],
)
def test_native_nan_counts_as_missing_like_pandas(native_config, make_source):
    src = _native_nan_source()
    ref_out, ref = fd.clean(src.to_pandas(), config=native_config, engine="pandas",
                            return_report=True)
    out, report = fd.clean(make_source(src), config=native_config, engine="polars",
                           return_report=True)
    assert list(out.columns) == list(ref_out.columns) == ["a", "s"]
    assert report.rows_after == ref.rows_after == 2
    assert report.missing_before == ref.missing_before == 5
    assert _steps(report) == _steps(ref)


def test_polars_written_parquet_nan_counts_as_missing(tmp_path, native_config):
    path = str(tmp_path / "nan.parquet")
    _native_nan_source().write_parquet(path)
    out, report = fd.clean(path, config=native_config, engine="polars", return_report=True)
    assert list(out.columns) == ["a", "s"]
    assert report.missing_before == 5


def test_infinite_values_are_excluded_from_outlier_fences():
    config = CleanConfig(strategy="conservative", fix_dtypes=False, verbose=False,
                         outliers="clip", outlier_method="iqr")
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0] + [float("inf")] * 4})
    ref, ref_report = fd.clean(df, config=config, engine="pandas", return_report=True)
    out, report = fd.clean(pl.from_pandas(df), config=config, engine="polars",
                           return_report=True)
    assert out["x"].tolist() == ref["x"].tolist()
    assert _steps(report) == _steps(ref_report)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"engine": "auto"}, {"engine": "polars", "strategy": "balanced"},
     {"engine": "polars", "strategy": "conservative"}],
    ids=["default", "auto", "polars-balanced", "polars-conservative"],
)
def test_lazy_frame_source_survives_pandas_fallback(kwargs):
    lf = pl.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", "z"]}).lazy()
    out = fd.clean(lf, verbose=False, **kwargs)
    frame = out.collect() if isinstance(out, pl.LazyFrame) else out
    assert len(frame) == 3


def test_lazy_frame_fallback_still_honours_error_policy():
    lf = pl.DataFrame({"a": [1.0, None, 3.0]}).lazy()
    with pytest.raises(FallbackError):
        fd.clean(lf, engine="polars", strategy="balanced", fallback_policy="error",
                 verbose=False)
