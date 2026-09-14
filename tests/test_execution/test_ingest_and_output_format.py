"""Inputs native engines cannot ingest (#206) and engine/output_format pairs (#205)."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata import CleanConfig
from freshdata.execution import EngineConfig, FallbackError

pytest.importorskip("polars")
pytest.importorskip("duckdb")

NATIVE = CleanConfig(strategy="conservative", fix_dtypes=False, verbose=False)
ENGINES = ("polars", "duckdb")


def _reasons(report):
    return [event["fallback_reason"] for event in report.fallback_events]


# -- #206 inputs that need the pandas reference --------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_mixed_object_column_falls_back_with_pandas_values(engine):
    df = pd.DataFrame({"m": [1, "a", None], "k": [1.0, 2.0, 3.0]})
    ref = fd.clean(df, config=NATIVE, engine="pandas")
    out, report = fd.clean(df, config=NATIVE, engine=engine, return_report=True)
    assert [type(v).__name__ for v in out["m"].dropna()] == ["int", "str"]
    # engine="pandas" returns a CleanResult wrapper; compare the plain frames.
    pd.testing.assert_frame_equal(pd.DataFrame(out), pd.DataFrame(ref))
    assert report.backend == "pandas"
    assert any("mixes value types" in reason for reason in _reasons(report))


@pytest.mark.parametrize("engine", ENGINES)
def test_duplicate_labels_fall_back_with_pandas_columns(engine):
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["x", "x"])
    ref = fd.clean(df, config=NATIVE, engine="pandas")
    out, report = fd.clean(df, config=NATIVE, engine=engine, return_report=True)
    assert list(out.columns) == list(ref.columns) == ["x", "x_2"]
    assert any("duplicate input column labels" in reason for reason in _reasons(report))


@pytest.mark.parametrize("engine", ENGINES)
def test_ingest_fallback_is_blocked_by_error_policy(engine):
    df = pd.DataFrame({"m": [1, "a", None]})
    with pytest.raises(FallbackError, match="mixes value types"):
        fd.clean(df, config=NATIVE, engine=engine, fallback_policy="error")


@pytest.mark.parametrize("engine", ENGINES)
def test_single_type_object_columns_stay_native(engine):
    df = pd.DataFrame({"s": ["a", None, "b"], "k": [1.0, 2.0, 3.0]})
    _, report = fd.clean(df, config=NATIVE, engine=engine, return_report=True)
    assert report.backend == engine
    assert report.fallback_events == []


# -- #205 native handle formats belong to one engine ---------------------------


@pytest.mark.parametrize(
    ("engine", "output_format"),
    [
        ("duckdb", "polars-lazy"),
        ("polars", "duckdb"),
        ("pandas", "duckdb"),
        ("pandas", "polars-lazy"),
        ("spark", "polars-lazy"),
        ("freshcore", "duckdb"),
    ],
)
def test_engine_config_rejects_a_foreign_native_handle(engine, output_format):
    with pytest.raises(ValueError, match="native"):
        EngineConfig(engine=engine, output_format=output_format)


@pytest.mark.parametrize(
    ("engine", "output_format"), [("duckdb", "polars-lazy"), ("polars", "duckdb")]
)
def test_clean_rejects_a_foreign_native_handle_under_error_policy(engine, output_format):
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    with pytest.raises(ValueError, match="native"):
        fd.clean(df, config=NATIVE, engine=engine, output_format=output_format,
                 fallback_policy="error")


@pytest.mark.parametrize(
    ("output_format", "handle"),
    [("duckdb", "DuckDBPyRelation"), ("polars-lazy", "LazyFrame")],
)
@pytest.mark.parametrize("engine", ["auto", None], ids=["auto", "default"])
def test_handle_format_selects_its_own_engine(output_format, handle, engine):
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    kwargs = {} if engine is None else {"engine": engine}
    out, report = fd.clean(df, config=NATIVE, output_format=output_format,
                           return_report=True, **kwargs)
    assert type(out).__name__ == handle
    assert report.fallback_events == []


@pytest.mark.parametrize(
    ("engine", "output_format"), [("duckdb", "duckdb"), ("polars", "polars-lazy")]
)
def test_handle_request_with_a_recorded_fallback_returns_pandas(engine, output_format):
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    # The default balanced strategy runs on the pandas reference.
    out, report = fd.clean(df, engine=engine, output_format=output_format,
                           return_report=True, verbose=False)
    assert isinstance(out, pd.DataFrame)
    assert report.fallback_events
