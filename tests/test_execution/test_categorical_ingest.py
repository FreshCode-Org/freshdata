"""Categorical pandas inputs retain text repairs and category metadata (#456)."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.execution import FallbackError, FallbackWarning
from freshdata.execution._ingest import pandas_ingest_fallback_reason
from freshdata.execution.backends._duckdb import DuckDBEngine
from freshdata.execution.backends._polars import PolarsEngine

pytest.importorskip("polars")
pytest.importorskip("duckdb")

ENGINES = ("polars", "duckdb")
KW = {"strategy": "conservative", "fix_dtypes": False, "verbose": False}


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "c": pd.Categorical([" beta", "alpha ", "N/A"], ordered=True),
        "id": [1, 2, 3],
    })


@pytest.mark.parametrize("engine", ENGINES)
def test_categorical_text_repairs_match_pandas_and_are_disclosed(engine):
    frame = _frame()
    original = frame.copy(deep=True)
    expected, reference = fd.clean(frame, engine="pandas", return_report=True, **KW)
    out, report = fd.clean(frame, engine=engine, return_report=True, **KW)

    pd.testing.assert_frame_equal(pd.DataFrame(out), pd.DataFrame(expected))
    pd.testing.assert_frame_equal(frame, original)
    assert out["c"].iloc[:2].tolist() == ["beta", "alpha"]
    assert pd.isna(out["c"].iloc[2])
    assert [(a.step, a.count) for a in report.actions] == [
        (a.step, a.count) for a in reference.actions
    ] == [("strip_whitespace", 2), ("normalize_sentinels", 1)]
    assert report.backend == "pandas"
    assert report.requested_backend == engine
    (event,) = report.fallback_events
    assert event["fallback_step"] == "pipeline"
    assert "categorical column 'c'" in event["fallback_reason"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("values", [
    pytest.param(pd.Categorical(["a ", "a", " N/A", None], ordered=True), id="merged-categories"),
    pytest.param(pd.Categorical([1, " a ", "N/A", None]), id="mixed-categories"),
    pytest.param(
        pd.Categorical([2, 1, None], categories=[2, 1, 3], ordered=True), id="numeric-order"
    ),
    pytest.param(
        pd.Categorical([None, None], categories=[" unused "], ordered=True), id="all-missing"
    ),
    pytest.param(pd.Categorical([], categories=[" unused "], ordered=True), id="empty"),
])
def test_category_domain_order_and_values_match_pandas(engine, values):
    frame = pd.DataFrame({"c": values, "id": range(len(values))})
    expected = fd.clean(frame, engine="pandas", drop_empty_columns=False, **KW)
    out, report = fd.clean(
        frame, engine=engine, drop_empty_columns=False, return_report=True, **KW
    )
    pd.testing.assert_frame_equal(pd.DataFrame(out), pd.DataFrame(expected))
    assert isinstance(out["c"].dtype, pd.CategoricalDtype)
    assert report.fallback_events


@pytest.mark.parametrize("engine", ENGINES)
def test_disabled_text_steps_preserve_the_category_domain(engine):
    frame = _frame()
    out, report = fd.clean(
        frame, engine=engine, strip_whitespace=False, normalize_sentinels=False,
        return_report=True, **KW,
    )
    pd.testing.assert_frame_equal(pd.DataFrame(out), frame)
    assert report.actions == []
    assert report.fallback_events


@pytest.mark.parametrize("engine", ENGINES)
def test_categorical_fallback_honours_warn_policy(engine):
    with pytest.warns(FallbackWarning, match="categorical column"):
        out = fd.clean(_frame(), engine=engine, fallback_policy="warn", **KW)
    assert out["c"].iloc[0] == "beta"


@pytest.mark.parametrize(("engine", "backend", "ingest"), [
    ("polars", PolarsEngine, "_to_lazy"),
    ("duckdb", DuckDBEngine, "_register_source"),
])
def test_error_policy_blocks_ingestion_and_pandas_work(monkeypatch, engine, backend, ingest):
    def forbidden(*args, **kwargs):
        raise AssertionError("strict fallback policy must reject before cleaning")

    monkeypatch.setattr(backend, ingest, forbidden)
    monkeypatch.setattr(backend, "_fallback", forbidden)
    with pytest.raises(FallbackError, match="categorical column 'c'"):
        fd.clean(_frame(), engine=engine, fallback_policy="error", **KW)


@pytest.mark.parametrize(("engine", "output_format"), [
    ("polars", "polars-lazy"), ("duckdb", "duckdb"),
])
def test_native_handle_requests_disclose_materialized_pandas_fallback(engine, output_format):
    out, report = fd.clean(
        _frame(), engine=engine, output_format=output_format, return_report=True, **KW
    )
    assert isinstance(out, pd.DataFrame)
    assert report.materialized is True
    assert report.backend == "pandas"
    assert report.fallback_events


def _arrow_frame(ordered=False) -> pd.DataFrame:
    pa = pytest.importorskip("pyarrow")
    array = pa.array([" beta", "alpha ", "N/A", None]).dictionary_encode().cast(
        pa.dictionary(pa.int8(), pa.string(), ordered=ordered)
    )
    return pd.DataFrame({"c": pd.Series(pd.arrays.ArrowExtensionArray(array)), "id": [1, 2, 3, 4]})


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("ordered", [False, True])
def test_arrow_dictionary_inputs_preserve_pandas_behavior_and_dtype(engine, ordered):
    frame = _arrow_frame(ordered)
    original = frame.copy(deep=True)
    expected, reference = fd.clean(frame, engine="pandas", return_report=True, **KW)

    out, report = fd.clean(frame, engine=engine, return_report=True, **KW)

    pd.testing.assert_frame_equal(pd.DataFrame(out), pd.DataFrame(expected))
    pd.testing.assert_frame_equal(frame, original)
    assert out["c"].dtype == frame["c"].dtype
    assert [(a.step, a.count) for a in report.actions] == [
        (a.step, a.count) for a in reference.actions
    ]
    assert report.backend == "pandas"
    assert report.requested_backend == engine
    assert "categorical column 'c'" in report.fallback_events[0]["fallback_reason"]


@pytest.mark.parametrize("engine", ENGINES)
def test_arrow_dictionary_fallback_honours_warn_policy(engine):
    with pytest.warns(FallbackWarning, match="categorical column"):
        fd.clean(_arrow_frame(), engine=engine, fallback_policy="warn", **KW)


@pytest.mark.parametrize(("engine", "backend", "ingest"), [
    ("polars", PolarsEngine, "_to_lazy"),
    ("duckdb", DuckDBEngine, "_register_source"),
])
def test_arrow_dictionary_error_policy_blocks_ingestion_and_pandas_work(
    monkeypatch, engine, backend, ingest
):
    def forbidden(*args, **kwargs):
        raise AssertionError("strict fallback policy must reject before cleaning")

    monkeypatch.setattr(backend, ingest, forbidden)
    monkeypatch.setattr(backend, "_fallback", forbidden)
    with pytest.raises(FallbackError, match="categorical column 'c'"):
        fd.clean(_arrow_frame(), engine=engine, fallback_policy="error", **KW)


@pytest.mark.parametrize("frame", [
    pytest.param(_frame(), id="categorical"),
    pytest.param(pd.DataFrame([[1, 2]], columns=["c", "c"]), id="duplicate-labels"),
    pytest.param(pd.DataFrame({"c": [1, "a"]}), id="mixed-values"),
    pytest.param(
        pd.DataFrame({"c": pd.period_range("2026-01", periods=2, freq="M")}), id="period"
    ),
])
def test_explicit_pandas_engine_does_not_need_an_ingest_fallback(frame):
    assert pandas_ingest_fallback_reason(frame, engine="pandas") is None


@pytest.mark.parametrize("ordered", [False, True])
def test_explicit_pandas_engine_accepts_arrow_dictionary_without_fallback(ordered):
    assert pandas_ingest_fallback_reason(_arrow_frame(ordered), engine="pandas") is None
