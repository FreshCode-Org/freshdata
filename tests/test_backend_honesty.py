"""Honesty guards for the out-of-core backend paths.

These assert the scalability behaviour freshdata advertises is *true*:

* ``output_format="duckdb"`` / ``"polars-lazy"`` return native, un-materialized
  handles (not a pandas DataFrame), and the report says it didn't materialize.
* The default path still materializes to pandas (backward compatible).
* Streaming Polars dedup is streaming-safe (does not force ``maintain_order``)
  and discloses the order trade-off.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata import CleanConfig, EngineConfig

# Conservative strategy keeps the native backends off the pandas decision engine,
# so the result can stay an un-materialized handle.
_CONSERVATIVE = CleanConfig(strategy="conservative", fix_dtypes=False)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"a": ["x", "x", "y"], "b": [" p", " p", "q "]})


def test_duckdb_native_handle_is_not_materialized() -> None:
    duckdb = pytest.importorskip("duckdb")
    out, report = fd.clean(
        _frame(), config=_CONSERVATIVE, engine="duckdb",
        output_format="duckdb", return_report=True,
    )
    assert isinstance(out, duckdb.DuckDBPyRelation)
    assert report.materialized is False
    assert report.to_dict().get("materialized") is False
    # The caller pulls rows explicitly.
    assert out.fetchdf().shape[1] == 2


def test_polars_lazy_handle_is_not_collected() -> None:
    pl = pytest.importorskip("polars")
    out, report = fd.clean(
        _frame(), config=_CONSERVATIVE, engine="polars",
        output_format="polars-lazy", return_report=True,
    )
    assert isinstance(out, pl.LazyFrame)
    assert report.materialized is False
    assert out.collect().shape[1] == 2


def test_default_path_still_materializes_pandas() -> None:
    pytest.importorskip("duckdb")
    out, report = fd.clean(
        _frame(), config=_CONSERVATIVE, engine="duckdb", return_report=True
    )
    assert isinstance(out, pd.DataFrame)
    assert report.materialized is True


def test_native_summary_discloses_non_materialization() -> None:
    pytest.importorskip("duckdb")
    _, report = fd.clean(
        _frame(), config=_CONSERVATIVE, engine="duckdb",
        output_format="duckdb", return_report=True,
    )
    text = report.summary()
    assert "not materialized" in text


def test_streaming_dedup_is_order_safe_and_disclosed() -> None:
    pytest.importorskip("polars")
    df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "x", "y", "y", "z"]})
    ec = EngineConfig(engine="polars", streaming=True)
    _, report = fd.clean(
        df, config=_CONSERVATIVE, engine="polars",
        engine_config=ec, return_report=True,
    )
    diffs = [d for d in report.backend_differences
             if d.get("step") == "drop_duplicates"]
    assert diffs, "streaming dedup should disclose the order trade-off"
    assert "order" in diffs[0]["detail"].lower()


def test_order_preserving_dedup_opt_out_no_disclosure() -> None:
    pytest.importorskip("polars")
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    ec = EngineConfig(engine="polars", streaming=True, streaming_dedup=False)
    _, report = fd.clean(
        df, config=_CONSERVATIVE, engine="polars",
        engine_config=ec, return_report=True,
    )
    diffs = [d for d in report.backend_differences
             if d.get("step") == "drop_duplicates"]
    assert not diffs
