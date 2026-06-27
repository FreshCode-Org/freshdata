"""Tests for engine='auto' selection and backend construction."""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.execution import EngineConfig, EngineSelector


def test_auto_small_df_selects_pandas():
    df = pd.DataFrame({"a": range(1000)})
    assert EngineSelector.select(df, EngineConfig()) == "pandas"


def test_auto_medium_df_selects_polars(monkeypatch):
    cfg = EngineConfig(row_count_auto_threshold_polars=100, row_count_auto_threshold_duckdb=10_000)
    df = pd.DataFrame({"a": range(500)})
    assert EngineSelector.select(df, cfg) in ("polars", "duckdb")


def test_auto_large_df_selects_duckdb():
    cfg = EngineConfig(row_count_auto_threshold_polars=10, row_count_auto_threshold_duckdb=100)
    df = pd.DataFrame({"a": range(500)})
    assert EngineSelector.select(df, cfg) == "duckdb"


def test_auto_parquet_path_selects_duckdb():
    assert EngineSelector.select("/data/foo.parquet", EngineConfig()) == "duckdb"


def test_auto_csv_path_selects_duckdb():
    assert EngineSelector.select("/data/foo.csv", EngineConfig()) == "duckdb"


def test_auto_polars_frame_selects_polars():
    pl = pytest.importorskip("polars")
    lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    assert EngineSelector.select(lf, EngineConfig()) == "polars"


def test_get_engine_returns_named_backend():
    assert EngineSelector.get_engine("pandas", EngineConfig()).name == "pandas"


def test_invalid_engine_name_rejected():
    with pytest.raises(ValueError):
        EngineConfig(engine="dask")


def test_invalid_output_format_rejected():
    with pytest.raises(ValueError):
        EngineConfig(output_format="csv")
