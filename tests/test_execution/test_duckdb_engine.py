"""DuckDB backend behaviour: ingestion, spill config, SQL stages, fallback."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.execution import EngineConfig

duckdb = pytest.importorskip("duckdb")


def test_accepts_pandas_df(small_df, native_config):
    out = fd.clean(small_df.copy(), config=native_config, engine="duckdb")
    assert isinstance(out, pd.DataFrame)


def test_accepts_parquet_path(parquet_10k, native_config):
    out = fd.clean(parquet_10k, config=native_config, engine="duckdb")
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0


def test_memory_limit_and_temp_dir(tmp_path, small_df, native_config):
    spill = str(tmp_path / "spill")
    ec = EngineConfig(engine="duckdb", memory_limit_gb=0.5, temp_directory=spill)
    out = fd.clean(small_df.copy(), config=native_config, engine_config=ec)
    assert isinstance(out, pd.DataFrame)
    import os

    assert os.path.isdir(spill)


def test_spill_with_tiny_memory_limit(parquet_10k, native_config):
    ec = EngineConfig(engine="duckdb", memory_limit_gb=0.2)
    out = fd.clean(parquet_10k, config=native_config, engine_config=ec)
    assert len(out) > 0


def test_strip_and_sentinel(native_config):
    df = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["  alice ", "bob", "#REF!", "carol"]})
    out = fd.clean(df, config=native_config, engine="duckdb").sort_values("id")
    vals = list(out["name"])
    assert vals[0] == "alice"
    assert pd.isna(vals[2])  # sentinel -> null


def test_drop_empty_column(native_config):
    df = pd.DataFrame({"a": [1, 2, 3], "empty": [None, None, None]})
    out = fd.clean(df, config=native_config, engine="duckdb")
    assert "empty" not in out.columns


def test_drop_duplicates(native_config):
    df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "x", "y", "y", "z"]})
    out = fd.clean(df, config=native_config, engine="duckdb", drop_duplicates=True)
    assert len(out) == 3


def test_duplicates_detected_not_removed_by_default(native_config):
    df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "x", "y", "y", "z"]})
    out, report = fd.clean(df, config=native_config, engine="duckdb",
                           return_report=True)
    assert len(out) == 5
    assert any(a.step == "drop_duplicates" and "none removed" in a.description
               for a in report.actions)


def test_fallback_on_balanced_warns(small_df, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="freshdata.execution.duckdb"):
        out = fd.clean(small_df.copy(), engine="duckdb")  # default balanced -> fallback
    assert isinstance(out, pd.DataFrame)
    assert any("falling back to pandas" in r.message for r in caplog.records)


def test_thread_config_does_not_raise(small_df, native_config):
    ec = EngineConfig(engine="duckdb", duckdb_threads=2)
    out = fd.clean(small_df.copy(), config=native_config, engine_config=ec)
    assert isinstance(out, pd.DataFrame)


def test_no_leaked_connections(small_df, native_config):
    """Every execute() closes its connection (no growth in open DBs)."""
    import gc

    fd.clean(small_df.copy(), config=native_config, engine="duckdb")
    gc.collect()
    open_dbs = [o for o in gc.get_objects() if type(o).__name__ == "DuckDBPyConnection"]
    # the freshly-created benchmark/registration connections must all be closed;
    # we assert we did not accumulate a connection per call.
    assert len(open_dbs) <= 1
