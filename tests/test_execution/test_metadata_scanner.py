"""Tests for the cheap metadata scanners."""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.execution import MetadataScanner


def test_from_pandas_basic():
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": ["x", "y", "z", "w"]})
    meta = {m.name: m for m in MetadataScanner.from_pandas(df)}
    assert meta["a"].row_count == 4
    assert meta["a"].non_null_count == 3
    assert abs(meta["a"].null_ratio - 0.25) < 1e-9
    assert meta["a"].is_numeric
    assert meta["b"].is_string
    assert meta["b"].is_empty is False


def test_from_pandas_empty_column():
    df = pd.DataFrame({"a": [1, 2], "empty": [None, None]})
    meta = {m.name: m for m in MetadataScanner.from_pandas(df)}
    assert meta["empty"].is_empty
    assert meta["a"].is_empty is False


def test_from_polars_lazy_matches_pandas():
    pl = pytest.importorskip("polars")
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": ["x", "y", None, "w"]})
    lf = pl.from_pandas(df).lazy()
    meta = {m.name: m for m in MetadataScanner.from_polars_lazy(lf)}
    assert meta["a"].row_count == 4
    assert meta["a"].non_null_count == 3
    assert meta["b"].non_null_count == 3
    assert meta["a"].is_numeric
    assert meta["b"].is_string


def test_from_parquet_path(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"a": [1, 2, 3, None], "b": ["x", None, "z", "w"]})
    path = str(tmp_path / "m.parquet")
    df.to_parquet(path)
    meta = {m.name: m for m in MetadataScanner.from_parquet_path(path)}
    assert meta["a"].row_count == 4
    assert abs(meta["a"].null_ratio - 0.25) < 0.01


def test_from_duckdb(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    conn = duckdb.connect()
    try:
        conn.register("t", df)
        meta = {m.name: m for m in MetadataScanner.from_duckdb(conn, "t")}
        assert meta["a"].row_count == 3
        assert meta["b"].is_empty
    finally:
        conn.close()


def test_from_duckdb_exact_counts_with_non_finite_floats():
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    try:
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES "
            "(1.0::DOUBLE, 1), ('inf'::DOUBLE, NULL), ('nan'::DOUBLE, 3), (NULL, 4)) v(x, y)"
        )
        meta = {m.name: m for m in MetadataScanner.from_duckdb(conn, "t")}
        assert meta["x"].row_count == 4
        assert meta["x"].non_null_count == 2  # NaN is missing (as in pandas); inf is a value
        assert meta["x"].native_dtype == "DOUBLE"
        assert meta["y"].non_null_count == 3
    finally:
        conn.close()
