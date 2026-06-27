"""The ``freshdata clean --engine ...`` scalable-backend CLI path."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from freshdata.enterprise.cli import main

pytest.importorskip("pyarrow")


@pytest.fixture
def parquet_in(tmp_path) -> str:
    df = pd.DataFrame(
        {
            "Customer ID": ["X10", "Y20", "X10", "Z30"],
            "amount": [1.0, 2.0, 1.0, 9999.0],
            " Note ": [" a ", "N/A", "a", "b"],
        }
    )
    path = str(tmp_path / "in.parquet")
    df.to_parquet(path)
    return path


@pytest.mark.parametrize("engine", ["polars", "duckdb", "auto"])
def test_clean_cli_engine_writes_output(parquet_in, tmp_path, engine, capsys):
    pytest.importorskip(engine if engine != "auto" else "duckdb")
    out_path = str(tmp_path / f"out_{engine}.parquet")
    rc = main([
        "clean", parquet_in, "-o", out_path,
        "--engine", engine, "--strategy", "conservative",
    ])
    assert rc == 0
    res = pd.read_parquet(out_path)
    assert "customer_id" in res.columns
    assert "freshdata:" in capsys.readouterr().out.lower() or len(res) >= 1


def test_clean_cli_duckdb_memory_limit(parquet_in, tmp_path):
    pytest.importorskip("duckdb")
    out_path = str(tmp_path / "out.parquet")
    rc = main([
        "clean", parquet_in, "-o", out_path,
        "--engine", "duckdb", "--memory-limit-gb", "4",
        "--strategy", "conservative", "--quiet",
    ])
    assert rc == 0
    assert pd.read_parquet(out_path).shape[0] >= 1


def test_clean_cli_engine_report(parquet_in, tmp_path):
    pytest.importorskip("polars")
    report_path = str(tmp_path / "report.json")
    rc = main([
        "clean", parquet_in, "-o", str(tmp_path / "out.parquet"),
        "--engine", "polars", "--strategy", "conservative",
        "--report", report_path, "--quiet",
    ])
    assert rc == 0
    with open(report_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "actions" in payload
