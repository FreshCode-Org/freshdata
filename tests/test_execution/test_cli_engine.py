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


# --------------------------------------------------------------------------- #
# --config on native engines: validated first, clean applied, enterprise gated #
# --------------------------------------------------------------------------- #
_NATIVE_ENGINES = ["polars", "duckdb", "spark", "freshcore", "auto"]


def _run_with_config(parquet_in, tmp_path, engine, payload, *extra):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(payload))
    out_path = tmp_path / f"out_{engine}.parquet"
    rc = main([
        "clean", parquet_in, "-o", str(out_path), "--engine", engine,
        "--config", str(cfg), "--quiet", *extra,
    ])
    return rc, out_path


@pytest.mark.parametrize("engine", _NATIVE_ENGINES)
@pytest.mark.parametrize(
    "payload, needle",
    [
        ({"clean": {"stratgy": "conservative"}}, "'stratgy' (did you mean 'strategy'?)"),
        ({"enterprise": {"enable_maskin": True}}, "'enable_maskin' (did you mean"),
        ({"enterprize": {}}, "'enterprize' (did you mean 'enterprise'?)"),
    ],
)
def test_clean_cli_engine_config_typos_error_on_every_engine(
    parquet_in, tmp_path, capsys, engine, payload, needle
):
    rc, out_path = _run_with_config(parquet_in, tmp_path, engine, payload)
    assert rc == 1
    captured = capsys.readouterr()
    assert needle in captured.err
    assert "Traceback" not in captured.err
    assert not out_path.exists()


@pytest.mark.parametrize("engine", _NATIVE_ENGINES)
def test_clean_cli_engine_rejects_enterprise_features(parquet_in, tmp_path, capsys, engine):
    payload = {
        "enterprise": {
            "masking": [{"name": "m", "columns": ["Customer ID"]}],
            "fail_under_trust": 80,
        }
    }
    rc, out_path = _run_with_config(parquet_in, tmp_path, engine, payload)
    assert rc == 1
    err = capsys.readouterr().err
    assert f"--engine {engine} cannot run: masking, fail_under_trust" in err
    assert "--engine pandas" in err
    assert not out_path.exists()


@pytest.mark.parametrize("engine", _NATIVE_ENGINES)
def test_clean_cli_engine_rejects_context_in_clean_section(parquet_in, tmp_path, capsys, engine):
    rc, out_path = _run_with_config(
        parquet_in, tmp_path, engine, {"clean": {"context": "never drop rows"}}
    )
    assert rc == 1
    assert "only supported on the pandas engine" in capsys.readouterr().err
    assert not out_path.exists()


@pytest.mark.parametrize("engine", ["polars", "duckdb", "spark"])
def test_clean_cli_engine_applies_clean_section(parquet_in, tmp_path, engine):
    # Before the fix the engine path never read --config, so columns were renamed.
    pytest.importorskip("pyspark" if engine == "spark" else engine)
    rc, out_path = _run_with_config(
        parquet_in, tmp_path, engine, {"clean": {"column_names": False}}
    )
    assert rc == 0
    assert "Customer ID" in pd.read_parquet(out_path).columns


@pytest.mark.parametrize("engine", ["polars", "duckdb"])
def test_clean_cli_engine_accepts_default_enterprise_section(parquet_in, tmp_path, engine):
    pytest.importorskip(engine)
    payload = {"enterprise": {"enable_masking": True, "enable_contracts": False, "drift": None}}
    rc, out_path = _run_with_config(parquet_in, tmp_path, engine, payload)
    assert rc == 0
    assert "customer_id" in pd.read_parquet(out_path).columns
