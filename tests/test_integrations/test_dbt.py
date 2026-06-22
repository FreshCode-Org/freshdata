"""Tests for the dbt integration (real SQLAlchemy + in-memory sqlite)."""

from __future__ import annotations

import json

import pytest
import sqlalchemy as sa

from freshdata.integrations import TrustGateError
from freshdata.integrations.dbt import FreshDataDbtTransform, gate_manifest
from freshdata.integrations.dbt.cli import main


@pytest.fixture
def warehouse(tmp_path, sample_df):
    conn = f"sqlite:///{tmp_path / 'wh.db'}"
    engine = sa.create_engine(conn)
    with engine.begin() as connection:
        sample_df.to_sql("orders", connection, index=False)
    engine.dispose()
    return conn


def _manifest(tmp_path, *, name="orders"):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "nodes": {
                    f"model.proj.{name}": {
                        "resource_type": "model",
                        "name": name,
                        "schema": None,
                        "alias": name,
                    },
                    "test.proj.t": {"resource_type": "test", "name": "t"},
                }
            }
        )
    )
    return path


def test_transform_writes_audit(warehouse, tmp_path):
    result = FreshDataDbtTransform(
        model_name="orders",
        conn_str=warehouse,
        output_dir=str(tmp_path),
        trust_score_threshold=0.0,
    ).run()
    assert result.passed is True
    audit = tmp_path / "orders_audit.json"
    assert audit.exists()
    assert json.loads(audit.read_text())["trust_score"] == result.trust_score


def test_transform_fail_raises(warehouse):
    with pytest.raises(TrustGateError):
        FreshDataDbtTransform(
            model_name="orders",
            conn_str=warehouse,
            trust_score_threshold=999.0,
            fail_on_low_score=True,
        ).run()


def test_transform_no_connection_raises(monkeypatch):
    monkeypatch.delenv("FRESHDATA_WAREHOUSE_CONN", raising=False)
    with pytest.raises(ValueError, match="warehouse connection"):
        FreshDataDbtTransform(model_name="orders").run()


def test_transform_uses_env_conn(warehouse, monkeypatch):
    monkeypatch.setenv("FRESHDATA_WAREHOUSE_CONN", warehouse)
    result = FreshDataDbtTransform(model_name="orders", trust_score_threshold=0.0).run()
    assert result.passed is True


def test_manifest_summary_all_passed(warehouse, tmp_path):
    summary = gate_manifest(
        str(_manifest(tmp_path)), conn_str=warehouse, trust_score_threshold=0.0
    )
    assert summary["models_processed"] == 1  # the dbt test node is ignored
    assert summary["failed_models"] == 0
    assert summary["all_passed"] is True
    assert summary["models"][0]["model"] == "orders"


def test_manifest_missing_table_recorded(warehouse, tmp_path):
    summary = gate_manifest(
        str(_manifest(tmp_path, name="ghost")), conn_str=warehouse
    )
    assert summary["models_processed"] == 1
    assert summary["failed_models"] == 1
    assert "error" in summary["models"][0]
    assert summary["all_passed"] is False


def test_cli_pass_and_fail_exit_codes(warehouse, tmp_path, capsys):
    manifest = str(_manifest(tmp_path))
    assert main(["--manifest", manifest, "--conn", warehouse, "--threshold", "0"]) == 0
    assert "all_passed" in capsys.readouterr().out
    rc = main(["--manifest", manifest, "--conn", warehouse, "--threshold", "999", "--fail"])
    assert rc == 1
