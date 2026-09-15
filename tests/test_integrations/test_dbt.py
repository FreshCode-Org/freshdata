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


def test_transform_rejects_audit_table_path_components(warehouse, tmp_path):
    with pytest.raises(ValueError, match="safe dbt model name"):
        FreshDataDbtTransform(
            model_name="../orders",
            conn_str=warehouse,
            output_dir=str(tmp_path),
            trust_score_threshold=0.0,
        ).run()
    assert not (tmp_path.parent / "orders_audit.json").exists()


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


def test_transform_rejects_invalid_policy():
    with pytest.raises(ValueError, match="on_low_score must be one of"):
        FreshDataDbtTransform(model_name="orders", on_low_score="erro")


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


def test_cli_stdout_is_exactly_one_json_document(warehouse, tmp_path, capsys):
    # sample_df has a duplicate row, so cleaning raises a warning; before the fix it
    # was printed to stdout ahead of the summary and broke `dbt-gate | jq`.
    manifest = str(_manifest(tmp_path))
    assert main(["--manifest", manifest, "--conn", warehouse, "--threshold", "0"]) == 0
    out = capsys.readouterr().out
    summary = json.loads(out)
    assert summary["models_processed"] == 1
    assert out.lstrip().startswith("{")


def test_cli_prints_during_gating_go_to_stderr(tmp_path, capsys, monkeypatch):
    import freshdata.integrations.dbt.cli as dbt_cli

    def noisy_gate(*args, **kwargs):
        print("chatty library output")
        return {"models": [], "skipped": [], "models_processed": 1, "failed_models": 0,
                "all_passed": True}

    monkeypatch.setattr(dbt_cli, "gate_manifest", noisy_gate)
    assert main(["--manifest", str(tmp_path / "manifest.json")]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["all_passed"] is True
    assert "chatty library output" in captured.err


def test_cli_missing_manifest_prints_one_line_error(capsys):
    code = main(["--manifest", "definitely_not_here.json"])
    assert code == 1
    err = capsys.readouterr().err
    assert "dbt-gate: error:" in err
    assert "definitely_not_here.json" in err
    assert "Traceback" not in err


# --------------------------------------------------------------------------- #
# #289: malformed manifests are one-line errors, not tracebacks                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "content", ["{not json", "[1, 2]", '{"metadata": {}, "results": []}', '{"nodes": []}']
)
def test_cli_malformed_manifest_prints_one_line_error(tmp_path, capsys, content):
    path = tmp_path / "manifest.json"
    path.write_text(content)
    assert main(["--manifest", str(path), "--fail"]) == 1
    captured = capsys.readouterr()
    assert "dbt-gate: error:" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_manifest_directory_prints_one_line_error(tmp_path, capsys):
    assert main(["--manifest", str(tmp_path), "--fail"]) == 1
    err = capsys.readouterr().err
    assert "dbt-gate: error:" in err
    assert "Traceback" not in err


# --------------------------------------------------------------------------- #
# #296: nothing gated must not look like a passing gate                        #
# --------------------------------------------------------------------------- #
def test_gate_manifest_rejects_non_manifest(tmp_path):
    path = tmp_path / "run_results.json"
    path.write_text(json.dumps({"metadata": {}, "results": []}))
    with pytest.raises(ValueError, match="not a dbt manifest"):
        gate_manifest(str(path))


def test_manifest_with_no_models_does_not_pass(tmp_path, capsys):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"nodes": {"test.proj.t": {"resource_type": "test"}}}))
    summary = gate_manifest(str(path))
    assert summary["models_processed"] == 0
    assert summary["failed_models"] == 0
    assert summary["all_passed"] is False

    assert main(["--manifest", str(path), "--fail"]) == 1
    err = capsys.readouterr().err
    assert "no models were gated" in err
    # Without --fail the run is still reported (exit 0), but not as a pass.
    assert main(["--manifest", str(path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["all_passed"] is False
    assert "no models were gated" in captured.err


# --------------------------------------------------------------------------- #
# #249: ephemeral / disabled models are skipped, not counted as failures       #
# --------------------------------------------------------------------------- #
def _manifest_with_unmaterialized(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "nodes": {
                    "model.proj.orders": {
                        "resource_type": "model",
                        "name": "orders",
                        "schema": None,
                        "config": {"materialized": "table"},
                    },
                    "model.proj.stg_x": {
                        "resource_type": "model",
                        "name": "stg_x",
                        "schema": None,
                        "config": {"materialized": "ephemeral"},
                    },
                    "model.proj.old": {
                        "resource_type": "model",
                        "name": "old",
                        "schema": None,
                        "config": {"materialized": "table", "enabled": False},
                    },
                }
            }
        )
    )
    return path


def test_manifest_skips_ephemeral_and_disabled_models(warehouse, tmp_path):
    summary = gate_manifest(
        str(_manifest_with_unmaterialized(tmp_path)),
        conn_str=warehouse,
        trust_score_threshold=0.0,
    )
    assert [m["model"] for m in summary["models"]] == ["orders"]
    assert summary["skipped"] == [
        {"model": "stg_x", "reason": "ephemeral"},
        {"model": "old", "reason": "disabled"},
    ]
    assert summary["models_processed"] == 1
    assert summary["failed_models"] == 0
    assert summary["all_passed"] is True


def test_cli_fail_passes_with_ephemeral_model(warehouse, tmp_path, capsys):
    manifest = str(_manifest_with_unmaterialized(tmp_path))
    rc = main(["--manifest", manifest, "--conn", warehouse, "--threshold", "0", "--fail"])
    assert rc == 0
    assert "stg_x" not in capsys.readouterr().err


def test_manifest_only_ephemeral_models_does_not_pass(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "nodes": {
                    "model.proj.stg_x": {
                        "resource_type": "model",
                        "name": "stg_x",
                        "config": {"materialized": "ephemeral"},
                    }
                }
            }
        )
    )
    summary = gate_manifest(str(path))
    assert summary["models_processed"] == 0
    assert summary["skipped"] == [{"model": "stg_x", "reason": "ephemeral"}]
    assert summary["all_passed"] is False
