"""Tests for the quality-ops bundling (``export_quality_ops``) and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
from freshdata import Action, CleanReport, QualityFinding
from freshdata.enterprise.cli import main as cli_main
from freshdata.enterprise.lineage import LineageTracker


def _findings() -> list[QualityFinding]:
    return [
        QualityFinding.create(severity="error", step="domain", column="email",
                              rule_name="email_not_null", message="nulls",
                              expected_condition="not_null"),
        QualityFinding.create(severity="warning", step="domain", column="status",
                              rule_name="status_set", message="bad",
                              extra={"value_set": ["open", "closed"]}),
    ]


def _report_with_violations() -> CleanReport:
    rep = CleanReport(domain="schema", domain_trust_score=0.5)
    rep.domain_findings = [
        {"rule_id": "email_not_null", "name": "email not null", "layer": "schema",
         "severity": "error", "fields": ("email",), "check": "not_null",
         "status": "violated", "n_violations": 1, "violation_rows": [0],
         "message": "null email", "repair": "none"},
    ]
    rep.actions = [Action(step="outliers", column="qty", description="capped",
                          count=2, risk="high")]
    return rep


def test_export_quality_ops_writes_all_artifacts(tmp_path):
    result = fd.export_quality_ops(
        _findings(),
        model_name="orders",
        suite_name="orders_suite",
        dbt_path=str(tmp_path / "schema.yml"),
        gx_path=str(tmp_path / "suite.json"),
        exception_table_path=str(tmp_path / "exc.csv"),
    )
    assert Path(result.dbt_path).exists()
    assert Path(result.gx_path).exists()
    assert Path(result.exception_table_path).exists()
    assert result.exception_table is not None
    # one run_id threaded through every finding
    run_ids = {f.lineage_run_id for f in result.findings}
    assert run_ids == {result.tracker.run_id}


def test_lineage_event_carries_artifact_facets(tmp_path):
    result = fd.export_quality_ops(
        _findings(),
        model_name="orders",
        suite_name="orders_suite",
        dbt_path=str(tmp_path / "schema.yml"),
        gx_path=str(tmp_path / "suite.json"),
        exception_table_path=str(tmp_path / "exc.csv"),
    )
    complete = result.lineage_event[-1]
    facets = complete["run"]["facets"]
    assert "dbt_tests_path" in facets
    assert "gx_suite_path" in facets
    assert "exception_table_path" in facets
    assert facets["dbt_tests_path"]["path"] == str((tmp_path / "schema.yml").resolve())


def test_uses_supplied_tracker_and_runs_only_requested_exporters(tmp_path):
    tracker = LineageTracker()
    result = fd.export_quality_ops(_findings(), dbt_path=str(tmp_path / "s.yml"),
                                   lineage=tracker)
    assert result.tracker is tracker
    assert result.gx_path is None
    assert result.exception_table_path is None
    facets = result.lineage_event[-1]["run"]["facets"]
    assert "dbt_tests_path" in facets
    assert "gx_suite_path" not in facets


def test_accepts_a_report_object(tmp_path):
    result = fd.export_quality_ops(_report_with_violations(),
                                   exception_table_path=str(tmp_path / "exc.csv"))
    assert len(result.findings) == 2  # domain violation + risky action
    assert result.to_dict()["exception_table_path"].endswith("exc.csv")


def test_cli_quality_ops(tmp_path):
    yaml = pytest.importorskip("yaml")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report_with_violations().to_dict()))

    dbt = tmp_path / "schema.yml"
    gx = tmp_path / "suite.json"
    exc = tmp_path / "exc.csv"
    lineage = tmp_path / "lineage.json"

    rc = cli_main([
        "quality-ops", str(report_path),
        "--dbt", str(dbt), "--gx", str(gx),
        "--exceptions", str(exc), "--lineage", str(lineage),
        "--model-name", "orders", "--quiet",
    ])
    assert rc == 0
    assert yaml.safe_load(dbt.read_text())["models"][0]["name"] == "orders"
    assert json.loads(gx.read_text())["expectation_suite_name"] == "orders_suite"
    assert len(pd.read_csv(exc)) >= 1
    facets = json.loads(lineage.read_text())[-1]["run"]["facets"]
    assert {"dbt_tests_path", "gx_suite_path", "exception_table_path"} <= set(facets)
