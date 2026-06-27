"""Tests for the normalized quality-finding model and ``report.to_findings()``."""

from __future__ import annotations

import pandas as pd

import freshdata as fd
from freshdata import Action, CleanReport, QualityFinding
from freshdata.findings import (
    REDACT_TOKEN,
    classify_finding,
    findings_from_dict,
    make_finding_id,
    normalize_severity,
)


def test_normalize_severity_maps_vocabularies():
    assert normalize_severity("high") == "error"
    assert normalize_severity("failed") == "error"
    assert normalize_severity("medium") == "warning"
    assert normalize_severity("warn") == "warning"
    assert normalize_severity("low") == "info"
    assert normalize_severity("passed") == "info"
    # unknown -> warning, and casing/whitespace are ignored
    assert normalize_severity("  WeIrD ") == "warning"


def test_finding_id_is_deterministic():
    a = make_finding_id("clean", "col", "rule", 1, "msg")
    b = make_finding_id("clean", "col", "rule", 1, "msg")
    c = make_finding_id("clean", "col", "rule", 2, "msg")
    assert a == b
    assert a != c
    assert len(a) == 12


def test_create_derives_id_and_normalizes_severity():
    f = QualityFinding.create(severity="high", step="domain", rule_name="r", message="m",
                              column="email", row_index=3)
    assert f.severity == "error"
    assert f.finding_id == make_finding_id("domain", "email", "r", 3, "m")


def test_observed_value_redacted_by_default():
    f = QualityFinding.create(severity="error", step="privacy", rule_name="SSN",
                              message="m", observed_value="123-45-6789", sensitive=True)
    assert f.display_observed() == REDACT_TOKEN
    assert f.to_dict()["observed_value"] == REDACT_TOKEN
    assert f.to_dict(include_pii=True)["observed_value"] == "123-45-6789"


def test_classify_finding():
    assert classify_finding(QualityFinding.create(
        severity="error", step="d", rule_name="x", message="m",
        expected_condition="not_null")) == "not_null"
    assert classify_finding(QualityFinding.create(
        severity="error", step="d", rule_name="duplicate_rows", message="m")) == "unique"
    assert classify_finding(QualityFinding.create(
        severity="error", step="d", rule_name="x", message="m",
        extra={"value_set": ["a"]})) == "accepted_values"
    assert classify_finding(QualityFinding.create(
        severity="error", step="d", rule_name="x", message="m",
        extra={"min_value": 0})) == "between"
    assert classify_finding(QualityFinding.create(
        severity="error", step="d", rule_name="x", message="m",
        extra={"to_model": "ref('m')", "to_field": "id"})) == "relationships"
    assert classify_finding(QualityFinding.create(
        severity="error", step="d", rule_name="opaque", message="m")) == "custom"


def _clean_report_with_violations() -> CleanReport:
    rep = CleanReport(domain="schema", domain_trust_score=0.5)
    rep.domain_findings = [
        {"rule_id": "r1", "name": "email not null", "layer": "schema", "severity": "error",
         "fields": ("email",), "check": "not_null", "status": "violated",
         "n_violations": 2, "violation_rows": [0, 2], "message": "nulls", "repair": "impute"},
        {"rule_id": "r2", "name": "ok", "layer": "schema", "severity": "warning",
         "fields": ("x",), "check": "unique", "status": "passed",
         "n_violations": 0, "violation_rows": [], "message": "", "repair": "none"},
    ]
    rep.domain_repairs = [
        {"rule_id": "r1", "strategy": "impute", "column": "email", "row": 0,
         "from": None, "to": "x@y.com", "status": "applied"},
    ]
    rep.actions = [Action(step="outliers", column="qty", description="capped 3",
                          count=3, risk="high")]
    return rep


def test_clean_report_to_findings():
    findings = _clean_report_with_violations().to_findings(lineage_run_id="RUN")
    # violated domain rule + risky action; the passed rule is skipped.
    assert len(findings) == 2
    by_rule = {f.rule_name: f for f in findings}
    dom = by_rule["r1"]
    assert dom.severity == "error"
    assert dom.column == "email"
    assert dom.row_selector == "rows: [0, 2]"
    assert dom.action_taken == "impute (1 row(s))"
    assert dom.lineage_run_id == "RUN"
    assert classify_finding(dom) == "not_null"
    act = by_rule["outliers"]
    assert act.severity == "error"  # high -> error
    assert act.column == "qty"


def test_findings_from_dict_matches_to_findings():
    rep = _clean_report_with_violations()
    direct = rep.to_findings(lineage_run_id="RUN")
    from_dict = findings_from_dict(rep.to_dict(), lineage_run_id="RUN")
    assert [f.finding_id for f in direct] == [f.finding_id for f in from_dict]


def test_empty_report_has_no_findings():
    assert CleanReport().to_findings() == []
    # A real clean of already-clean data also yields none.
    _, rep = fd.clean(pd.DataFrame({"a": [1, 2, 3]}), return_report=True)
    assert rep.to_findings() == []
