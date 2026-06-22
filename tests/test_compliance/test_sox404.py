"""SOX-404 transformation control evidence generator."""

from __future__ import annotations

from freshdata.compliance import ComplianceConfig, generate_compliance_report


def _sox(report, **kw):
    return generate_compliance_report(report, ["sox_404"], **kw)["sox_404"]


def test_trust_gate_fails_below_threshold(make_report):
    report = make_report({"step": "missing", "column": "a", "count": 0})
    sox = _sox(report, config=ComplianceConfig(trust_score=50.0))
    assert sox.data["trust_score_gate_passed"] is False
    assert sox.passed is False


def test_high_risk_in_exception_summary(make_report):
    report = make_report({"step": "outliers", "column": "amount", "count": 3, "risk": "high"})
    sox = _sox(report, config=ComplianceConfig(trust_score=95.0))
    summary = sox.data["exception_summary"]
    assert summary["high_risk_count"] == 1
    assert "amount" in summary["high_risk_columns"]
    assert summary["requires_manual_review"] is True
    assert sox.passed is False
    assert sox.errors


def test_attestation_caveat_present(make_report):
    report = make_report({"step": "missing", "column": "a", "count": 0})
    sox = _sox(report, config=ComplianceConfig(trust_score=95.0))
    assert sox.data["attestation"]["caveat"]
    assert sox.data["caveat"]


def test_gate_unevaluated_without_score(make_report):
    report = make_report({"step": "missing", "column": "a", "count": 0})
    sox = _sox(report)  # no trust score available anywhere
    assert sox.data["trust_score_gate_passed"] is None
    assert sox.passed is True  # no high-risk action, gate not failed
    assert sox.warnings


def test_preventive_vs_detective_control_type(make_report):
    report = make_report(
        {"step": "strip_whitespace", "column": "name", "count": 2},  # Preventive
        {"step": "outliers", "column": "amount", "count": 1},  # Detective
    )
    sox = _sox(report, config=ComplianceConfig(trust_score=90.0))
    control = {e["column"]: e["control_type"] for e in sox.data["transformation_log"]}
    assert control["name"] == "Preventive"
    assert control["amount"] == "Detective"
