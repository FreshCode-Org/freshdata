"""21 CFR §11.10(e) audit-trail generator."""

from __future__ import annotations

from freshdata.compliance import ComplianceConfig, generate_compliance_report


def _cfr(report, **kw):
    return generate_compliance_report(report, ["21cfr_11"], **kw)["21cfr_11"]


def test_one_audit_entry_per_action(sample_report):
    cfr = _cfr(sample_report)
    assert len(cfr.data["audit_entries"]) == len(sample_report.actions)


def test_session_id_is_constant_across_entries(sample_report):
    cfr = _cfr(sample_report)
    session_ids = {entry["session_id"] for entry in cfr.data["audit_entries"]}
    assert len(session_ids) == 1
    assert cfr.data["session_header"]["session_id"] == session_ids.pop()


def test_drop_row_maps_to_delete(make_report):
    report = make_report({"step": "drop_duplicates", "column": None, "count": 2})
    cfr = _cfr(report)
    assert cfr.data["audit_entries"][0]["record_action_type"] == "DELETE"


def test_modify_without_preimage_fails(make_report):
    report = make_report({"step": "outliers", "column": "amount", "count": 3, "risk": "medium"})
    cfr = _cfr(report)
    assert cfr.passed is False
    assert cfr.warnings
    assert cfr.data["audit_entries"][0]["original_value_class"] == "not_captured"


def test_imputes_drops_and_repairs_pass(make_report):
    report = make_report(
        {"step": "missing", "column": "age", "count": 2},  # imputation (count > 0)
        {"step": "drop_empty_rows", "column": None, "count": 1},
        {"step": "fix_dtypes", "column": "age", "count": 4},
        {"step": "strip_whitespace", "column": "name", "count": 3},
    )
    cfr = _cfr(report)
    assert cfr.passed is True
    assert not cfr.warnings


def test_header_caveat_and_operator(sample_report):
    cfr = _cfr(
        sample_report,
        config=ComplianceConfig(retention_days=3650, operator_id="svc-1", trust_score=88.0),
    )
    header = cfr.data["session_header"]
    assert header["retention_days"] == 3650
    assert header["data_trust_score"] == 88.0
    assert cfr.data["caveat"]
    assert all(e["operator_id"] == "svc-1" for e in cfr.data["audit_entries"])
    assert all(e["non_obscuring_guarantee"] is True for e in cfr.data["audit_entries"])


def test_strict_normalization_flags_whitespace(make_report):
    report = make_report({"step": "strip_whitespace", "column": "name", "count": 2})
    # Lenient (default): a whitespace trim is a non-obscuring normalisation.
    assert _cfr(report).passed is True
    # Strict: the rewrite is treated as obscuring (no pre-image retained).
    strict = _cfr(report, config=ComplianceConfig(strict_cfr_normalization=True))
    assert strict.passed is False
    assert strict.warnings
    entry = strict.data["audit_entries"][0]
    assert entry["original_value_class"] == "not_captured"
    assert entry["non_obscuring_guarantee"] is False
