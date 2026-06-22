"""Bundle, entry point, adapter, and shared helpers."""

from __future__ import annotations

import json

import pytest

import freshdata as fd
from freshdata.compliance import (
    ComplianceConfig,
    generate_compliance_report,
)
from freshdata.compliance._adapter import build_context, classify_step
from freshdata.compliance._base import sha256_df

ALL_FRAMEWORKS = ["21cfr_11", "gdpr_30", "alcoa_plus", "sox_404", "hipaa_safe_harbor"]


def test_top_level_reexport():
    assert hasattr(fd, "generate_compliance_report")
    assert fd.ComplianceConfig is ComplianceConfig


def test_bundle_has_all_frameworks(sample_report):
    bundle = generate_compliance_report(sample_report, ALL_FRAMEWORKS)
    assert set(bundle.summary()) == set(ALL_FRAMEWORKS)
    assert set(bundle.frameworks) == set(ALL_FRAMEWORKS)
    assert all(key in bundle for key in ALL_FRAMEWORKS)
    assert len(bundle) == 5


def test_bundle_json_and_frame(sample_report):
    bundle = generate_compliance_report(sample_report, ALL_FRAMEWORKS)
    assert json.loads(bundle.to_json())  # valid JSON, no TypeError on uuids/None
    assert not bundle.to_frame().empty


def test_unknown_framework_raises(sample_report):
    with pytest.raises(ValueError, match="Unknown frameworks"):
        generate_compliance_report(sample_report, ["bogus"])


def test_each_report_json_and_frame(sample_report):
    bundle = generate_compliance_report(sample_report, ALL_FRAMEWORKS)
    for key in ALL_FRAMEWORKS:
        report = bundle[key]
        assert isinstance(json.loads(report.to_json()), dict)
        assert not report.to_frame().empty
        assert report.data["caveat"]


def test_enterprise_result_passed_as_report(sample_report, enterprise_stub):
    result = enterprise_stub(sample_report, overall=91.0, masked=["email"], clusters=["diagnosis"])
    bundle = generate_compliance_report(result, ALL_FRAMEWORKS)
    assert bundle["sox_404"].data["data_trust_score"] == 91.0
    coverage = bundle["hipaa_safe_harbor"].data["identifier_coverage"]
    assert coverage["email"]["status"] == "addressed"


def test_enterprise_result_kwarg_drives_trust_gate(sample_report, enterprise_stub):
    result = enterprise_stub(sample_report, overall=70.0)
    bundle = generate_compliance_report(sample_report, ["sox_404"], enterprise_result=result)
    assert bundle["sox_404"].data["data_trust_score"] == 70.0
    assert bundle["sox_404"].data["trust_score_gate_passed"] is False


def test_fuzzy_cluster_synthesized_from_enterprise(sample_report, enterprise_stub):
    result = enterprise_stub(sample_report, clusters=["diagnosis"])
    bundle = generate_compliance_report(sample_report, ["sox_404"], enterprise_result=result)
    log = bundle["sox_404"].data["transformation_log"]
    assert any(entry["action_type"] == "fuzzy_cluster" for entry in log)


def test_domain_trust_score_is_scaled(make_report):
    report = make_report({"step": "missing", "column": "a", "count": 0})
    report.domain = "healthcare"
    report.domain_trust_score = 0.9
    bundle = generate_compliance_report(report, ["sox_404"])
    assert bundle["sox_404"].data["data_trust_score"] == 90.0


def test_classify_step():
    assert classify_step("missing", 0) == ("flag", "unchanged")
    assert classify_step("missing", 3) == ("impute", "was_missing")
    assert classify_step("outliers", 1) == ("outlier_cap", "not_captured")
    assert classify_step("brand_new_step", 1)[0] == "brand_new_step"


def test_build_context_without_dataframe(sample_report):
    ctx = build_context(sample_report, ComplianceConfig())
    assert ctx.core_action_count == len(sample_report.actions)
    assert ctx.trust_score is None
    assert ctx.session_id.startswith("SESSION-")


def test_sha256_df_is_deterministic(sample_df):
    assert sha256_df(sample_df) == sha256_df(sample_df)
    assert len(sha256_df(sample_df)) == 64
