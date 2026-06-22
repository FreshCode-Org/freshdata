"""HIPAA Safe Harbor 18-identifier coverage generator."""

from __future__ import annotations

import pytest

from freshdata.compliance import (
    ComplianceConfig,
    ComplianceGapError,
    generate_compliance_report,
)


def _hipaa(report, **kw):
    return generate_compliance_report(report, ["hipaa_safe_harbor"], **kw)["hipaa_safe_harbor"]


def test_all_eighteen_identifiers_present(sample_report):
    hipaa = _hipaa(sample_report)
    assert len(hipaa.data["identifier_coverage"]) == 18
    assert {v["id"] for v in hipaa.data["identifier_coverage"].values()} == set(range(1, 19))


def test_email_addressed_when_masked(sample_report, sample_df):
    hipaa = _hipaa(
        sample_report,
        config=ComplianceConfig(masked_columns=["email", "patient_id"]),
        dataframe=sample_df,
    )
    assert hipaa.data["identifier_coverage"]["email"]["status"] == "addressed"
    assert "email" in hipaa.data["identifier_coverage"]["email"]["columns_masked"]


def test_email_detected_not_addressed(sample_report, sample_df):
    hipaa = _hipaa(sample_report, dataframe=sample_df)
    assert hipaa.data["identifier_coverage"]["email"]["status"] == "detected_not_addressed"
    assert "email" in hipaa.data["gaps"]


def test_coverage_pct_is_addressed_over_eighteen(make_report):
    report = make_report({"step": "missing", "column": "a", "count": 0})
    hipaa = _hipaa(report, config=ComplianceConfig(masked_columns=["email"]))
    assert hipaa.data["summary"]["addressed"] == 1
    assert hipaa.data["summary"]["coverage_pct"] == round(1 / 18 * 100, 4)


def test_fail_on_hipaa_gap_raises(sample_report, sample_df):
    with pytest.raises(ComplianceGapError):
        generate_compliance_report(
            sample_report,
            ["hipaa_safe_harbor"],
            config=ComplianceConfig(fail_on_hipaa_gap=True),
            dataframe=sample_df,
        )


def test_gap_sets_passed_false_and_errors(sample_report, sample_df):
    hipaa = _hipaa(sample_report, dataframe=sample_df)
    assert hipaa.passed is False
    assert hipaa.errors
    assert hipaa.data["caveat"]
