"""HIPAA Safe Harbor 18-identifier coverage generator."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
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


# --- #245: coverage without dataframe= -----------------------------------------


def test_clean_records_input_columns_after_rename():
    raw = pd.DataFrame({"Patient Email": ["a@b.com", "c@d.com"], "v": [1, 2]})
    _, report = fd.clean(raw, return_report=True, verbose=False)
    assert report.input_columns == ["patient_email", "v"]
    assert "input_columns" not in report.to_dict()  # stable audit payload unchanged


def test_untouched_identifier_columns_fail_without_dataframe():
    raw = pd.DataFrame(
        {"ssn": ["123-45-6789", "987-65-4321"], "patient_email": ["a@b.com", "c@d.com"]}
    )
    _, report = fd.clean(raw, return_report=True, verbose=False)
    no_df = _hipaa(report)
    with_df = _hipaa(report, dataframe=raw)
    assert no_df.passed is False
    assert no_df.data["gaps"] == with_df.data["gaps"] == ["email", "ssn"]
    assert no_df.data["coverage_verifiable"] is True
    assert no_df.warnings == []


def test_masked_identifiers_pass_without_dataframe():
    raw = pd.DataFrame({"email": ["a@b.com", "c@d.com"], "v": [1, 2]})
    _, report = fd.clean(raw, return_report=True, verbose=False)
    hipaa = _hipaa(report, config=ComplianceConfig(masked_columns=["email"]))
    assert hipaa.passed is True
    assert hipaa.data["coverage_verifiable"] is True


def test_no_column_evidence_is_unverifiable_and_not_passed(make_report):
    report = make_report({"step": "missing", "column": "age", "count": 0})
    hipaa = _hipaa(report)
    assert hipaa.data["gaps"] == []
    assert hipaa.data["coverage_verifiable"] is False
    assert hipaa.passed is False
    assert any("not verifiable without dataframe=" in w for w in hipaa.warnings)
    assert hipaa.errors == []


def test_dataframe_makes_synthetic_report_verifiable(make_report):
    report = make_report({"step": "missing", "column": "age", "count": 0})
    hipaa = _hipaa(report, dataframe=pd.DataFrame({"age": [1, 2]}))
    assert hipaa.data["coverage_verifiable"] is True
    assert hipaa.passed is True
    assert hipaa.warnings == []


# --- #283: identifier hints match name tokens, not arbitrary substrings -------


@pytest.mark.parametrize(
    "column",
    [
        "description",
        "shipping_cost",
        "last_updated",
        "business_unit",
        "hotel",
        "cancelled",
        "province",
        "fluid",
        "surface",
        "backlinks",
        "filename",
    ],
)
def test_short_hints_do_not_match_inside_words(make_report, column):
    hipaa = _hipaa(make_report(), dataframe=pd.DataFrame({column: [1, 2]}))
    found = {k: v["columns_found"] for k, v in hipaa.data["identifier_coverage"].items()}
    assert not any(found.values()), found
    assert hipaa.passed is True


def test_issue_283_repro_passes(make_report):
    df = pd.DataFrame(
        {
            "description": ["ok", "fine"],
            "last_updated": ["y", "n"],
            "shipping_cost": [1, 2],
            "business_unit": ["a", "b"],
        }
    )
    _, report = fd.clean(df, return_report=True, verbose=False)
    hipaa = _hipaa(report, dataframe=df)
    assert hipaa.passed is True
    assert hipaa.data["gaps"] == []


@pytest.mark.parametrize(
    ("column", "identifier"),
    [
        ("ip", "ip_addresses"),
        ("client_ip", "ip_addresses"),
        ("IPAddress", "ip_addresses"),
        ("IPv4Address", "ip_addresses"),
        ("date_of_birth", "dates"),
        ("DateOfBirth", "dates"),
        ("dateofbirth", "dates"),
        ("visit_date", "dates"),
        ("patientemail", "email"),
        ("PatientEmail", "email"),
        ("SSN", "ssn"),
        ("home_tel", "phone"),
        ("zip5", "geographic"),
        ("zipcode", "geographic"),
        ("surname", "names"),
        ("firstName", "names"),
        ("vehicle_vin", "vehicle_identifiers"),
        ("record_uid", "other_unique"),
    ],
)
def test_identifier_hints_still_detect_real_identifiers(make_report, column, identifier):
    hipaa = _hipaa(make_report(), dataframe=pd.DataFrame({column: [1, 2]}))
    assert column in hipaa.data["identifier_coverage"][identifier]["columns_found"]
    assert hipaa.passed is False


def test_long_hint_does_not_start_mid_token(make_report):
    hipaa = _hipaa(make_report(), dataframe=pd.DataFrame({"ship_address": [1, 2]}))
    coverage = hipaa.data["identifier_coverage"]
    assert coverage["ip_addresses"]["columns_found"] == []
    assert coverage["geographic"]["columns_found"] == ["ship_address"]


def test_apply_plan_records_input_columns():
    df = pd.DataFrame(
        {
            "ssn": ["123-45-6789", "987-65-4321", "111-22-3333", "444-55-6666"],
            "email_addr": ["a@@b.com", "x @ y.com", "ok@ok.com", "junk"],
        }
    )
    plan = fd.suggest_plan(df, semantic_mode="auto", verbose=False).repair_plan
    _, report = fd.apply_plan(df, plan)
    assert report.input_columns == ["ssn", "email_addr"]
    assert _hipaa(report).data["coverage_verifiable"] is True
