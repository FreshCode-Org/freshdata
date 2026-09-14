"""GDPR Article 30 + Article 17 generator."""

from __future__ import annotations

from freshdata.compliance import ComplianceConfig, generate_compliance_report


def _gdpr(report, **kw):
    return generate_compliance_report(report, ["gdpr_30"], **kw)["gdpr_30"]


def test_personal_data_includes_masked_email(sample_report):
    gdpr = _gdpr(sample_report, config=ComplianceConfig(masked_columns=["email"]))
    assert "email" in gdpr.data["article_30"]["personal_data_categories"]


def test_erasure_log_one_per_drop_or_mask(make_report):
    report = make_report(
        {"step": "drop_empty_columns", "column": "junk", "count": 1},  # drop_col
        {"step": "drop_duplicates", "column": None, "count": 2},  # drop_row
        {"step": "missing", "column": "age", "count": 0},  # flag — not an erasure
    )
    gdpr = _gdpr(report, config=ComplianceConfig(masked_columns=["email"]))
    log = gdpr.data["article_17"]["erasure_log"]
    assert len(log) == 3  # 2 drops + 1 mask
    assert {e["erasure_type"] for e in log} == {"DROP_COLUMN", "DROP_ROW", "PII_ANONYMISE"}


def test_grounds_for_erasure_non_empty(make_report):
    report = make_report({"step": "drop_empty_columns", "column": "junk", "count": 1})
    gdpr = _gdpr(report)
    assert gdpr.data["erasure_log"]
    assert all(e["grounds_for_erasure"] for e in gdpr.data["erasure_log"])


def test_passed_true_with_warning_when_no_pii(make_report):
    report = make_report({"step": "missing", "column": "age", "count": 0})
    gdpr = _gdpr(report, config=ComplianceConfig(data_subject_categories=["patients"]))
    assert gdpr.passed is True
    assert gdpr.data["article_30"]["personal_data_categories"] == []
    assert gdpr.warnings


def test_article_30_record_fields(make_report):
    report = make_report({"step": "missing", "column": "age", "count": 0})
    gdpr = _gdpr(
        report,
        config=ComplianceConfig(controller_name="Acme Corp", legal_basis="consent"),
    )
    article_30 = gdpr.data["article_30"]
    assert article_30["controller_name"] == "Acme Corp"
    assert article_30["legal_basis"] == "consent"
    assert article_30["third_country_transfers"] is False
    assert article_30["automated_decision_making"] is True
    assert gdpr.data["caveat"]


# --- #287: security measures reflect evidence, not a constant list ------------


def test_security_measures_omit_masking_when_nothing_masked(sample_report, sample_df):
    gdpr = _gdpr(sample_report, dataframe=sample_df)
    measures = gdpr.data["article_30"]["security_measures"]
    assert not any("mask" in m.lower() for m in measures)
    assert not any("sha-256" in m.lower() or "salt" in m.lower() for m in measures)
    assert "Audit trail generation" in measures
    # No enterprise trust score was supplied, so none is claimed.
    assert not any("trust score" in m.lower() for m in measures)


def test_security_measures_list_masking_from_config(make_report):
    report = make_report({"step": "missing", "column": "age", "count": 0})
    gdpr = _gdpr(report, config=ComplianceConfig(masked_columns=["email", "ssn"]))
    measures = gdpr.data["article_30"]["security_measures"]
    assert measures[0] == "PII masking/anonymisation applied to 2 column(s)"


def test_security_measures_name_recorded_strategies(sample_report, enterprise_stub):
    result = enterprise_stub(sample_report, overall=90.0, masked=["email"])
    gdpr = _gdpr(result)
    measures = gdpr.data["article_30"]["security_measures"]
    assert measures[0] == (
        "PII masking/anonymisation applied to 1 column(s) (strategies: sha256+salt)"
    )
    assert "Data Trust Score computed" in measures
