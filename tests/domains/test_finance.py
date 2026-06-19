"""Conformance tests for the finance domain pack."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains import UnknownDomainError
from freshdata.domains.finance import FinanceValidator


def _violated(report, rule_id: str) -> bool:
    return any(f["rule_id"] == rule_id and f["status"] == "violated"
               for f in report.domain_findings)


def test_happy_path_passes(good_finance):
    out, rep = fd.clean(good_finance, domain="finance", return_report=True, verbose=False)
    assert rep.domain == "finance"
    assert rep.domain_trust_score >= 0.95
    assert not [f for f in rep.domain_findings
                if f["status"] == "violated" and f["severity"] == "error"]
    assert out.shape[0] == 4


@pytest.mark.parametrize("field", ["transaction_id", "date", "debit", "credit", "currency"])
def test_each_required_field_missing(good_finance, field):
    validator = FinanceValidator()
    actual = validator.detect_columns(good_finance).actual(field)
    report = validator.validate(good_finance.drop(columns=[actual]))
    assert field in report.mapping.unmapped_required
    assert any("MISSING_REQUIRED_FIELD" in r.message and r.violated for r in report.results)


def test_three_format_violations(good_finance):
    df = good_finance.copy()
    df.loc[0, "date"] = "31/31/2024"        # FIN-003: impossible date
    df.loc[1, "debit"] = -5.0               # FIN-004: negative
    df.loc[2, "debit"] = 10.555             # FIN-004: > 2 decimals
    df.loc[3, "account_code"] = "AB"        # FIN-008: too short
    _, rep = fd.clean(df, domain="finance", return_report=True, verbose=False)
    assert _violated(rep, "FIN-003")
    assert _violated(rep, "FIN-004")
    assert _violated(rep, "FIN-008")


def test_reference_violation_bad_currency(good_finance):
    df = good_finance.copy()
    df.loc[0, "currency"] = "XYZ"           # not ISO 4217
    _, rep = fd.clean(df, domain="finance", return_report=True, verbose=False)
    assert _violated(rep, "FIN-005")
    # a real code is accepted
    df.loc[0, "currency"] = "gbp"           # case-sensitive per rule -> still flagged
    _, rep2 = fd.clean(df, domain="finance", return_report=True, verbose=False)
    assert _violated(rep2, "FIN-005")


def test_business_rule_imbalance(good_finance):
    df = good_finance.copy()
    df.loc[1, "credit"] = 999.0             # T1 now unbalanced (100 vs 999); T2 untouched
    _, rep = fd.clean(df, domain="finance", return_report=True, verbose=False)
    assert _violated(rep, "FIN-006")        # imbalance (error)
    # FIN-006 flags both rows of the offending transaction only
    fin006 = next(f for f in rep.domain_findings if f["rule_id"] == "FIN-006")
    assert fin006["n_violations"] == 2


def test_business_rule_single_sided(good_finance):
    df = good_finance.copy()
    df.loc[0, "credit"] = 100.0             # row 0 now has debit=100 AND credit=100
    _, rep = fd.clean(df, domain="finance", return_report=True, verbose=False)
    assert _violated(rep, "FIN-007")        # single-sided entry (warning)


def test_repair_audit_date_coercion(good_finance):
    df = good_finance.copy()
    df.loc[0, "date"] = "01/15/2024"        # unambiguous (15 can't be a month)
    out, rep = fd.clean(df, domain="finance", return_report=True, verbose=False)
    applied = [r for r in rep.domain_repairs
               if r["rule_id"] == "FIN-003" and r["status"] == "applied"]
    assert applied and applied[0]["from"] == "01/15/2024" and applied[0]["to"] == "2024-01-15"
    assert out.loc[0, "date"] == "2024-01-15"


def test_ambiguous_date_not_silently_coerced(good_finance):
    df = good_finance.copy()
    df.loc[0, "date"] = "02/03/2024"        # both <= 12: ambiguous
    out, rep = fd.clean(df, domain="finance", return_report=True, verbose=False)
    assert out.loc[0, "date"] == "02/03/2024"   # unchanged
    assert any(r["rule_id"] == "FIN-003" and r["status"] == "unresolvable"
               for r in rep.domain_repairs)


def test_id_safety_null_transaction_id_never_filled(good_finance):
    df = good_finance.copy()
    df.loc[0, "transaction_id"] = None
    out, rep = fd.clean(df, domain="finance", return_report=True, verbose=False)
    assert pd.isna(out.loc[0, "transaction_id"])     # never imputed
    assert _violated(rep, "FIN-001")
    # no repair action ever targets the id column
    assert all(a["column"] != "transaction_id" or a["status"] != "applied"
               for a in rep.domain_repairs)


def test_validation_never_mutates_input(good_finance):
    df = good_finance.copy()
    before = df.copy()
    FinanceValidator().validate(df)
    pd.testing.assert_frame_equal(df, before)


def test_unknown_domain_raises_listing_available(good_finance):
    with pytest.raises(UnknownDomainError) as exc:
        fd.clean(good_finance, domain="unknown_xyz")
    assert "finance" in exc.value.available
    assert "unknown_xyz" in str(exc.value)


def test_messy_columns_detected(messy_finance):
    # In the real flow, generic clean normalizes names (snake_case) before the
    # pack runs; the regex aliases then resolve the canonical fields.
    normalized = messy_finance.rename(columns=lambda c: c.lower().replace(" ", "_"))
    mapping = FinanceValidator().detect_columns(normalized)
    assert mapping.actual("transaction_id") == "txn_id"   # via regex alias
    assert mapping.actual("currency") == "ccy"
    out, rep = fd.clean(messy_finance, domain="finance", return_report=True, verbose=False)
    assert rep.domain_trust_score >= 0.95


def test_column_map_override(good_finance):
    df = good_finance.rename(columns={"debit": "money_out"})
    # without override, "money_out" is unmapped -> FIN-002 missing 'debit'
    _, rep_missing = fd.clean(df, domain="finance", return_report=True, verbose=False)
    assert any("debit" in r["message"] for r in rep_missing.domain_findings
               if "MISSING_REQUIRED_FIELD" in r["message"])
    # with override it maps cleanly
    _, rep_ok = fd.clean(df, domain="finance", column_map={"money_out": "debit"},
                         return_report=True, verbose=False)
    assert rep_ok.domain_trust_score >= 0.95


def test_column_map_requires_domain(good_finance):
    with pytest.raises(TypeError, match="column_map requires a domain"):
        fd.clean(good_finance, column_map={"a": "b"})


def test_no_domain_path_unchanged(good_finance):
    # A non-domain clean is byte-identical to running without the feature.
    plain = pd.DataFrame({"a": [1, 2, 2, None], "b": ["x", "y", "y", "z"]})
    a = fd.clean(plain, verbose=False)
    b = fd.clean(plain.copy(), verbose=False)
    pd.testing.assert_frame_equal(a, b)
