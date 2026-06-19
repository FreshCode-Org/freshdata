"""Conformance tests for the retail (GS1) domain pack."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains import UnknownDomainError
from freshdata.domains.retail import RetailValidator


def _check_digit(body: str) -> str:
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return str((10 - total % 10) % 10)


GTIN14 = "0001234567890" + _check_digit("0001234567890")   # 00012345678905
GTIN13 = "400638133393" + _check_digit("400638133393")     # 4006381333931
GLN13 = "401234500000" + _check_digit("401234500000")      # 4012345000009


def _violated(rep, rule_id: str) -> bool:
    return any(f["rule_id"] == rule_id and f["status"] == "violated" for f in rep.domain_findings)


@pytest.fixture
def good_retail():
    return pd.DataFrame({
        "gtin": [GTIN14, GTIN13],
        "gln": [GLN13, GLN13],
        "product_description": ["Premium Widget", "Deluxe Gadget"],
        "brand_name": ["Acme", "Globex"],
        "net_content": [500.0, 1.5],
        "net_content_uom": ["GRM", "LTR"],
        "country_of_origin": ["US", "DEU"],
        "gpc_brick_code": ["10000045", "10000046"],
    })


def test_happy_path_passes(good_retail):
    out, rep = fd.clean(good_retail, domain="retail", return_report=True, verbose=False)
    assert rep.domain == "retail"
    assert rep.domain_trust_score >= 0.95
    assert not [f for f in rep.domain_findings
                if f["status"] == "violated" and f["severity"] == "error"]
    assert out.shape[0] == 2


@pytest.mark.parametrize("field", ["gtin"])
def test_required_field_missing(good_retail, field):
    validator = RetailValidator()
    actual = validator.detect_columns(good_retail).actual(field)
    report = validator.validate(good_retail.drop(columns=[actual]))
    assert field in report.mapping.unmapped_required
    assert any("MISSING_REQUIRED_FIELD" in r.message and r.violated for r in report.results)


def test_three_format_violations(good_retail):
    df = good_retail.copy()
    df.loc[0, "gtin"] = "123"               # GS1-002: wrong length
    df.loc[1, "gln"] = "999"                # GS1-004: not 13 digits / bad check
    df.loc[0, "gpc_brick_code"] = "12AB"    # GS1-008: not 8 digits
    _, rep = fd.clean(df, domain="retail", return_report=True, verbose=False)
    assert _violated(rep, "GS1-002")
    assert _violated(rep, "GS1-004")
    assert _violated(rep, "GS1-008")


def test_bad_gtin_check_digit(good_retail):
    df = good_retail.copy()
    df.loc[0, "gtin"] = "00012345678900"    # right length, wrong check digit
    _, rep = fd.clean(df, domain="retail", return_report=True, verbose=False)
    assert _violated(rep, "GS1-003")


def test_reference_violations_country_and_uom(good_retail):
    df = good_retail.copy()
    df.loc[0, "country_of_origin"] = "ZZ"   # not ISO 3166-1
    df.loc[1, "net_content_uom"] = "BOGUS"  # not UN/CEFACT
    _, rep = fd.clean(df, domain="retail", return_report=True, verbose=False)
    assert _violated(rep, "GS1-005")
    assert _violated(rep, "GS1-006")


def test_business_rule_content_uom_consistency(good_retail):
    df = good_retail.copy()
    df.loc[0, "net_content_uom"] = None     # content present but uom missing
    _, rep = fd.clean(df, domain="retail", return_report=True, verbose=False)
    assert _violated(rep, "GS1-009")


def test_repair_audit_strip_nondigits(good_retail):
    df = good_retail.copy()
    df.loc[0, "gtin"] = "0-0012345-67890-5"   # strippable to the valid GTIN14
    out, rep = fd.clean(df, domain="retail", return_report=True, verbose=False)
    applied = [r for r in rep.domain_repairs
               if r["rule_id"] == "GS1-002" and r["status"] == "applied"]
    assert applied and applied[0]["from"] == "0-0012345-67890-5" and applied[0]["to"] == GTIN14
    assert out.loc[0, "gtin"] == GTIN14


def test_repair_handles_duplicate_index_labels(good_retail):
    df = good_retail.copy()
    df.index = [7, 7]
    df.iloc[0, df.columns.get_loc("gtin")] = "0-0012345-67890-5"
    out, rep = fd.clean(df, domain="retail", return_report=True, verbose=False)
    assert out.index.tolist() == [7, 7]
    assert out.iloc[0]["gtin"] == GTIN14
    applied = [
        action
        for action in rep.domain_repairs
        if action["rule_id"] == "GS1-002" and action["status"] == "applied"
    ]
    assert applied and applied[0]["row"] == 7


def test_id_safety_null_gtin_never_filled(good_retail):
    df = good_retail.copy()
    df.loc[0, "gtin"] = None
    out, rep = fd.clean(df, domain="retail", return_report=True, verbose=False)
    assert pd.isna(out.loc[0, "gtin"])           # never imputed
    assert _violated(rep, "GS1-001")
    assert not [r for r in rep.domain_repairs
                if r["rule_id"] == "GS1-001" and r["status"] == "applied"]


def test_validation_never_mutates_input(good_retail):
    before = good_retail.copy()
    RetailValidator().validate(good_retail)
    pd.testing.assert_frame_equal(good_retail, before)


def test_unknown_domain_lists_available(good_retail):
    with pytest.raises(UnknownDomainError) as exc:
        fd.clean(good_retail, domain="unknown_xyz")
    assert {"finance", "retail", "transport"} <= set(exc.value.available)


def test_standalone_import():
    assert RetailValidator().domain_name == "retail"   # importable on its own (top of file)
