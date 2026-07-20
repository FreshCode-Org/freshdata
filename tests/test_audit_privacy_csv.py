"""Regression tests for the July 2026 production-audit defects P1-3/P1-4/P1-5.

Every test here FAILS on the pre-audit code:

* P1-3 — PII regex false positives: dates, Aadhaar-like groups, licence IDs
  and IBAN digit tails were reported as PHONE / CREDIT_CARD.
* P1-4 — ``anonymize()`` with no rules and no detection_config warned and
  returned the data unchanged (silent no-op on a privacy API).
* P1-5 — ``clean_csv`` wrote CSV output without formula sanitization by
  default (OWASP CSV injection).
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata._util import sanitize_csv_formulas
from freshdata.enterprise.config import MaskingRule, PIIDetectionConfig
from freshdata.enterprise.privacy import anonymize, detect_in_text, detect_pii

# --------------------------------------------------------------------------- #
# P1-3: false-positive matrix — none of these are PHONE or CREDIT_CARD
# --------------------------------------------------------------------------- #

FALSE_POSITIVES = [
    "1990-05-15",  # ISO date (confirmed audit repro: was PHONE)
    "12/31/1999",  # US date
    "31.12.1999",  # EU dotted date
    "2026-01-15 09:00-17:00",  # delivery window / timestamp range
    "550e8400-e29b-41d4",  # UUID fragment (confirmed audit repro)
    "550e8400-e29b-41d4-a716-446655440000",  # full UUID
    "D123-4567-8901",  # driver-licence ID (confirmed audit repro: was PHONE)
    "2345 6789 0123",  # Aadhaar-like 4-4-4 (confirmed audit repro: was PHONE)
    "2345-6789-0123",  # Aadhaar-like, dashed
    "1.2.3",  # semver
    "v1.2.3-build.456",  # semver with metadata
    "123456789",  # ordinary int
    "00012345678905",  # GTIN-ish bare digit run
    "1234.5678",  # ordinary float
    "3.14159265",  # ordinary float, long mantissa
    "DE89 3704 0044 0532 0130 00",  # valid IBAN (confirmed: was CREDIT_CARD+PHONE)
    "DE89370400440532013000",  # valid IBAN, compact
    "4111111111111112",  # 16 digits, Luhn-invalid
]


@pytest.mark.parametrize("value", FALSE_POSITIVES)
def test_no_phone_or_credit_card_false_positive(value):
    hits = {e.entity_type for e in detect_in_text(value)}
    assert "PHONE" not in hits, f"{value!r} misdetected as PHONE"
    assert "CREDIT_CARD" not in hits, f"{value!r} misdetected as CREDIT_CARD"


# --------------------------------------------------------------------------- #
# P1-3: true-positive matrix — tightening must not lose real detections
# --------------------------------------------------------------------------- #

REAL_PHONES = [
    "+1-202-555-0148",
    "(202) 555-0148",
    "202-555-0148",
    "202.555.0148",
    "+44 20 7946 0958",
    "+12025550148",
]

LUHN_VALID_CARDS = [
    "4111111111111111",
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "5500 0000 0000 0004",
    "378282246310005",  # Amex, 15 digits
]

VALID_IBANS = [
    "GB29NWBK60161331926819",
    "DE89370400440532013000",
    "DE89 3704 0044 0532 0130 00",
]


@pytest.mark.parametrize("value", REAL_PHONES)
def test_real_phone_formats_still_detected(value):
    assert "PHONE" in {e.entity_type for e in detect_in_text(value)}


def test_bare_digit_run_needs_phoneish_context():
    assert "PHONE" in {
        e.entity_type for e in detect_in_text("2025550148", column="phone_number")
    }
    assert not detect_in_text("2025550148", column="order_id")


@pytest.mark.parametrize("value", LUHN_VALID_CARDS)
def test_luhn_valid_cards_detected_with_checksum_confidence(value):
    hits = [e for e in detect_in_text(value) if e.entity_type == "CREDIT_CARD"]
    assert hits and hits[0].source == "checksum" and hits[0].score >= 0.95


@pytest.mark.parametrize("value", VALID_IBANS)
def test_valid_ibans_detected_as_iban_only(value):
    hits = {e.entity_type for e in detect_in_text(value)}
    assert "IBAN" in hits
    assert not hits & {"PHONE", "CREDIT_CARD"}


def test_detect_pii_dataframe_metadata_stays_honest():
    df = pd.DataFrame({"signup_date": ["1990-05-15"], "phone": ["+1-202-555-0148"]})
    report = detect_pii(df)
    assert report.metadata["ner"] is False
    hits = {(e.metadata["column"], e.entity_type) for e in report.entities}
    assert ("phone", "PHONE") in hits
    assert not any(col == "signup_date" and ent == "PHONE" for col, ent in hits)


def test_truthbench_style_canaries_stay_detected():
    # Synthetic canaries (reserved .invalid TLD) must survive the tightening.
    df = pd.DataFrame({"email": ["miyuki@example.invalid"]})
    assert "EMAIL" in detect_pii(df).entity_types


# --------------------------------------------------------------------------- #
# P1-4: anonymize() fails closed instead of silently no-opping
# --------------------------------------------------------------------------- #


def test_anonymize_without_config_raises():
    df = pd.DataFrame({"email": ["a@b.com"]})
    with pytest.raises(ValueError, match=r"rules.*detection_config"):
        anonymize(df)


def test_anonymize_with_rules_still_works():
    df = pd.DataFrame({"email": ["a@b.com"]})
    rule = MaskingRule(name="e", columns=("email",), strategy="redact")
    out, report = anonymize(df, rules=(rule,))
    assert list(out["email"]) == ["***"]
    assert report.cells_changed == 1


def test_anonymize_with_detection_config_still_works():
    df = pd.DataFrame({"notes": ["ssn 123-45-6789"]})
    out, report = anonymize(df, detection_config=PIIDetectionConfig())
    assert "123-45-6789" not in out["notes"].iloc[0]
    assert report.entities_found >= 1


# --------------------------------------------------------------------------- #
# P1-5: CSV formula injection — safe by default
# --------------------------------------------------------------------------- #

PAYLOADS = ["=1+1", "+cmd|' /C calc'!A0", "@SUM(A1)", "-2+3", "\t=1+2", " =1+1"]


def test_sanitizer_covers_leading_whitespace_triggers():
    df = pd.DataFrame({"note": [" =1+1", "\t =2", "  @SUM(A1)"]})
    out = sanitize_csv_formulas(df)
    assert list(out["note"]) == ["' =1+1", "'\t =2", "'  @SUM(A1)"]
    # non-mutation contract
    assert df["note"].iloc[0] == " =1+1"


def _write_payload_csv(tmp_path) -> str:
    src = tmp_path / "in.csv"
    pd.DataFrame(
        {"note": PAYLOADS, "amount": [-12.5, 3.0, 1.0, 2.0, 5.5, 9.9]}
    ).to_csv(src, index=False)
    return str(src)


def test_clean_csv_sanitizes_by_default(tmp_path):
    out = tmp_path / "out.csv"
    result = fd.clean_csv(_write_payload_csv(tmp_path), output_path=out)
    text = out.read_text()
    # default cleaning trims leading whitespace, so every payload reaches the
    # file in trigger position — and every one must be neutralized
    for quoted in ["'=1+1", "'+cmd|", "'@SUM(A1)", "'-2+3", "'=1+2"]:
        assert quoted in text, f"{quoted!r} missing from sanitized output"
    assert "\n=1+1" not in text and ",=1+1" not in text
    # numeric column round-trips untouched (negative numbers are not formulas)
    assert "-12.5" in text and "'-12.5" not in text
    assert result["amount"].dtype.kind == "f"
    # the returned frame is never sanitized, only the written artifact
    assert (result["note"] == "=1+1").any()


def test_clean_csv_opt_out_round_trips_exactly(tmp_path):
    out = tmp_path / "out.csv"
    fd.clean_csv(_write_payload_csv(tmp_path), output_path=out, sanitize_formulas=False)
    text = out.read_text()
    assert "'=" not in text and "'@" not in text and "'+cmd" not in text
    assert "=1+1" in text and "@SUM(A1)" in text and "-2+3" in text
