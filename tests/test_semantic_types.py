"""Semantic-type inference: detectors, hint precedence, veto, infer_roles."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.semantic.semantic_types import (
    MIN_DISTINCT_SUPPORT,
    SEMANTIC_TYPES,
    _type_shares,
    infer_semantic_type,
)


def _series(values: list[str]) -> pd.Series:
    return pd.Series(values)


EMAILS = [f"user{i}@example{i % 3}.com" for i in range(8)]
PHONES = ["+91 98765 43210", "09123 456789", "98765-43210", "+91 90000 11111",
          "091234 56789", "9876543210", "+919812345678", "98111 22333"]


def test_taxonomy_is_complete():
    for required in (
        "email", "phone", "url", "country", "currency_amount", "quantity_with_unit",
        "person_name", "address", "city", "postal_code", "national_id",
        "category_code", "free_text", "boolean_like", "date_like", "identifier",
        "unknown",
    ):
        assert required in SEMANTIC_TYPES


@pytest.mark.parametrize(
    "values, expected",
    [
        (EMAILS, "email"),
        (PHONES, "phone"),
        ([f"https://shop{i}.example.com/p" for i in range(6)], "url"),
        (["India", "China", "France", "Brazil", "Japan", "Kenya"], "country"),
        (["$1,200.50", "$900.00", "$12.00", "$5.25", "$77.10"], "currency_amount"),
        (["5kg", "12kg", "3.5kg", "700g", "1.2kg"], "quantity_with_unit"),
        (["2024-01-15", "2024-02-01", "2023-12-31", "2024-03-09", "2024-04-02"], "date_like"),
        (["110001", "400076", "560034", "700019", "122002"], "postal_code"),
        (["12 MG Road", "7 Brigade Street", "221B Baker Street", "14 Ring Road", "3 Park Ave"],
         "address"),
        (["Asha Rao", "Ravi Kumar", "Neha Sharma", "Kiran Nair", "Vikram Singh"], "person_name"),
        (["SKU-101", "SKU-102", "AB-9", "XY-77", "QQ-3"], "category_code"),
    ],
)
def test_content_detectors(values, expected):
    result = infer_semantic_type("col", _series(values))
    assert result.semantic_type == expected
    assert result.confidence >= 0.6
    assert result.evidence


def test_explicit_hint_wins_over_content():
    result = infer_semantic_type("col", _series(EMAILS), hint="category_code")
    assert result.semantic_type == "category_code"
    assert result.confidence == 1.0
    assert result.evidence[0].kind == "context_hint"


def test_content_detector_vetoes_embedding_vote():
    result = infer_semantic_type(
        "col", _series(EMAILS), embedding_vote=("person_name", 0.99)
    )
    assert result.semantic_type == "email"  # content wins
    assert any(e.kind == "conflict" and "overruled" in e.detail for e in result.evidence)


def test_embedding_vote_fills_gap_but_is_capped():
    values = [f"opaque-{i}-token" for i in range(8)]
    result = infer_semantic_type("colx", _series(values), embedding_vote=("city", 0.95))
    assert result.semantic_type == "city"
    assert result.confidence <= 0.7  # model-only labels never certify
    assert any(e.kind == "embedding" for e in result.evidence)


def test_low_sample_returns_unknown_low_confidence():
    result = infer_semantic_type("email", _series(EMAILS[: MIN_DISTINCT_SUPPORT - 1]))
    assert result.semantic_type == "unknown"
    assert result.confidence <= 0.3


def test_role_signals_map_to_types():
    ids = pd.Series([f"C{i:05d}" for i in range(30)])
    assert infer_semantic_type("cust_id", ids, role="id").semantic_type == "identifier"
    prose = pd.Series(["a long note " * 5 + str(i) for i in range(6)])
    assert infer_semantic_type("notes", prose, role="text").semantic_type == "free_text"


def test_name_hint_used_when_content_is_inconclusive():
    # 3/8 emails: below the 60% detection bar but not contradicted, so the
    # column name may still suggest the type at hint-level confidence.
    values = EMAILS[:3] + [f"C-{i}-x{i}" for i in range(5)]
    result = infer_semantic_type("email_addr", _series(values))
    assert result.semantic_type == "email"
    assert result.confidence == pytest.approx(0.5)


def test_strict_name_hint_vetoed_when_content_contradicts():
    # A column *named* email whose sampled values match email 0% must not be
    # certified as email by the name alone (regression: name-hint false positive).
    result = infer_semantic_type("email", _series([str(i) for i in range(1, 9)]))
    assert result.semantic_type != "email"
    assert any(e.kind == "conflict" for e in result.evidence)


def test_boolean_detected_below_distinct_support():
    # Booleans inherently have ~2 distinct values; they must not be swallowed
    # by the distinct-support gate (regression: boolean_like was unreachable).
    two_token = infer_semantic_type("subscribed", _series(["yes", "no", "yes", "no"]))
    assert two_token.semantic_type == "boolean_like"
    assert two_token.confidence >= 0.8

    bool_dtype = infer_semantic_type("active", pd.Series([True, False] * 5))
    assert bool_dtype.semantic_type == "boolean_like"


def test_bare_binary_01_not_forced_boolean():
    # {"0", "1"} is just as likely a numeric indicator: stays gated.
    result = infer_semantic_type("flag", _series(["0", "1", "0", "1"]))
    assert result.semantic_type == "unknown"


def test_infer_roles_gains_additive_columns():
    df = pd.DataFrame(
        {
            "email_addr": EMAILS[:6],
            "mob_no": PHONES[:6],
            "monthly_revenue": ["$100.00", "$200.00", "$150.00", "$90.00", "$75.50", "$60.25"],
            "age": [25, 32, 41, 28, 35, 30],
        }
    )
    roles = fd.infer_roles(df)
    for col in ("column", "role", "missing_pct", "cardinality", "skew",
                "domain_sensitive", "primary_missing_model"):
        assert col in roles.columns  # original surface intact
    for col in ("semantic_type", "semantic_type_confidence", "semantic_type_evidence"):
        assert col in roles.columns  # additive Phase-3 columns
    by_col = roles.set_index("column")
    assert by_col.loc["email_addr", "semantic_type"] == "email"
    assert by_col.loc["mob_no", "semantic_type"] == "phone"
    assert by_col.loc["monthly_revenue", "semantic_type"] == "currency_amount"
    assert (roles["semantic_type_confidence"] <= 1.0).all()


PUNYCODE_EMAILS = [f"user{i}@example.xn--p1ai" for i in range(40)]
INTL_PHONES = [f"+49 (0) 30 - 1234 - {i:04d}" for i in range(40)]


def test_punycode_tld_emails_are_email():
    result = infer_semantic_type("contact", _series(PUNYCODE_EMAILS))
    assert result.semantic_type == "email"


def test_long_formatted_international_phones_are_phone():
    # 24 characters: longer than the old 17-character cap, 15 digits.
    assert len(INTL_PHONES[0]) > 17
    result = infer_semantic_type("contact", _series(INTL_PHONES))
    assert result.semantic_type == "phone"


def test_phone_digit_count_capped_at_e164_maximum():
    fifteen = [f"+1 234 567 890 1{i:04d}" for i in range(10)]
    assert infer_semantic_type("contact", _series(fifteen)).semantic_type == "phone"
    for digits in (16, 17, 20):
        values = [str(10 ** (digits - 1) + i) for i in range(40)]
        result = infer_semantic_type("contact", _series(values))
        assert result.semantic_type != "phone", digits


def test_ascii_emails_and_us_phones_still_detected():
    emails = [f"first.last+{i}@mail.example.org" for i in range(10)]
    assert infer_semantic_type("col", _series(emails)).semantic_type == "email"
    us_phones = [f"(555) 010-{i:04d}" for i in range(10)] + [
        f"+1 415 555 {i:04d}" for i in range(10)
    ]
    assert infer_semantic_type("col", _series(us_phones)).semantic_type == "phone"


def test_numeric_id_columns_not_classified_as_phone():
    short_ids = [str(1000 + i) for i in range(40)]  # too few digits
    long_ids = [str(123456789012345678 + i) for i in range(40)]  # over E.164 max
    for values in (short_ids, long_ids):
        assert infer_semantic_type("order", _series(values)).semantic_type != "phone"
    # Integer-typed ID columns: the same guards apply after str() conversion.
    int_ids = pd.Series([123456789012345678 + i for i in range(40)], dtype="int64")
    assert infer_semantic_type("order", int_ids).semantic_type != "phone"
    # A role=id column short-circuits before any content detector.
    phone_shaped = _series([str(9876543210 + i) for i in range(40)])
    assert infer_semantic_type("cust", phone_shaped, role="id").semantic_type == "identifier"


def test_infer_roles_uses_validator_patterns():
    # Repeated values keep the role categorical/numeric, so infer_roles reaches
    # the content detectors instead of short-circuiting on role id/text.
    def repeat(values: list) -> list:
        return (values * 10)[:40]

    df = pd.DataFrame(
        {
            "order_ref": repeat([str(12345678901234567 + i) for i in range(6)]),
            "order_num": repeat([12345678901234567 + i for i in range(6)]),
            "contact": repeat(INTL_PHONES[:6]),
            "mail": repeat(PUNYCODE_EMAILS[:6]),
        }
    )
    by_col = fd.infer_roles(df).set_index("column")
    assert by_col.loc["order_ref", "semantic_type"] != "phone"
    assert by_col.loc["order_num", "semantic_type"] != "phone"
    assert by_col.loc["contact", "semantic_type"] == "phone"
    assert by_col.loc["mail", "semantic_type"] == "email"


PARITY_SAMPLES = [
    # emails
    "user@example.xn--p1ai", "a+tag@example.com", "b@example.com", "x@y.io",
    "no-tld@host", "two@@example.com", "sp ace@example.com", "a@b.c", "a@b.xn--",
    # phones
    "+49 (0) 30 12345678", "+44 (0) 20 7946 0958", "+1 (555) 010-9999",
    "555-0100 123", "+1 234 567 890 12345", "+49 (0) 30 - 1234 - 0001",
    "9876543210", "12345", "123-45", "+1234567890123456", "+1 (555) 010-9999 ext 4",
    "(((((((((((())))))", "+" + "1 " * 16, "(" * 31, "12345678901234567",
    # neither
    "hello world", "2024-01-15",
]


@pytest.mark.parametrize("value", PARITY_SAMPLES)
def test_email_phone_verdicts_match_fieldcheck(value: str) -> None:
    # Semantic inference must accept exactly what validate_fields accepts.
    shares = _type_shares([value])
    for semantic_type in ("email", "phone"):
        report = fd.validate_fields(pd.DataFrame({"c": [value]}), {"c": semantic_type})
        validator_accepts = report.issues == []
        assert (shares[semantic_type] == 1.0) is validator_accepts, (semantic_type, value)


def test_infer_roles_respects_explicit_hint():
    df = pd.DataFrame({"code": EMAILS[:6]})
    roles = fd.infer_roles(
        df, semantic_context={"columns": {"code": {"semantic_type": "category_code"}}}
    )
    assert roles.set_index("column").loc["code", "semantic_type"] == "category_code"
    assert roles.set_index("column").loc["code", "semantic_type_confidence"] == 1.0
