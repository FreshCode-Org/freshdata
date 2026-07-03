"""Semantic-type inference: detectors, hint precedence, veto, infer_roles."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.semantic.semantic_types import (
    MIN_DISTINCT_SUPPORT,
    SEMANTIC_TYPES,
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
    values = [f"C-{i}-x{i}" for i in range(8)]  # matches no detector at 60%
    result = infer_semantic_type("email_addr", _series(values))
    assert result.semantic_type == "email"
    assert result.confidence == pytest.approx(0.5)


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


def test_infer_roles_respects_explicit_hint():
    df = pd.DataFrame({"code": EMAILS[:6]})
    roles = fd.infer_roles(
        df, semantic_context={"columns": {"code": {"semantic_type": "category_code"}}}
    )
    assert roles.set_index("column").loc["code", "semantic_type"] == "category_code"
    assert roles.set_index("column").loc["code", "semantic_type_confidence"] == 1.0
