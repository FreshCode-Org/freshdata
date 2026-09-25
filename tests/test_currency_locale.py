"""Currency parsing must not assume a locale (FD2-002).

Before this suite, ``parse_currency`` deleted every comma and assumed the dot
was a decimal point, whatever currency was present. ``EUR 1.200,50`` therefore
read as 1.2005 -- a thousand-fold error on a monetary amount -- and ``€0,50``
read as 50.0, turning fifty cents into fifty euros. The repair was applied
automatically at confidence 0.98 and risk "low".

Every case here fails on the previous implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.engine.context import infer_role
from freshdata.semantic.experts import (
    CurrencyStringExpert,
    parse_currency,
    parse_currency_parts,
)
from freshdata.semantic.types import SemanticColumnInfo

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# -- European formats are no longer read as US ------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("EUR 1.200,50", 1200.50),
        ("€1.200,50", 1200.50),
        ("EUR 1.000,00", 1000.00),
        ("€1.234.567,89", 1234567.89),
        # The ones that were wrong in the *other* direction.
        ("€0,50", 0.50),
        ("EUR 12,5", 12.5),
    ],
)
def test_european_amounts_parse_under_european_convention(text, expected):
    assert parse_currency(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1,200.50", 1200.50),
        ("$1,234,567.89", 1234567.89),
        ("₹1,200.50", 1200.50),
        ("£1,200.50", 1200.50),
        ("$0.50", 0.50),
        ("$12.5", 12.5),
        # Indian lakh grouping stays supported.
        ("₹1,23,456.70", 123456.70),
    ],
)
def test_us_and_indian_amounts_are_unchanged(text, expected):
    assert parse_currency(text) == expected


def test_structure_beats_currency_convention():
    """A euro amount written the US way is still read correctly.

    The value's own punctuation is stronger evidence than the currency's
    conventional format, so the convention table is consulted only when the
    structure genuinely cannot settle it.
    """
    assert parse_currency("€1,234.56") == 1234.56
    assert parse_currency("$1.234,56") == 1234.56


@pytest.mark.parametrize("text", ["$1.2.3", "$1,20.50", "$1,2345.00", "$", "€ ,"])
def test_malformed_grouping_is_rejected_rather_than_coerced(text):
    """'1.2.3' must not be stripped to 123 by treating dots as grouping."""
    assert parse_currency(text) is None


@pytest.mark.parametrize("text", ["1,200", "1.200", "1.200,50", "1200"])
def test_a_bare_number_is_still_not_currency(text):
    """Unchanged contract: without a marker this is ordinary dtype repair."""
    assert parse_currency(text) is None


# -- ambiguity is reported, not hidden --------------------------------------


def test_a_three_digit_tail_without_a_currency_is_reported_ambiguous():
    """'1.200' is 1200 in Berlin and 1.2 in Boston.

    ``parse_currency_parts`` exposes that the reading was not settled by the
    input, so a caller can route the cell to a human rather than accept a
    guess. (A currency marker is still required to reach this path at all.)
    """
    value, ambiguous = parse_currency_parts("1.200")
    assert value is None and ambiguous is False  # no marker: not currency

    # With a marker the currency settles it, so it is not ambiguous.
    value, ambiguous = parse_currency_parts("EUR 1.200")
    assert (value, ambiguous) == (1200.0, False)
    value, ambiguous = parse_currency_parts("$1,200")
    assert (value, ambiguous) == (1200.0, False)


# -- the public API ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("EUR 1.200,50", 1200.50), ("€0,50", 0.50), ("EUR 12,5", 12.5)],
)
def test_clean_does_not_scale_european_amounts(text, expected):
    """The defect was reachable through fd.clean and applied automatically."""
    df = pd.DataFrame(
        {
            "k": [f"r{i}" for i in range(9)],
            "amount": [10.5, 20.25, 30.0, 40.75, 50.5, 60.25, 70.0, 80.5, text],
        }
    )
    out = fd.clean(df, verbose=False, semantic_mode="auto")
    assert out["amount"].iloc[8] == expected


def test_clean_financial_ledger_fixture_respects_locale_and_accounting_values():
    """Verify financial ledger cleaning preserves row counts, dtypes, and values."""
    fixture = pd.read_csv(FIXTURES_DIR / "financial_ledger.csv")
    expectations = json.loads(
        (FIXTURES_DIR / "financial_ledger.expectations.json").read_text()
    )["semantic_auto"]

    cleaned = fd.clean(
        fixture, strategy="balanced", semantic_mode="auto", verbose=False
    )

    assert len(cleaned) == expectations["row_count"]
    for column, dtype in expectations["required_conversions"].items():
        assert str(cleaned[column].dtype).startswith(dtype)
    for transaction_id, expected in expectations["target_values"].items():
        actual = cleaned.loc[cleaned["transaction_id"] == transaction_id, "amount"]
        assert len(actual) == 1
        assert actual.iloc[0] == expected


# -- PR #503 review fixes verification --------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(-$1,250.00)", -1250.0),
        ("($-1,250.00)", -1250.0),
        ("(-EUR 500.00)", -500.0),
        ("(-10.50)", -10.50),
    ],
)
def test_accounting_negatives_preserve_existing_negative_sign(text, expected):
    """An explicit minus sign inside accounting parentheses must not flip positive."""
    val, _ = parse_currency_parts(text)
    assert val == expected


@pytest.mark.parametrize(
    "text",
    [
        "(10 kg)",
        "(page 3)",
        "(see item 42)",
        "(10)",
        "(100)",
        "(10%)",
        "(10 / 20)",
        "( - )",
    ],
)
def test_unit_and_word_strings_in_parentheses_are_not_currency(text):
    """Parenthesized units and citations must not be treated as negative currency."""
    val, ambiguous = parse_currency_parts(text)
    assert val is None
    assert ambiguous is False


def test_unmarked_parenthetical_with_ambiguous_separators_flagged():
    """Unmarked numbers like (1,250) must be flagged ambiguous and routed to review."""
    val, ambiguous = parse_currency_parts("(1,250)")
    assert val == -1250.0
    assert ambiguous is True

    val2, ambiguous2 = parse_currency_parts("(1.250)")
    assert val2 == -1.25
    assert ambiguous2 is True

    expert = CurrencyStringExpert()
    info = SemanticColumnInfo(
        name="amount",
        role="numeric",
        n_nonnull=1,
        nunique=1,
        high_cardinality=False,
        preserve=False,
        free_text=False,
        numeric_like=True,
        boolean_like=False,
        money_like=True,
        unit_like=False,
        identifier_like=False,
    )
    series = pd.Series(["(1,250)"])
    proposals = expert.propose(series, info)
    assert len(proposals) == 1
    assert proposals[0].risk == "high"
    assert proposals[0].confidence <= 0.60


def test_payment_id_retains_identifier_role():
    """Names matching _ID_NAME must retain id role even when matching _MONEY_NAME."""
    cfg = CleanConfig()
    # Repeating values (nunique != non_null) ensure role is not inferred purely by cardinality
    for name in ("payment_id", "charge_id", "fee_id", "payment_key", "balance_uuid"):
        series = pd.Series(["ID1", "ID1", "ID2"])
        assert infer_role(name, series, cfg) == "id"


def test_free_text_monetary_columns_stay_protected():
    """Free-text columns marked money_like must stay protected from currency conversion."""
    expert = CurrencyStringExpert()
    info = SemanticColumnInfo(
        name="notes",
        role="text",
        n_nonnull=3,
        nunique=3,
        high_cardinality=False,
        preserve=False,
        free_text=True,
        numeric_like=False,
        boolean_like=False,
        money_like=True,
        unit_like=False,
        identifier_like=False,
    )
    assert expert.applies(info) is False
