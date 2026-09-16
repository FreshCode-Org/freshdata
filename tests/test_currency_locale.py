"""Currency parsing must not assume a locale (FD2-002).

Before this suite, ``parse_currency`` deleted every comma and assumed the dot
was a decimal point, whatever currency was present. ``EUR 1.200,50`` therefore
read as 1.2005 -- a thousand-fold error on a monetary amount -- and ``€0,50``
read as 50.0, turning fifty cents into fifty euros. The repair was applied
automatically at confidence 0.98 and risk "low".

Every case here fails on the previous implementation.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.semantic.experts import parse_currency, parse_currency_parts

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
