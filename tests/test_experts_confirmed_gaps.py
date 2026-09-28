"""Behaviour of ``semantic/experts.py`` that no test pinned.

Each test here closes a mutant that survived the *full* suite (not just a
targeted subset) on ``28ce1c3``. Every one was also checked to be observable:
the mutated code produces different output on the input used below, so these
are real gaps, not equivalent mutants.
"""
from __future__ import annotations

import itertools
import math

import pandas as pd
import pytest

import freshdata as fd
from freshdata.semantic import experts
from freshdata.semantic.experts import (
    _resolve_date,
    looks_like_identifier_value,
    parse_currency_parts,
    parse_number_words,
)

# ── accounting negatives need BOTH parentheses (bool#10) ────────────────────


@pytest.mark.parametrize("value", ["(1,200", "(1.50", "1,200)", "($5", "(€1.200,50"])
def test_an_unbalanced_parenthesis_is_not_an_accounting_negative(value):
    """``(1,200)`` is -1200; ``(1,200`` is malformed and must not parse.

    Treating a lone opening parenthesis as the accounting-negative marker
    strips the first *and last* characters, so ``"(1,200"`` became ``"1,20"``
    and was read as -1.2, unambiguous -- a malformed value silently turned
    into a confident wrong number.
    """
    assert parse_currency_parts(value) == (None, False)


def test_a_balanced_accounting_negative_still_parses():
    """Control for the test above."""
    assert parse_currency_parts("(1,200.50)") == (-1200.5, False)


# ── no ISO churn when the column cannot convert (cmp#33) ────────────────────


def test_valid_iso_dates_are_left_alone_beside_an_unresolvable_value():
    """Canonical ISO strings are re-encoded only alongside a *low-risk* repair.

    Here the only non-ISO value is day/month-ambiguous (high risk), so the
    column cannot convert as a whole. Rewriting the valid ISO dates to
    timestamps anyway would be pure representation churn, and would leave an
    object column mixing ``Timestamp``s with the one string still under
    review. ``fix_dtypes=False`` keeps the values as strings until the
    semantic stage; with it on, the dtype step converts the column first and
    this path is never reached.
    """
    values = ["2021-03-15", "2021-04-20", "2021-06-30", "05/01/2021"] * 5
    result = fd.clean(pd.DataFrame({"d": values}), semantic_mode="auto",
                      fix_dtypes=False, verbose=False)

    assert result.data["d"].tolist() == values
    semantic = [a for a in result.report().actions if a.step == "semantic"]
    assert [a.status for a in semantic] == ["suggested"]
    assert "05/01/2021" in semantic[0].description


# ── a bare "hundred" means one hundred (const_num#34) ──────────────────────


@pytest.mark.parametrize("text,expected", [
    ("hundred", 100), ("hundred and five", 105), ("hundred thousand", 100_000),
])
def test_a_bare_hundred_counts_as_one_hundred(text, expected):
    """``hundred`` with no count before it is one hundred, like ``thousand``.

    Existing tests only used ``"one hundred"``, where the explicit count hides
    the default: ``(current or 1) * 100`` could default to any number and no
    test noticed.
    """
    assert parse_number_words(text) == expected


# ── lakh grouping checks every inner group (const_num#46) ──────────────────


@pytest.mark.parametrize("value", ["₹12,3,456", "$1,2,345", "₹1,2,34,567"])
def test_a_malformed_inner_group_is_rejected(value):
    """Indian grouping is 2-digit groups then a final 3-digit group.

    A group of one digit anywhere before the last is malformed and must not
    parse. Only the first inner group was being checked by any test, so a
    grouping check that skipped the one before last went unnoticed and
    ``"$1,2,345"`` would have read as 12345 -- unambiguous.
    """
    assert parse_currency_parts(value) == (None, False)


@pytest.mark.parametrize("value,expected", [
    ("₹1,23,456", 123456.0), ("₹12,34,567.50", 1234567.5),
])
def test_well_formed_lakh_grouping_still_parses(value, expected):
    """Control for the test above."""
    assert parse_currency_parts(value) == (expected, False)


# ── a grouping separator cannot lead (const_num#38) ────────────────────────


@pytest.mark.parametrize("value", ["$,123", "€,123.45", "£,000"])
def test_a_leading_grouping_separator_is_rejected(value):
    """``,123`` has a thousands separator with no thousands before it.

    The first group is required to be non-empty; checking the second instead
    would have accepted these as 123 and 123.45.
    """
    assert parse_currency_parts(value) == (None, False)


def test_a_leading_decimal_point_is_still_a_decimal():
    """Control: ``.500`` is a decimal, not a grouping, and parses."""
    assert parse_currency_parts("$.500") == (0.5, False)


# ── an accounting negative of exactly one, and of zero (const_num#55) ──────


@pytest.mark.parametrize("value", ["($1.00)", "(1.00)", "(€1,00)"])
def test_an_accounting_negative_of_one_is_minus_one(value):
    """The zero special case must not swallow a debit of exactly one.

    ``-abs(value) if value != 0 else 0.0`` exists only to keep zero from
    becoming ``-0.0``. Keyed on the wrong value it would turn a one-unit
    debit into 0.0 -- a dollar silently erased -- and no test had a debit of
    exactly one.
    """
    assert parse_currency_parts(value)[0] == -1.0


def test_an_accounting_negative_zero_is_positive_zero():
    """``($0.00)`` is zero, not negative zero (which prints as ``-0.0``)."""
    value, _ = parse_currency_parts("($0.00)")
    assert value == 0.0
    assert math.copysign(1.0, value) == 1.0


# ── date resolutions: a valid confidence, and the partial-date guess ───────


def _resolve(raw: str, dayfirst: bool | None = None, reference_date: str | None = None):
    return _resolve_date(raw, dayfirst=dayfirst, reference_date=reference_date)


@pytest.mark.parametrize("raw,kwargs", [
    ("2021-03-15", {}), ("2021/03/15", {}), ("March 5, 2021", {}), ("5 March 2021", {}),
    ("13/01/2021", {}), ("05/01/2021", {}), ("05/01/2021", {"dayfirst": True}),
    ("2025-03", {}), ("yesterday", {}), ("yesterday", {"reference_date": "2021-03-15"}),
    ("yesterday", {"reference_date": "not a date"}),
])
def test_every_date_resolution_has_a_valid_confidence(raw, kwargs):
    """A confidence is a score in [0, 1] at the point it is produced (const_num#91).

    The report clamps downstream, which hid an ISO-date confidence of 1.03
    from every existing test; the resolver's own output must be valid.
    """
    res = _resolve(raw, **kwargs)
    assert res is not None
    assert 0.0 <= res.confidence <= 1.0


def test_a_partial_date_offers_the_first_of_that_month_for_review():
    """``2025-03`` has no day: never applied, but the reviewer sees 1 March 2025.

    Reading the month group as the year (const_num#92) turned the guess into
    year 0003 and no test looked at the guess itself.
    """
    res = _resolve("2025-03")
    assert res.value == pd.Timestamp(2025, 3, 1)
    assert res.risk == "high"


# ── the leading-zero identifier veto (const_num#113) ───────────────────────


@pytest.mark.parametrize("value,expected", [
    ("05", True), ("0123", True), ("007", True), ("00", True),
    ("10", False), ("100", False), ("1", False), ("0", False),
])
def test_a_leading_zero_marks_a_value_as_an_identifier(value, expected):
    """``05`` and ``0123`` are codes; ``10`` and ``100`` are numbers.

    This veto keeps the spelled-number and category experts off zero-padded
    codes. Every existing test used ``"007"`` -- which has a zero in its first
    *and* second place -- so checking the wrong character position went
    unnoticed: ``05`` and ``0123`` lost their protection and ``10`` gained it.
    """
    assert looks_like_identifier_value(value) is expected


# ── every expert proposes with a valid base confidence (const_num#120, #129) ──

_EXPERT_CORPUS = {
    "amount": ["$15.00", "$1,200.50", "(1,200)", "$20.00", "€1.234,56", "$1.200"],
    "price": ["$15.00", "£20.00", "€30.00", "$40.00", "¥500", "$60.00"],
    "weight_kg": ["5 kg", "12 kg", "7 lb", "3 kg", "8 kg", "10 g"],
    "active": ["yes", "no", "Y", "N", "true", "0"],
    "gender": ["M", "F", "male", "Female", "m", "f"],
    "status": ["Active", "active", "ACTIVE", "Inactive", "inactive", "Pending"],
    "qty": ["three", "twelve", "forty two", "7", "one hundred", "9"],
    "when": ["yesterday", "2021-03-15", "today", "March 5, 2021", "2021-04-01", "05/01/2021"],
}


def test_every_expert_proposes_with_a_base_confidence_in_range(monkeypatch):
    """A base confidence is a score in [0, 1] where it is produced.

    ``confidence_from_evidence`` clamps to 0.999 downstream, so a literal of
    1.01 in an expert (the currency and unit experts' 0.96, mutated) reached
    no report and failed no test. This checks the contract at the source, for
    every proposal every expert makes on a corpus touching each of them.
    """
    seen: list[float] = []
    real = experts.make_proposal

    def spy(**kwargs):
        seen.append(kwargs["base_confidence"])
        return real(**kwargs)

    monkeypatch.setattr(experts, "make_proposal", spy)
    for column, values in _EXPERT_CORPUS.items():
        fd.clean(pd.DataFrame({column: values * 6}), semantic_mode="auto",
                 fix_dtypes=False, verbose=False)
    assert len(seen) >= 30, f"corpus reached too few proposals ({len(seen)})"
    out_of_range = sorted({c for c in seen if not 0.0 <= c <= 1.0})
    assert not out_of_range, out_of_range


# ── an empty body is not ambiguous (const_bool#12) ──────────────────────────


@pytest.mark.parametrize("value", ["", "   ", "()", "( )"])
def test_nothing_between_the_parentheses_is_neither_money_nor_ambiguous(value):
    """``ambiguous`` describes a value; with no value there is nothing to be unsure of."""
    assert parse_currency_parts(value) == (None, False)


# ── equivalence proofs: survivors that cannot be killed ─────────────────────
#
# These are deliberately not source-text tripwires. The mutation harness
# writes each mutant through ``ast.unparse``, which reformats the whole file,
# so a test that greps ``inspect.getsource`` fails on *every* mutant and would
# report nearly all of them killed. Each proof quotes its premise instead.


def test_the_decimal_separator_tiebreak_cannot_tie():
    """``cmp#7``: ``rfind(".") > rfind(",")`` -> ``>=`` is equivalent.

    The comparison sits under ``if dots and commas:``, so both characters are
    present and each ``rfind`` is a real index. Two different characters never
    share an index, so the two sides are never equal and ``>`` and ``>=``
    agree. Checked exhaustively over every body of length <= 7 on the
    alphabet that matters.
    """
    for n in range(2, 8):
        for body in map("".join, itertools.product("1.,", repeat=n)):
            if "." in body and "," in body:
                assert body.rfind(".") != body.rfind(",")


def test_the_single_separator_tail_ignores_maxsplit():
    """``const_num#47``: ``body.rsplit(sep, 1)[1]`` -> ``rsplit(sep, 2)`` is equivalent.

    ``tail`` is read only in the branch where the separator occurs exactly
    once (``occurrences > 1`` short-circuits to grouping first). With one
    occurrence, ``rsplit`` yields the same two parts for any ``maxsplit >= 1``.
    Checked exhaustively over every body of length <= 7.
    """
    for sep in ".,":
        for n in range(1, 8):
            for body in map("".join, itertools.product(f"1{sep}", repeat=n)):
                if body.count(sep) == 1:
                    assert body.rsplit(sep, 1)[1] == body.rsplit(sep, 2)[1]


def test_the_integer_part_ignores_maxsplit():
    """``const_num#51``: ``split(decimal_sep, maxsplit=1)[0]`` -> ``maxsplit=2`` is equivalent.

    Element ``[0]`` is the text before the *first* separator, whatever the
    ``maxsplit`` (for any ``maxsplit >= 1``). Checked exhaustively over every
    body of length <= 7.
    """
    for sep in ".,":
        for n in range(1, 8):
            for body in map("".join, itertools.product("1.,", repeat=n)):
                assert body.split(sep, maxsplit=1)[0] == body.split(sep, maxsplit=2)[0]


def test_a_thirteenth_month_is_rejected_whichever_branch_reads_it():
    """``const_num#105``/``#107``: ``b <= 12`` -> ``b <= 13`` (and ``a``) are equivalent.

    The day/month decision is ``a > 12 and b <= 12`` (day-first), then
    ``b > 12 and a <= 12`` (month-first), then ``a > 12 and b > 12`` -> not a
    date. Loosening 12 to 13 changes only inputs where the would-be month is
    exactly 13. The original sends those to "not a date"; the mutant assigns
    month 13, and ``_safe_timestamp`` rejects month 13, so the result is
    ``None`` either way. Checked for every day and a range of years.
    """
    for year in (1970, 2000, 2024, 2099):
        for day in range(0, 100):
            assert experts._safe_timestamp(year, 13, day) is None
    for other in range(13, 100):
        assert _resolve(f"{other}/13/2024") is None
        assert _resolve(f"13/{other}/2024") is None


def test_a_grouping_check_never_sees_an_empty_remainder():
    """``const_bool#5``: ``if not rest: return True`` -> ``False`` is equivalent.

    ``_valid_grouping`` returns early unless ``sep in part``. Its only caller
    passes ``group_sep``, which is always ``"."`` or ``","``, and
    ``lstrip("+-")`` cannot remove either, so ``split(sep)`` yields at least
    two groups and ``rest`` is never empty. Checked exhaustively over every
    part of length <= 7 on the alphabet that matters.
    """
    for sep in ".,":
        for n in range(1, 8):
            for part in map("".join, itertools.product("1.,+-", repeat=n)):
                if sep in part:
                    assert len(part.lstrip("+-").split(sep)[1:]) >= 1
