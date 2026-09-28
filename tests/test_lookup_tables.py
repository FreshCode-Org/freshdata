"""Every entry of the number-word and month-name tables, pinned.

Mutation testing of ``semantic/experts.py`` left 98 numeric survivors. Sorted
by syntactic context, only 7 are operands of a comparison -- the thresholds
the brief names. 45 are dict values, and those turn out to be two lookup
tables: the spelled-number map (``_ONES``/``_TENS``/``_BIG``) and the
month-name map (``_MONTHS``). Only 10 of the 30 number words appear anywhere
in ``tests/``, and the month names not at all, so changing ``"seven": 7`` to
``"seven": 8`` or ``"march": 3`` to ``"march": 4`` broke nothing.

Neither is a defect today -- both tables are correct. The gap is that a typo
in either would ship, and the consequence is a wrong *value*, silently: a
column of spelled numbers converts to the wrong integers, or ``"14 March
2021"`` is written as April 14th and reported as an unambiguous, low-risk,
auto-appliable repair.

Every expectation below is written out literally. Deriving it from the table
under test would make the test agree with any table, including a broken one.
"""
from __future__ import annotations

import pandas as pd
import pytest

from freshdata.semantic.experts import _resolve_date, parse_number_words


def _resolve(raw: str, dayfirst: bool | None = None):
    """``_resolve_date`` with its two required keyword arguments supplied."""
    return _resolve_date(raw, dayfirst=dayfirst, reference_date=None)

# ── spelled numbers ─────────────────────────────────────────────────────────

ONES = [
    ("zero", 0), ("one", 1), ("two", 2), ("three", 3), ("four", 4),
    ("five", 5), ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9),
    ("ten", 10), ("eleven", 11), ("twelve", 12), ("thirteen", 13),
    ("fourteen", 14), ("fifteen", 15), ("sixteen", 16), ("seventeen", 17),
    ("eighteen", 18), ("nineteen", 19),
]
TENS = [
    ("twenty", 20), ("thirty", 30), ("forty", 40), ("fifty", 50),
    ("sixty", 60), ("seventy", 70), ("eighty", 80), ("ninety", 90),
]
BIG = [("thousand", 1_000), ("million", 1_000_000), ("billion", 1_000_000_000)]


@pytest.mark.parametrize("word,expected", ONES + TENS, ids=str)
def test_every_number_word_parses_to_its_own_value(word, expected):
    assert parse_number_words(word) == expected


@pytest.mark.parametrize("word,scale", BIG, ids=str)
def test_every_scale_word_multiplies_by_its_own_factor(word, scale):
    """A bare scale word means one of it; a preceding count multiplies it."""
    assert parse_number_words(word) == scale
    assert parse_number_words(f"three {word}") == 3 * scale


@pytest.mark.parametrize("word,expected", TENS, ids=str)
def test_a_tens_word_composes_with_every_unit(word, expected):
    """``forty two`` is 42, not 40 or 2 -- the two tables must combine."""
    for unit, value in ONES[1:10]:
        assert parse_number_words(f"{word} {unit}") == expected + value
        assert parse_number_words(f"{word}-{unit}") == expected + value


def test_the_number_words_are_all_distinct():
    """Two words mapping to one value would make a table typo invisible above."""
    values = [v for _, v in ONES + TENS + BIG]
    assert len(set(values)) == len(values)


def test_an_unknown_token_still_makes_the_whole_phrase_a_non_number():
    """The tables must not be consulted token-by-token in isolation."""
    assert parse_number_words("twenty apples") is None
    assert parse_number_words("apples twenty") is None
    assert parse_number_words("three hundred dollars") is None


# ── month names ─────────────────────────────────────────────────────────────

MONTHS = [
    ("jan", 1), ("january", 1), ("feb", 2), ("february", 2),
    ("mar", 3), ("march", 3), ("apr", 4), ("april", 4), ("may", 5),
    ("jun", 6), ("june", 6), ("jul", 7), ("july", 7), ("aug", 8),
    ("august", 8), ("sep", 9), ("sept", 9), ("september", 9),
    ("oct", 10), ("october", 10), ("nov", 11), ("november", 11),
    ("dec", 12), ("december", 12),
]


@pytest.mark.parametrize("name,month", MONTHS, ids=str)
def test_every_month_name_resolves_to_its_own_month(name, month):
    """Both orderings the month-name pattern accepts, and case-insensitively."""
    for raw in (f"{name} 14, 2021", f"14 {name} 2021", f"14 {name.upper()} 2021"):
        res = _resolve(raw)
        assert res is not None, raw
        assert res.value == pd.Timestamp(2021, month, 14), raw


def test_the_month_table_covers_every_month_exactly_once():
    """A missing month would be left to the ambiguous numeric path instead."""
    assert sorted({m for _, m in MONTHS}) == list(range(1, 13))


@pytest.mark.parametrize("name,month", MONTHS, ids=str)
def test_a_month_name_date_is_never_reported_as_ambiguous(name, month):
    """A named month settles the order, so no ``dayfirst`` hint is consulted."""
    res = _resolve(f"14 {name} 2021")
    assert res.risk == "low"
    for hint in (True, False, None):
        assert _resolve(f"14 {name} 2021", dayfirst=hint).value == res.value


def test_an_unknown_month_name_is_not_a_date():
    """A word-shaped token that is not a month must not fall back to a guess."""
    assert _resolve("14 smarch 2021") is None
    assert _resolve("smarch 14, 2021") is None
