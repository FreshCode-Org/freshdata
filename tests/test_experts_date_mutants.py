"""Boundary tests for the day/month disambiguation in ``semantic/experts.py``.

Mutation testing of ``_resolve_date`` and ``looks_like_date_value`` left a
cluster of survivors on the one decision that turns ``"05/12/2024"`` into
either May 12th or the 5th of December. Nothing else in the suite noticed when
the boundary was moved, so a mutant that misreads the order writes a *wrong
date* into the user's data and reports it as an unambiguous, low-risk,
auto-appliable repair. That is the worst failure mode this library has: a
silent semantic corruption that looks like a confident fix.

The value ``12`` is the whole crux, because it is the largest legal month:

    a > 12, b <= 12   -> the first token cannot be a month; day-first, certain
    b > 12, a <= 12   -> the second token cannot be a month; month-first, certain
    a > 12, b > 12    -> neither token can be a month; not a date at all
    otherwise         -> both <= 12, so only the ``dayfirst`` hint can decide,
                         and with no hint the reading stays ambiguous: a
                         guess is surfaced for audit at confidence 0.75 /
                         risk "high" so the policy gate never applies it

Each mutant loosened one of those comparisons (``>`` -> ``>=``, ``<=`` -> ``<``,
``and`` -> ``or``), which either promotes a genuinely ambiguous value to a
confident wrong answer or demotes a certain one to a guess. The tests below pin
the behaviour at exactly 12 on both sides, and at 12 against 13.

The second group covers ``looks_like_date_value``, the cheap shape probe that
decides whether a column is even *eligible* for date repair. Its boolean
returns and its ``or``-chain of date regexes could all be flipped unnoticed:
an ``and``-chain there would require a string to match all three mutually
exclusive patterns at once, so no value would ever look like a date and the
date expert would silently go dark on every dataset.

Mutants killed here (ids from the repo's mutation harness):
``cmp#16``, ``cmp#18``, ``cmp#19``, ``bool#12``, ``const_bool#19``..
``const_bool#23`` and ``const_bool#26``. ``cmp#20``, ``cmp#21`` and ``bool#16``
are proven equivalent in the last test rather than left unexplained.

(Re-running that harness needs ``PYTHONDONTWRITEBYTECODE=1``: every mutant of
this module unparses to within a byte or two of the same size and they are
written within the same mtime second, so CPython happily reuses the previous
mutant's cached bytecode and the reported verdicts drift between runs.)
"""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.semantic.experts import (
    _NUMERIC_DATE_RE,
    _resolve_date,
    is_plain_number,
    looks_like_date_value,
)


def _resolve(raw: str, dayfirst: bool | None = None):
    return _resolve_date(raw, dayfirst=dayfirst, reference_date=None)


# -- numeric day/month order: the boundary at 12 ------------------------------


def test_a_first_token_of_exactly_twelve_is_still_an_ambiguous_month():
    """``12/05`` must not be read as day 12 just because 12 is not below 13.

    Kills ``cmp#16`` (``a > 12`` -> ``a >= 12``): the mutant takes the
    "first token cannot be a month" branch for a token that can, and returns
    2024-05-12 at confidence 0.95 / risk "low" -- auto-appliable, and the
    wrong month.
    """
    res = _resolve("12/05/2024")
    assert res is not None
    assert res.value == pd.Timestamp(2024, 12, 5)
    assert res.confidence == 0.75
    assert res.risk == "high"
    assert "ambiguous" in res.detail


def test_a_first_token_of_exactly_twelve_obeys_an_explicit_dayfirst_hint():
    """With a hint, ``12/05`` reads both ways -- and only the hint decides.

    Also kills ``cmp#16``: the mutant would answer 2024-05-12 for *both*
    hints, ignoring ``dayfirst=False`` entirely.
    """
    assert _resolve("12/05/2024", dayfirst=True).value == pd.Timestamp(2024, 5, 12)
    assert _resolve("12/05/2024", dayfirst=False).value == pd.Timestamp(2024, 12, 5)


def test_a_second_token_of_exactly_twelve_is_still_an_ambiguous_day():
    """``05/12`` must not be read as month 5 just because 12 is not below 13.

    Kills ``cmp#18`` (``b > 12`` -> ``b >= 12``): the mutant takes the
    "second token cannot be a month" branch and reports the US reading as
    certain, so a European ``5 December`` column is silently rewritten to
    May 12th at risk "low".
    """
    res = _resolve("05/12/2024")
    assert res is not None
    assert res.value == pd.Timestamp(2024, 5, 12)
    assert res.confidence == 0.75
    assert res.risk == "high"
    assert "ambiguous" in res.detail


def test_a_second_token_of_exactly_twelve_obeys_an_explicit_dayfirst_hint():
    """Also kills ``cmp#18``: under the mutant ``dayfirst=True`` is ignored."""
    assert _resolve("05/12/2024", dayfirst=True).value == pd.Timestamp(2024, 12, 5)
    assert _resolve("05/12/2024", dayfirst=False).value == pd.Timestamp(2024, 5, 12)


def test_both_tokens_exactly_twelve_stay_ambiguous_without_a_hint():
    """``12/12`` is the fixed point of the boundary: neither token is excluded.

    The two readings happen to coincide, but the *decision* must still be the
    ambiguous one, because the rule is about which token is the month.
    """
    res = _resolve("12/12/2024")
    assert res is not None
    assert res.value == pd.Timestamp(2024, 12, 12)
    assert res.confidence == 0.75
    assert res.risk == "high"


def test_a_first_token_of_exactly_twelve_resolves_when_the_second_exceeds_twelve():
    """``12/13`` is certain: 13 cannot be a month, so 12 is the month.

    Kills ``cmp#19`` (``a <= 12`` -> ``a < 12``): the mutant refuses the
    month-first branch for a first token of exactly 12, drops through to the
    no-hint fallback and downgrades a *certain* date to a 0.75/"high" guess,
    so a perfectly resolvable column stops being repaired.
    """
    res = _resolve("12/13/2024")
    assert res is not None
    assert res.value == pd.Timestamp(2024, 12, 13)
    assert res.confidence == 0.95
    assert res.risk == "low"
    assert "unambiguously" in res.detail


@pytest.mark.parametrize("raw", ["12/13/2024", "12/31/2024", "12-25-2024"])
def test_a_first_token_of_twelve_against_a_larger_second_ignores_the_dayfirst_hint(raw):
    """Evidence beats convention: 13/31/25 cannot be a month under any hint.

    Also kills ``cmp#19``: under the mutant these fall through to the hint
    branches, where ``dayfirst=True`` reads them as day 12 of month 13/31/25
    -- an impossible date, silently returned as "not a date" (``None``).
    """
    expected = _resolve(raw).value
    assert expected is not None
    assert _resolve(raw, dayfirst=True).value == expected
    assert _resolve(raw, dayfirst=False).value == expected


def test_a_second_token_of_exactly_twelve_resolves_when_the_first_exceeds_twelve():
    """``13/12`` is the mirror case: 13 cannot be a month, so 12 is."""
    res = _resolve("13/12/2024")
    assert res is not None
    assert res.value == pd.Timestamp(2024, 12, 13)
    assert res.confidence == 0.95
    assert res.risk == "low"
    assert "unambiguously" in res.detail


def test_two_tokens_that_both_exceed_twelve_are_not_a_date():
    """``13/13`` has no month at all, so the value is left alone entirely."""
    assert _resolve("13/13/2024") is None
    assert _resolve("13/13/2024", dayfirst=True) is None
    assert _resolve("31/31/2024", dayfirst=False) is None


def test_an_explicit_dayfirst_false_resolves_an_otherwise_ambiguous_value():
    """``dayfirst=False`` is a real branch, not a synonym for "no hint".

    Kills ``const_bool#26`` (``dayfirst is False`` -> ``dayfirst is True``):
    the mutant's condition can never hold (an earlier branch already claimed
    ``dayfirst is True``), so an explicit month-first hint silently decays
    into the ambiguous fallback. The guessed *value* is identical there, which
    is exactly why no existing test caught it -- only the confidence and risk
    change, and with them whether the repair is ever applied.
    """
    res = _resolve("05/06/2024", dayfirst=False)
    assert res is not None
    assert res.value == pd.Timestamp(2024, 5, 6)
    assert res.confidence == 0.95
    assert res.risk == "low"
    assert "unambiguously" in res.detail

    hinted_the_other_way = _resolve("05/06/2024", dayfirst=True)
    assert hinted_the_other_way.value == pd.Timestamp(2024, 6, 5)
    assert hinted_the_other_way.confidence == 0.95

    no_hint = _resolve("05/06/2024")
    assert no_hint.value == pd.Timestamp(2024, 5, 6)  # same guess, but only a guess
    assert no_hint.confidence == 0.75
    assert no_hint.risk == "high"


# -- the eligibility probe ----------------------------------------------------


@pytest.mark.parametrize("value", [None, 12, 20240512, 3.5, True, ["2024-05-12"]])
def test_a_non_string_never_looks_like_a_date_value(value):
    """Kills ``const_bool#20``: the mutant calls every non-string a date,
    so an integer column of ids becomes eligible for date repair."""
    assert looks_like_date_value(value) is False


@pytest.mark.parametrize("text", ["", " ", "\t\n", "   "])
def test_a_blank_string_never_looks_like_a_date_value(text):
    """Kills ``const_bool#21``: the mutant makes empty cells look like dates."""
    assert looks_like_date_value(text) is False


@pytest.mark.parametrize("text", ["today", "Yesterday", " TOMORROW ", "Today"])
def test_relative_date_phrases_look_like_date_values(text):
    """Kills ``const_bool#22``: the mutant hides ``today``/``yesterday`` from
    the eligibility probe, so columns full of relative phrases -- the very
    values that most need review -- are never even considered."""
    assert looks_like_date_value(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "2024-05-12",  # only _ISO_DATE_RE matches
        "2024/05/12",  # only _ISO_SLASH_DATE_RE matches
        "05/12/2024",  # only _NUMERIC_DATE_RE matches
        "5-12-2024",  # only _NUMERIC_DATE_RE matches, dash form
    ],
)
def test_each_numeric_date_shape_looks_like_a_date_value_on_its_own(text):
    """Kills ``bool#12`` and ``const_bool#23``.

    The three patterns are mutually exclusive, so the ``and``-chain mutant
    (``bool#12``) can never be satisfied by any string, and flipping the
    ``return True`` (``const_bool#23``) has the same effect: every numeric
    date shape stops looking like a date and the date expert goes dark.
    """
    assert looks_like_date_value(text) is True


@pytest.mark.parametrize("text", ["May 12, 2024", "12 May 2024", "12 Sept. 2024"])
def test_month_name_dates_look_like_date_values(text):
    assert looks_like_date_value(text) is True


@pytest.mark.parametrize("text", ["hello", "12", "2024", "SKU-30", "not a date"])
def test_non_date_strings_do_not_look_like_date_values(text):
    assert looks_like_date_value(text) is False


@pytest.mark.parametrize("value", [None, [1], {"a": 1}, object(), pd.Timestamp("2024-01-01")])
def test_values_of_an_unhandled_type_are_not_plain_numbers(value):
    """Kills ``const_bool#19``: the fall-through of ``is_plain_number`` must be
    ``False``. The mutant declares every unhandled type -- ``None``, lists,
    timestamps -- an ordinary number, which mis-routes those columns to the
    numeric experts."""
    assert is_plain_number(value) is False


# -- equivalence proof --------------------------------------------------------


def test_the_neither_token_is_a_month_branch_cannot_be_loosened():
    """Proof that three surviving mutants are equivalent, not missing tests.

    ``_resolve_date`` decides the numeric order with::

        if a > 12 and b <= 12:      # (1) day-first, certain
        elif b > 12 and a <= 12:    # (2) month-first, certain
        elif a > 12 and b > 12:     # (3) neither token can be a month

    Three mutants of branch (3) survive every test, and none can be killed:

    * ``cmp#20``  ``a >= 12 and b > 12``
    * ``cmp#21``  ``a > 12 and b >= 12``
    * ``bool#16`` ``a > 12 or b > 12``

    Branch (3) is only ever evaluated when (1) and (2) are both false, and on
    that reachable set all four predicates coincide:

    * ``cmp#20`` differs from the original only at ``a == 12 and b > 12`` --
      but that state satisfies (2) (``b > 12 and 12 <= 12``), so (3) is never
      reached there;
    * ``cmp#21`` differs only at ``a > 12 and b == 12`` -- that state
      satisfies (1) (``a > 12 and 12 <= 12``);
    * ``bool#16`` differs only when exactly one token exceeds 12 -- those are
      precisely the states claimed by (1) and (2).

    ``_NUMERIC_DATE_RE`` captures ``\\d{1,2}`` for both tokens, so the entire
    input domain is ``0..99`` squared and the argument is checked by
    exhaustion below rather than only argued in prose.
    """
    assert _NUMERIC_DATE_RE.match("100/05/2024") is None  # domain really is 0..99
    assert _NUMERIC_DATE_RE.match("99/99/2024") is not None

    reached = 0
    for a in range(100):
        for b in range(100):
            if (a > 12 and b <= 12) or (b > 12 and a <= 12):
                continue  # branch (1) or (2) returns first
            reached += 1
            original = a > 12 and b > 12
            assert (a >= 12 and b > 12) == original, (a, b)  # cmp#20
            assert (a > 12 and b >= 12) == original, (a, b)  # cmp#21
            assert (a > 12 or b > 12) == original, (a, b)  # bool#16
    assert reached == 13 * 13 + 87 * 87  # both <= 12, or both > 12
