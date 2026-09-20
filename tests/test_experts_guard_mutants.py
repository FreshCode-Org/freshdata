"""The `applies()` guards that decide whether a semantic expert runs at all.

Mutation testing of `freshdata.semantic.experts` left the *entry* guards of the
value experts uncovered: the `and` that pairs a shape flag with `not
free_text`, the `or` chain that vetoes free-text/identifier/boolean columns,
and the `return False` early exits those guards use. Flipping any of them is
the "identifier-protection removed" mutation class, and it is the most
dangerous one in the library: a value expert that runs on an identifier column
will happily rewrite `"007"` to `7` and `"0001"` to `1`, silently destroying
leading zeros in customer ids, SKUs, ZIP codes and account numbers. The damage
is unrecoverable from the output alone -- nothing downstream can tell whether
`7` was ever `"007"`.

The identifier carve-out has five independent protection layers; these guards
are one of them, and a layer that is untested is a layer that is not there.
Each test below kills a specific confirmed survivor and names it, so that if
an assertion is ever weakened the reason it existed is on the page.

Sibling suites: `test_policy_guard_mutants.py` (the policy gate and the
byte-identity guard) and `test_guard_protected.py` (end-to-end protection).
"""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.semantic.experts import (
    CategorySynonymExpert,
    CurrencyStringExpert,
    DatePhraseExpert,
    IdentifierProtectionExpert,
    SpelledNumberExpert,
    looks_like_identifier_value,
)
from freshdata.semantic.types import SemanticColumnInfo


def _info(**kw) -> SemanticColumnInfo:
    """A column info with every shape flag off; override only what matters."""
    base = {
        "name": "cust_id",
        "role": "categorical",
        "n_nonnull": 3,
        "nunique": 3,
        "high_cardinality": False,
        "preserve": False,
        "free_text": False,
        "numeric_like": False,
        "boolean_like": False,
        "money_like": False,
        "unit_like": False,
        "identifier_like": False,
    }
    base.update(kw)
    return SemanticColumnInfo(**base)


# -- a shape flag is necessary, never merely sufficient-by-absence ----------


def test_the_spelled_number_expert_stays_off_a_column_that_is_not_numeric_like():
    """Kills experts bool#20 (`numeric_like and not free_text` -> `or not`).

    With `or`, every column that merely fails to be free text -- which is most
    columns, including every identifier column -- satisfied the guard, and the
    number-word rewriter ran on ids, SKUs and postcodes.
    """
    assert SpelledNumberExpert().applies(_info(role="text")) is False
    assert SpelledNumberExpert().applies(_info(identifier_like=True)) is False


def test_the_spelled_number_expert_runs_on_a_plain_numeric_column():
    """The guard must still let the expert do its job."""
    assert SpelledNumberExpert().applies(_info(numeric_like=True)) is True


def test_the_spelled_number_expert_stays_off_free_text_even_when_numeric_like():
    assert SpelledNumberExpert().applies(_info(numeric_like=True, free_text=True)) is False


def test_the_currency_expert_stays_off_a_column_that_is_not_money_like():
    """Kills experts bool#23 (`money_like and not free_text` -> `or not`).

    Same failure shape as bool#20: `or` turned "not free text" into a licence
    to strip currency formatting from any column, identifiers included.
    """
    assert CurrencyStringExpert().applies(_info(role="text")) is False
    assert CurrencyStringExpert().applies(_info(identifier_like=True)) is False


def test_the_currency_expert_runs_on_a_plain_money_column():
    assert CurrencyStringExpert().applies(_info(money_like=True)) is True


def test_the_currency_expert_stays_off_free_text_even_when_money_like():
    assert CurrencyStringExpert().applies(_info(money_like=True, free_text=True)) is False


# -- the category veto needs ONE reason, not all three ----------------------


@pytest.mark.parametrize("flag", ["free_text", "identifier_like", "boolean_like"])
def test_the_category_expert_is_vetoed_by_any_single_disqualifying_flag(flag):
    """Kills experts bool#28 (`free_text or identifier_like or boolean_like`
    -> all `and`) and const_bool#32 (that branch's `return False` -> `True`).

    Under `and` the veto fired only for a column that was free text *and* an
    identifier *and* boolean -- i.e. never -- so the category normalizer ran on
    identifier columns and folded `"007"`/`"07"` case-and-whitespace variants
    together. Under the constant flip the veto branch *admitted* the column it
    had just disqualified. Each flag on its own must stop the expert.
    """
    assert CategorySynonymExpert().applies(_info(**{flag: True})) is False


def test_the_category_expert_runs_on_an_ordinary_categorical_column():
    """The veto must not be the whole guard: a clean categorical still applies."""
    assert CategorySynonymExpert().applies(_info(role="categorical")) is True


def test_the_category_expert_defers_to_the_reference_expert_when_allowed_values_exist():
    """Kills experts const_bool#33 (the `allowed_values` `return False` ->
    `True`).

    An explicit reference list is handled by `ReferenceExpert`, which has fuzzy
    matching and ambiguity handling. Flipping this exit made both experts
    propose on the same values, producing duplicate and potentially conflicting
    repairs for one cell.
    """
    assert CategorySynonymExpert().applies(_info(allowed_values=("active", "inactive"))) is False


# -- the date expert is vetoed by identifier-likeness on its own ------------


def test_the_date_expert_stays_off_an_identifier_column_that_also_reads_as_a_date():
    """Kills experts bool#33 (`date_like and not free_text and not
    identifier_like` -> all `or`).

    Under `or` the guard was satisfied by *not* being an identifier, or by not
    being free text, so it passed for essentially every column -- and an
    identifier column of numeric codes that happens to parse as dates would be
    converted to timestamps outright. Date-shaped identifiers are common
    (`"20240115"`-style batch codes), which is why this veto exists.
    """
    assert DatePhraseExpert().applies(_info(date_like=True, identifier_like=True)) is False
    assert DatePhraseExpert().applies(_info(date_like=True, free_text=True)) is False
    assert DatePhraseExpert().applies(_info(identifier_like=True)) is False


def test_the_date_expert_needs_the_column_to_actually_read_as_a_date():
    """The `or` mutant also let a column with no date evidence through."""
    assert DatePhraseExpert().applies(_info()) is False


def test_the_date_expert_runs_on_a_clean_date_column():
    assert DatePhraseExpert().applies(_info(date_like=True)) is True


# -- the leading-zero identifier test ---------------------------------------


def test_a_single_zero_is_not_an_identifier_but_a_zero_padded_number_is():
    """Kills experts cmp#22 (`len(s) > 1` -> `len(s) >= 1`).

    The length check is what makes this the *leading-zero* test: a zero is only
    a pad when something follows it. With `>=` the bare string `"0"` -- an
    ordinary value in a count, flag or score column -- was classified as an
    identifier, so the spelled-number and category experts silently skipped it
    while repairing every value around it, leaving the column half-normalized.
    `"0"` is the only input where the two versions differ.
    """
    assert looks_like_identifier_value("0") is False
    assert looks_like_identifier_value("00") is True
    assert looks_like_identifier_value("007") is True
    assert looks_like_identifier_value("7") is False


def test_the_leading_zero_test_ignores_surrounding_whitespace():
    """The value is stripped first, so padding cannot smuggle a code through."""
    assert looks_like_identifier_value(" 007 ") is True
    assert looks_like_identifier_value(" 0 ") is False


# -- the protective veto record is only emitted when it is earned -----------


def test_the_identifier_veto_records_nothing_for_values_no_expert_would_touch():
    """Kills experts const_bool#37 (`transformable`'s final `return False` ->
    `True`).

    `IdentifierProtectionExpert` exists to leave an audit entry when protection
    actually did something. With the flip every non-string value counted as
    transformable, so a plain integer id column produced a protective record
    describing a veto that never had anything to veto -- audit noise that makes
    the real protective records harder to find.
    """
    info = _info(identifier_like=True)
    assert IdentifierProtectionExpert().propose(pd.Series([1, 2, 3]), info) == []


def test_the_identifier_veto_records_a_skip_when_a_value_expert_would_have_struck():
    """The counterpart: zero-padded codes are exactly what protection is for."""
    info = _info(identifier_like=True)
    proposals = IdentifierProtectionExpert().propose(pd.Series(["007", "008"]), info)
    assert len(proposals) == 1
    assert proposals[0].issue_type == "identifier_like"
    assert proposals[0].proposed_value is None  # a no-op on the data by design
