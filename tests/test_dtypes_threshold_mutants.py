"""Boundary tests for the silent-coercion thresholds in ``steps/dtypes.py``.

``fix_dtypes`` is the step that rewrites a column's type without being asked.
Two families of constants decide how aggressive that is:

* **parse shares** — how much of a column must parse before the whole column
  is converted (``numeric_threshold`` / ``datetime_threshold``, the
  ``* 0.8`` cheap sample screens, and ``CONTAMINATION_SHARE``);
* **collateral budgets** — how many cells may be coerced to missing while
  converting (``max(3, 0.1 * n)``), and how many casualties are recorded
  (``COERCED_CELLS_CAP``).

Mutation testing of the module showed that every one of those comparisons
could be loosened or tightened (``>=`` -> ``>``, ``<`` -> ``<=``), and every
``and`` in the datetime decision chain turned into an ``or``, without a single
test in the 7339-test suite failing. Each test below pins one of them at the
value, just below it and just above it -- the way
``tests/test_scoring_boundaries.py`` pins the risk tiers -- and names the
mutant it kills, so weakening the boundary again has to argue with the reason.

Nothing here duplicates ``tests/test_dtypes.py``: that file pins the *default*
95% threshold end to end, while these pin the surrounding decision boundaries
(sample screens, contamination reporting, coercion budgets, the mixed-format
datetime fallback chain) that the default-path tests never reach.
"""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata._util import PANDAS_MAJOR, sample_series
from freshdata.config import CleanConfig
from freshdata.report import CleanReport
from freshdata.steps.dtypes import (
    COERCED_CELLS_CAP,
    CONTAMINATION_SHARE,
    _finalize_numeric,
    _record_coerced,
    fix_dtypes,
    refine_numeric_after_semantic,
    suggest_conversion,
)

#: A value that looks like a date to the pre-screen but parses to NaT in every
#: format (day 32 does not exist).
UNPARSEABLE_DATEISH = "13/32/2021"


def _suggest(values, **options):
    """``suggest_conversion`` for one column built from *values*."""
    return suggest_conversion(pd.Series(values, dtype=object), CleanConfig(**options))


def _warnings_for(values, **options) -> list[str]:
    """Warnings ``fix_dtypes`` records for a single column named ``amount``."""
    report = CleanReport()
    fix_dtypes(pd.DataFrame({"amount": values}), CleanConfig(**options), report)
    return report.warnings


def _contamination_warnings(values, **options) -> list[str]:
    return [w for w in _warnings_for(values, **options) if "looks numeric" in w]


def _sample_order(n: int, size: int, random_state: int = 0) -> list[int]:
    """The positions ``sample_series`` draws, in the order it draws them.

    The screens under test run on the sample, not the column, so a test that
    needs the sample to disagree with the column has to know which positions
    land in it. ``sample_series`` samples by position and ignores the values.
    """
    return list(sample_series(pd.Series(range(n)), size, random_state).index)


# ---------------------------------------------------------------------------
# The locale number format (cmp#1, cmp#2)
# ---------------------------------------------------------------------------


def test_the_default_decimal_point_is_not_a_second_pass_trigger():
    """Kills cmp#1: ``config.decimal != "."`` -> ``==`` in ``noise_chars``.

    The "worth a second pass" test exists to catch *formatted* numbers --
    currency and grouping separators. A bare ``"."`` is not evidence of
    formatting, and admitting it hands the formatted-number path columns that
    a plain parse already rejected: ``"+ 1.5"`` is not a number to pandas, but
    the second pass strips its whitespace and would convert the column.
    """
    plain_decimals = ["+ 1.5", "+ 2.5", "+ 3.5", "+ 4.5"]
    assert pd.to_numeric(pd.Series(plain_decimals), errors="coerce").isna().all()
    target, converted, n_coerced = _suggest(plain_decimals)
    assert (target, converted, n_coerced) == ("none", None, 0)

    # Control: real formatting characters *do* trigger the second pass, so the
    # test above pins the contents of the noise set, not a dead code path.
    target, converted, _ = _suggest(["$ 1.5", "$ 2.5", "$ 3.5", "$ 4.5"])
    assert target == "numeric"
    assert converted.tolist() == [1.5, 2.5, 3.5, 4.5]


def test_a_european_decimal_comma_is_rewritten_before_parsing():
    """Kills cmp#2: ``config.decimal != "."`` -> ``==`` in ``cleanup``.

    Under the mutant the rewrite runs only for the locale that does not need
    it, so ``"1.234,56"`` reaches ``to_numeric`` as ``"1234,56"`` and the
    column silently stays text.
    """
    target, converted, n_coerced = _suggest(
        ["1.234,56", "2.000,10", "3.500,75", "900,25"], decimal=",", thousands="."
    )
    assert target == "numeric"
    assert converted.tolist() == [1234.56, 2000.10, 3500.75, 900.25]
    assert n_coerced == 0


# ---------------------------------------------------------------------------
# int64 admission (cmp#5)
# ---------------------------------------------------------------------------


def test_a_column_with_no_parsed_values_is_never_called_integral():
    """Kills cmp#5: ``len(nonnull) > 0`` -> ``>= 0``.

    ``(empty % 1 == 0).all()`` is ``True`` -- the emptiness guard is the only
    thing standing between an all-missing column and the integral branch. With
    a nullable dtype the mutant does not merely mislabel the column, it raises
    ``TypeError`` out of ``int(pd.NA)``.
    """
    all_missing = _finalize_numeric(pd.Series([pd.NA], dtype="Int64"))
    assert str(all_missing.dtype) == "float64"
    assert bool(all_missing.isna().all())

    empty = _finalize_numeric(pd.Series([], dtype="float64"))
    assert str(empty.dtype) == "float64"

    # One parsed value is enough to reach the integral branch.
    assert str(_finalize_numeric(pd.Series([1.0])).dtype) == "int64"


# ---------------------------------------------------------------------------
# The boolean vocabulary window (cmp#7)
# ---------------------------------------------------------------------------


def test_the_whole_boolean_vocabulary_still_converts():
    """Kills cmp#7: ``len(uniques) > 8`` -> ``>= 8``.

    The vocabulary has exactly eight spellings; a column using all of them is
    the boundary case the window exists to admit.
    """
    all_eight = ["true", "t", "yes", "y", "false", "f", "no", "n"]
    assert len(all_eight) == 8
    target, converted, _ = _suggest(all_eight)
    assert target == "boolean"
    assert converted.tolist() == [True, True, True, True, False, False, False, False]

    # A ninth spelling is one too many: not a boolean column.
    assert _suggest([*all_eight, "maybe"])[0] == "none"


# ---------------------------------------------------------------------------
# The numeric sample screen and the formatted-number threshold (cmp#10, cmp#13)
# ---------------------------------------------------------------------------


def test_the_numeric_sample_screen_admits_its_exact_boundary_rate():
    """Kills cmp#10: ``sample rate >= threshold * 0.8`` -> ``>``.

    The screen is a cheap rejection on a sample, deliberately slacker than the
    real threshold (``* 0.8``) so a sample that under-represents the column
    does not veto a conversion the column would pass. A column sitting exactly
    on the screen must therefore still be parsed in full: here the sample
    parses 2/5 = 0.40 == 0.5 * 0.8 while the column parses 7/10.
    """
    sampled = set(_sample_order(10, 5))
    values = [str(i) if i not in sampled else "abc" for i in range(10)]
    for position in sorted(sampled)[:2]:  # exactly 2 of the 5 sampled parse
        values[position] = str(position)
    assert len(values) == 10

    target, converted, n_coerced = _suggest(
        values, numeric_threshold=0.5, sample_size=5
    )
    assert target == "numeric"
    assert n_coerced == 3
    assert int(converted.notna().sum()) == 7


def test_the_formatted_number_pass_converts_at_exactly_the_threshold():
    """Kills cmp#13: ``rate < threshold`` -> ``<=`` in the second pass.

    ``threshold`` is the share that *must* parse, so a column parsing exactly
    that share converts. Here the currency values rescue the column to exactly
    0.50 with ``numeric_threshold=0.5``.
    """
    values = ["$1,000", "$2,000", "abc", "def"]
    target, converted, n_coerced = _suggest(values, numeric_threshold=0.5)
    assert target == "numeric"
    assert converted.dropna().tolist() == [1000, 2000]
    assert n_coerced == 2

    # A hair below the same threshold declines the column.
    assert _suggest([*values, "ghi"], numeric_threshold=0.5)[0] == "none"


# ---------------------------------------------------------------------------
# The datetime decision chain (cmp#24, bool#12, bool#14, bool#15, bool#16)
# ---------------------------------------------------------------------------


#: ``format="mixed"`` -- and with it the whole second-chance branch of the
#: datetime chain -- exists only on pandas >= 2, which the source guards with
#: ``PANDAS_MAJOR >= 2``. On pandas 1 the mutants below are equivalent and the
#: behaviour these tests pin is not reachable.
mixed_format_parsing = pytest.mark.skipif(
    PANDAS_MAJOR < 2, reason='format="mixed" parsing needs pandas >= 2'
)


def test_the_plain_datetime_screen_gates_the_full_column_parse():
    """Kills bool#12 and bool#14: ``sample is not None and rate >= ...``
    -> ``or``.

    Under either mutant the ``is not None`` test alone decides, the sample rate
    is never consulted, and a column whose sample says "not dates" is parsed
    and converted anyway. The sample here is five unparseable date-shaped
    values; the other 95 are ISO dates, so the column *would* clear the 95%
    threshold if the screen let it through.
    """
    sampled = set(_sample_order(100, 5))
    values = [
        UNPARSEABLE_DATEISH if i in sampled else f"2021-01-{(i % 28) + 1:02d}"
        for i in range(100)
    ]
    assert sum(v == UNPARSEABLE_DATEISH for v in values) == 5

    assert _suggest(values, sample_size=5) == ("none", None, 0)

    # Control: with the screen satisfied the same 95/100 column does convert.
    assert _suggest(values, sample_size=100)[0] == "datetime"


@mixed_format_parsing
def test_the_mixed_format_screen_admits_its_exact_boundary_rate():
    """Kills cmp#24: ``mixed sample rate >= threshold * 0.8`` -> ``>``.

    The mixed-format screen is the second chance for a column whose single
    inferred format came up short. A sample sitting exactly on the screen
    (2/5 = 0.40 == 0.5 * 0.8) must still get the full mixed parse, which here
    converts 7 of the 10 values.
    """
    order = _sample_order(10, 5)
    sampled = set(order)
    # Month-name values parse only under format="mixed" once the inferred
    # single format comes from the ISO value that leads the sample.
    values = ["Jan 5, 2021"] * 10
    values[order[0]] = "2021-06-07"  # plain-parseable, fixes the inferred format
    for position in order[2:]:  # three unparseable values complete the sample
        values[position] = UNPARSEABLE_DATEISH
    assert sum(v == UNPARSEABLE_DATEISH for v in values) == 3
    assert len(sampled) == 5

    target, converted, n_coerced = _suggest(
        values, datetime_threshold=0.5, sample_size=5
    )
    assert target == "datetime"
    assert n_coerced == 3
    assert int(converted.notna().sum()) == 7


@mixed_format_parsing
def test_a_column_that_clears_the_threshold_is_not_reparsed_as_mixed():
    """Kills bool#15: ``rate < threshold and PANDAS_MAJOR >= 2`` -> ``or``.

    On pandas 2 the right operand is always true, so ``or`` makes the
    mixed-format re-parse unconditional: a column that already cleared the
    threshold is re-parsed and, whenever the slower parser recovers one more
    cell, silently converted from a *different* parse than the one that was
    validated. The nanosecond timestamp below is exactly such a cell.
    """
    values = [f"2021-01-{i + 1:02d}" for i in range(19)]
    values.append("2021-02-06 11:00:00.987654321")
    target, converted, n_coerced = _suggest(values)
    assert target == "datetime"
    assert n_coerced == 1  # the nanosecond value is a coercion casualty
    assert bool(converted.isna().iloc[-1])


@mixed_format_parsing
def test_a_failed_mixed_reparse_leaves_the_plain_parse_in_place():
    """Kills bool#16: ``mixed is not None and ...`` -> ``or``.

    ``_parse_datetime`` returns ``None`` when pandas refuses the column
    outright -- here a tz-aware timestamp next to naive strings, which
    ``format="mixed"`` rejects. The ``is not None`` guard is the only thing
    that keeps the comparison from dereferencing it, so the mutant raises
    ``AttributeError`` instead of declining the column.
    """
    values = [
        "2021-01-05",
        "2021-02-06",
        "2021-03-07",
        pd.Timestamp("2021-04-08", tz="UTC"),
        UNPARSEABLE_DATEISH,
    ]
    assert _suggest(values) == ("none", None, 0)


# ---------------------------------------------------------------------------
# The contamination warning (cmp#29, cmp#30, cmp#31, cmp#32)
# ---------------------------------------------------------------------------


def test_the_contamination_report_starts_at_four_values():
    """Kills cmp#29: ``len(nonnull) < 4`` -> ``<=``.

    Four is the smallest column the warning is willing to describe; the mutant
    silently raises the floor to five.
    """
    assert len(_contamination_warnings(["1", "2", "3", "x"])) == 1
    assert _contamination_warnings(["1", "2", "x"]) == []


def test_the_contamination_share_floor_is_inclusive():
    """Kills cmp#30: ``CONTAMINATION_SHARE <= share`` -> ``<``.

    ``0.6`` is the documented boundary fieldcheck's consensus inference
    honours; a column sitting exactly on it is contaminated, not textual.
    """
    exactly_at_the_floor = ["1", "2", "3", "x1", "x2"]  # 3/5 == 0.6 parse
    assert len(exactly_at_the_floor) == 5
    assert CONTAMINATION_SHARE == 3 / 5
    [warning] = _contamination_warnings(exactly_at_the_floor)
    assert "60% of values parse" in warning

    # Just below the floor the column is legitimately textual: stay silent.
    assert _contamination_warnings(["1", "2", "x1", "x2", "x3"]) == []


def test_a_sample_that_parses_completely_is_not_contamination():
    """Kills cmp#31: ``share < 1.0`` -> ``<=``.

    The upper bound is what makes this a report about *contamination*: a
    sample in which everything parses is evidence of a clean numeric sample,
    not of a few bad cells. Here the sample parses 5/5 while three values
    outside it do not -- the mutant reports the column on the strength of a
    sample that saw none of them.
    """
    sampled = set(_sample_order(10, 5))
    unsampled = [i for i in range(10) if i not in sampled]
    values = [str(i) for i in range(10)]
    for position in unsampled[:3]:
        values[position] = f"x{position}"

    assert _contamination_warnings(values, sample_size=5) == []

    # Control: once the sample sees them, the same column is reported.
    assert len(_contamination_warnings(values, sample_size=10)) == 1


def test_the_contamination_warning_tolerates_exactly_the_bad_cell_budget():
    """Kills cmp#32: ``len(bad) > max(3, 0.1 * n)`` -> ``>=``.

    "A few odd values" is defined as at most ``max(3, 10%)`` of the column.
    Three bad cells in ten is the budget exactly and must be reported; four is
    a non-numeric minority -- a legitimately textual column -- and must stay
    silent.
    """
    at_the_budget = [str(i) for i in range(7)] + ["x1", "x2", "x3"]
    [warning] = _contamination_warnings(at_the_budget)
    assert "3 value(s) cannot be parsed" in warning

    one_over_the_budget = [str(i) for i in range(6)] + ["x1", "x2", "x3", "x4"]
    assert _contamination_warnings(one_over_the_budget) == []


# ---------------------------------------------------------------------------
# The recorded-casualty cap (cmp#33)
# ---------------------------------------------------------------------------


def _record(n_casualties: int) -> CleanReport:
    before = pd.Series([f"bad{i}" for i in range(n_casualties)])
    converted = pd.Series([float("nan")] * n_casualties)
    report = CleanReport()
    _record_coerced("amount", before, converted, report, CleanConfig())
    return report


def test_exactly_the_cap_is_not_reported_as_truncated():
    """Kills cmp#33: ``len(originals) <= COERCED_CELLS_CAP`` -> ``<``.

    At exactly the cap every casualty *is* recorded, so claiming the payload
    was truncated tells the reader to go looking for values that are not
    missing.
    """
    truncation_note = f"first {COERCED_CELLS_CAP} recorded"

    at_the_cap = _record(COERCED_CELLS_CAP)
    [warning] = at_the_cap.warnings
    assert truncation_note not in warning
    assert len(at_the_cap.coerced_cells["amount"]) == COERCED_CELLS_CAP

    one_over_the_cap = _record(COERCED_CELLS_CAP + 1)
    [warning] = one_over_the_cap.warnings
    assert truncation_note in warning
    assert len(one_over_the_cap.coerced_cells["amount"]) == COERCED_CELLS_CAP
    assert len(one_over_the_cap.coerced_rows["amount"]) == COERCED_CELLS_CAP + 1


# ---------------------------------------------------------------------------
# The post-semantic retry (cmp#34, cmp#35, cmp#36)
# ---------------------------------------------------------------------------

#: A number in an unrecognized grouping ("1,23,456.70"): numeric-shaped, so the
#: retry may quarantine it, but no parser here reads it.
UNREADABLE_NUMBER = "1,23,456.70"


def _refine(values, **options):
    df = pd.DataFrame({"amount": values})
    report = CleanReport()
    out = refine_numeric_after_semantic(df, ["amount"], CleanConfig(**options), report)
    return out["amount"], report


def test_the_post_semantic_retry_starts_at_four_values():
    """Kills cmp#34: ``n < 4`` -> ``<=``.

    Four non-missing values is the smallest column the retry will consider;
    the mutant silently skips it.
    """
    converted, _ = _refine(["1", "2", "3", "4"])
    assert str(converted.dtype) == "int64"
    assert converted.tolist() == [1, 2, 3, 4]

    untouched, _ = _refine(["1", "2", "3"])
    assert str(untouched.dtype) == "object"


def test_the_post_semantic_retry_converts_at_exactly_the_contamination_share():
    """Kills cmp#35: ``rate < CONTAMINATION_SHARE`` -> ``<=``.

    The retry deliberately uses the contamination boundary instead of the
    strict threshold, so a column parsing exactly ``CONTAMINATION_SHARE``
    converts and its casualties are quarantined.
    """
    converted, report = _refine(["1", "2", "3", UNREADABLE_NUMBER, "9,87,654.30"])
    assert str(converted.dtype) == "Int64"
    assert converted.tolist()[:3] == [1, 2, 3]
    assert int(converted.isna().sum()) == 2
    assert list(report.coerced_cells["amount"].values()) == [
        UNREADABLE_NUMBER,
        "9,87,654.30",
    ]

    # One value below the boundary and the column is left alone.
    untouched, _ = _refine(["1", "2", UNREADABLE_NUMBER, "9,87,654.30", "8,76,543.20"])
    assert str(untouched.dtype) == "object"


def test_the_post_semantic_retry_tolerates_exactly_the_casualty_budget():
    """Kills cmp#36: ``n_coerced > max(3, int(0.1 * n))`` -> ``>=``.

    Same budget as the contamination warning, used here to decide whether
    cells may be quarantined. Three casualties in ten is the budget exactly.
    """
    unreadable = ["1,23,456.70", "9,87,654.30", "8,76,543.20"]
    at_the_budget = [str(i) for i in range(7)] + unreadable
    converted, _ = _refine(at_the_budget)
    assert str(converted.dtype) == "Int64"
    assert int(converted.isna().sum()) == 3

    one_over_the_budget = [str(i) for i in range(6)] + [*unreadable, "7,65,432.10"]
    untouched, _ = _refine(one_over_the_budget)
    assert str(untouched.dtype) == "object"


# ---------------------------------------------------------------------------
# Recorded, not endorsed
# ---------------------------------------------------------------------------


def test_mixed_utc_offsets_are_reported_as_a_conversion_to_object():
    """Pins current behaviour found while covering bool#16 -- a defect.

    Values carrying *different* UTC offsets cannot share a datetime64 column,
    so pandas returns an object-dtype Series of Timestamps. ``_try_datetime``
    measures that object Series against the threshold, passes it, and the
    column is announced as a dtype conversion: the report says "converted to
    object" and the frame keeps an object column. A conversion that ends in
    ``object`` is not a conversion, and because the result is not
    datetime64 the ambiguity note in ``_record_coerced`` can never fire for
    such a column either.

    Nothing here argues the behaviour is right; it is pinned so that fixing
    it has to change this test deliberately.
    """
    values = [
        "2021-01-05 00:00:00+01:00",
        "2021-01-06 00:00:00+02:00",
        "2021-01-07 00:00:00+03:00",
        "2021-01-08 00:00:00+04:00",
    ]
    target, converted, n_coerced = _suggest(values)
    assert (target, n_coerced) == ("datetime", 0)
    assert str(converted.dtype) == "object"

    report = CleanReport()
    frame = fix_dtypes(pd.DataFrame({"when": values}), CleanConfig(), report)
    assert str(frame["when"].dtype) == "object"
    assert [a.description for a in report.actions] == ["converted to object"]
