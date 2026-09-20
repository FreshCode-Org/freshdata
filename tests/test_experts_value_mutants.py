"""Value-expert properties that the suite asserted nowhere.

Mutation testing ``freshdata.semantic.experts`` left four survivors in the
value-repair path. Each one is a silent correctness hazard:

* ``cmp#26`` -- the ``canon`` tie-break in :class:`CategorySynonymExpert`
  (``count > best[1]`` -> ``count >= best[1]``). It picks which surface
  spelling of a casefold key becomes the canonical one. Flipping the operator
  flips the *direction* of every tied normalization (``'usa' -> 'USA'`` becomes
  ``'USA' -> 'usa'``), which rewrites user data whenever the confidence gate
  lets the proposal auto-apply. Nothing asserted the direction.

* ``cmp#27`` -- ``allowed[key] != value`` -> ``== value``. The inverted form
  emits a no-op "repair" for every value that already *is* its allowed value,
  and stops repairing the values that actually deviate. Nothing asserted that
  an already-conforming value is left alone.

* ``const_bool#24`` -- :class:`_DateResolution` is ``@dataclass(frozen=True)``.
  It is the audit record for one date decision (value, confidence, risk,
  detail); immutability is what keeps a resolution from being edited after the
  policy gate has scored it. Nothing asserted the frozen-ness.

* ``const_bool#31`` -- ``_value_counts`` uses ``value_counts(dropna=True)``.
  Counting missing cells as if they were a distinct category would inflate a
  column's apparent dominant value and feed NaN into experts that assume
  ``isinstance(v, str)``. Nothing asserted that nulls are excluded.

The tie-break tests double as characterization of a suspected defect: the
winner of a tie is decided by *row order*, not by the data. See
``test_tied_spellings_make_the_first_appearing_row_win_the_canonical_form``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.semantic.experts import (
    CategorySynonymExpert,
    _DateResolution,
    _value_counts,
)
from freshdata.semantic.types import SemanticColumnInfo


def _info(**kw) -> SemanticColumnInfo:
    base = {
        "name": "country",
        "role": "categorical",
        "n_nonnull": 6,
        "nunique": 2,
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


def _pairs(proposals) -> list[tuple[object, object]]:
    return [(p.raw_value, p.proposed_value) for p in proposals]


# --- cmp#26: the canonical-spelling tie-break ------------------------------- #
def test_the_more_frequent_spelling_wins_the_canonical_form_outright():
    """A strict majority must decide the canonical form, not the tie-break.

    Guards the ``count > best[1]`` comparison from degenerating into an
    order-only rule: ``'USA'`` appears once and ``'usa'`` three times, so the
    proposal must run towards ``'usa'`` even though ``'USA'`` is the first row.
    """
    series = pd.Series(["USA", "usa", "usa", "usa"])
    proposals = CategorySynonymExpert().propose(series, _info(n_nonnull=4))
    assert _pairs(proposals) == [("USA", "usa")]


def test_tied_spellings_make_the_first_appearing_row_win_the_canonical_form():
    """Characterizes the tie-break: on an exact tie the first row wins.

    ``count > best[1]`` keeps the first spelling that ``value_counts`` yields,
    and pandas breaks a count tie by order of first appearance. So a 3-vs-3
    ``'USA'``/``'usa'`` column normalizes towards whichever spelling the file
    happened to list first.

    This is a **suspected defect**, not a property worth having: the two frames
    below hold the same multiset of values and differ only in row order, yet
    the expert proposes opposite repairs. The test pins the behaviour so the
    mutant (``count >= best[1]``, which makes the *last* spelling win) dies and
    so any deliberate fix has to come past this assertion.
    """
    expert = CategorySynonymExpert()
    upper_first = pd.Series(["USA", "USA", "USA", "usa", "usa", "usa"])
    lower_first = pd.Series(["usa", "usa", "usa", "USA", "USA", "USA"])

    assert _pairs(expert.propose(upper_first, _info())) == [("usa", "USA")]
    assert _pairs(expert.propose(lower_first, _info())) == [("USA", "usa")]


def test_row_order_alone_flips_an_auto_applied_tied_normalization():
    """The tie-break reaches user data once the confidence gate opens.

    The tied case scores 0.93/medium, below the 0.95 default, so by default it
    only ever surfaces as a *suggestion* whose direction flips with row order.
    Lowering ``semantic_auto_threshold`` to 0.90 -- a documented knob -- makes
    the same proposal auto-apply, and two row permutations of one column then
    produce two different cleaned frames. Recorded as the public-API evidence
    for the defect described above.
    """
    upper_first = pd.DataFrame({"country": ["USA", "USA", "USA", "usa", "usa", "usa"]})
    lower_first = pd.DataFrame({"country": ["usa", "usa", "usa", "USA", "USA", "USA"]})
    kwargs = {
        "semantic_mode": "auto",
        "semantic_auto_threshold": 0.90,
        "return_report": True,
        "verbose": False,
    }

    out_upper, report_upper = fd.clean(upper_first, **kwargs)
    out_lower, report_lower = fd.clean(lower_first, **kwargs)

    assert list(out_upper["country"]) == ["USA"] * 6
    assert list(out_lower["country"]) == ["usa"] * 6
    statuses = {a.status for r in (report_upper, report_lower) for a in r if a.step == "semantic"}
    assert statuses == {"automatic"}


def test_a_tied_normalization_is_only_suggested_under_the_default_threshold():
    """At stock settings the tie never rewrites data -- it is reported instead."""
    df = pd.DataFrame({"country": ["USA", "USA", "USA", "usa", "usa", "usa"]})
    out, report = fd.clean(df, semantic_mode="auto", return_report=True, verbose=False)

    assert list(out["country"]) == ["USA", "USA", "USA", "usa", "usa", "usa"]
    semantic = [a for a in report if a.step == "semantic"]
    assert [a.status for a in semantic] == ["suggested"]
    assert semantic[0].metadata["raw_value"] == "usa"
    assert semantic[0].metadata["proposed_value"] == "USA"


# --- cmp#27: the allowed-values check --------------------------------------- #
def test_a_value_that_already_equals_its_allowed_value_is_left_alone():
    """No proposal may be emitted for a value already in canonical form.

    ``allowed[key] != value`` is what suppresses the no-op; inverting it to
    ``== value`` would emit ``'Red' -> 'Red'`` repairs for clean data.
    """
    series = pd.Series(["Red", "Red", "Blue", "Blue"])
    info = _info(name="colour", n_nonnull=4, allowed_values=("Red", "Blue"))
    assert CategorySynonymExpert().propose(series, info) == []


def test_a_deviating_value_is_mapped_onto_its_allowed_value():
    """The other half of the same branch: deviations *are* repaired, at 0.96."""
    series = pd.Series(["red", "red", "Blue", "Blue"])
    info = _info(name="colour", n_nonnull=4, allowed_values=("Red", "Blue"))
    proposals = CategorySynonymExpert().propose(series, info)

    assert _pairs(proposals) == [("red", "Red")]
    assert proposals[0].confidence == pytest.approx(0.96)
    assert any(e.kind == "context_hint" for e in proposals[0].evidence)


def test_an_allowed_values_column_is_routed_away_from_the_synonym_expert():
    """``applies`` hands explicit reference lists to ReferenceExpert instead.

    Documents why the two tests above must call ``propose`` directly: the
    allowed-values branch is unreachable through this expert's own gate.
    """
    expert = CategorySynonymExpert()
    assert expert.applies(_info(allowed_values=("Red", "Blue"))) is False
    assert expert.applies(_info()) is True


# --- const_bool#24: the date-resolution audit record ------------------------ #
def test_a_date_resolution_record_cannot_be_edited_after_construction():
    """``_DateResolution`` is the audit trail for one date decision.

    If it were mutable, a downstream caller could lower a risk or raise a
    confidence after the fact and the report would no longer describe what was
    actually decided.
    """
    resolution = _DateResolution(None, 0.50, "high", "no reference_date supplied")

    assert dataclasses.is_dataclass(resolution)
    assert dataclasses.fields(_DateResolution)  # guards against an empty record
    for field, new_value in (
        ("value", pd.Timestamp("2024-01-01")),
        ("confidence", 0.99),
        ("risk", "low"),
        ("detail", "rewritten"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(resolution, field, new_value)

    assert resolution.value is None
    assert resolution.confidence == pytest.approx(0.50)
    assert resolution.risk == "high"


# --- const_bool#31: missing values are not a category ----------------------- #
def test_value_counts_excludes_missing_cells_from_the_distinct_table():
    """Nulls must never become a counted category.

    ``dropna=False`` would give NaN its own row, so a column that is mostly
    empty would look like it has a dominant ``NaN`` value -- and experts that
    iterate the table assuming ``isinstance(raw, str)`` would silently skip the
    real data while the share-based evidence was computed against a phantom.
    """
    series = pd.Series(["a", "a", "b", None, np.nan])
    counts = _value_counts(series)

    assert list(counts.index) == ["a", "b"]
    assert counts.to_dict() == {"a": 2, "b": 1}
    assert int(counts.sum()) == 3
    assert not any(pd.isna(v) for v in counts.index)


def test_value_counts_drops_missing_cells_even_when_they_are_the_majority():
    """The all-but-one-null case: the table must not be led by NaN."""
    series = pd.Series([1.0, 1.0, np.nan, np.nan, np.nan])
    counts = _value_counts(series)

    assert len(counts) == 1
    assert counts.index[0] == 1.0
    assert int(counts.iloc[0]) == 2


def test_value_counts_prefers_a_precomputed_native_table_over_rescanning():
    """The attrs fast path short-circuits before the pandas call it mutates."""
    precomputed = pd.Series({"x": 7, "y": 2}, dtype="int64")
    series = pd.Series(["x", "y", None])
    series.attrs["fd_value_counts"] = precomputed

    assert _value_counts(series) is precomputed
