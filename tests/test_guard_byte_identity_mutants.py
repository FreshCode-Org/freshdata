"""Byte-identity properties of the protected-column guard that nothing pinned.

``freshdata.guard`` is the last of five independent layers that keep an
identifier column (``"007"``, ``"0001"``) from being silently rewritten: it
compares every hard-protected column against a deep snapshot taken before
cleaning and raises :class:`ProtectedColumnError` unless the column came back
**byte-identical**. Because it is the last layer, a weakened check here is not
caught by anything downstream — the corrupted frame is simply returned.

Each test below kills one confirmed mutation survivor in ``guard.py`` and names
it, so a future weakening of the assertion has to argue with the reason. The
most serious of them is ``bool#2``: flipping ``if left_na or right_na`` to
``and`` makes the one-sided-missing branch fall through to ``==``, and a cell
type with a permissive ``__eq__`` then compares *equal to a missing value* —
the guard would pass a column it exists to reject. Note that
``tests/test_policy_guard_mutants.py`` documents this operator (under its older
number ``bool#5``) as a proven-equivalent mutant; see
:func:`test_a_missing_cell_swapped_for_a_permissive_object_is_a_violation` for
the counter-example that refutes that claim.
"""

from __future__ import annotations

from unittest.mock import ANY

import numpy as np
import pandas as pd
import pytest

from freshdata.guard import (
    ProtectedColumnError,
    _cell_equal,
    _is_missing,
    _series_identical,
    snapshot_protected,
    verify_protected,
)
from freshdata.report import CleanReport


class _MatchesAnything:
    """A cell whose ``__eq__`` is permissive, like :data:`unittest.mock.ANY`.

    Object columns may hold arbitrary values (the guard's own comment concedes
    that "exotic cell types compare however they like"): wildcard sentinels,
    matcher objects and lazy expression objects all answer ``True`` to ``==``.
    """

    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "<matches-anything>"


class _ReportWithoutAdd:
    """A report-shaped object that does not implement ``add`` (duck-typing)."""

    def __init__(self) -> None:
        self.actions: list[object] = []


# -- the row-count branch must not realign an equal-length column -----------


def test_a_reordered_protected_column_of_equal_length_is_a_violation():
    """Kills guard cmp#6 (``len(after) < len(before)`` -> ``<=``).

    The branch exists only for row-*dropping* steps; at equal length it must
    not run, because it realigns the snapshot with ``before.loc[after.index]``.
    With ``<=`` a protected column whose rows were permuted (values now sitting
    against different rows of the frame — a real corruption) is realigned back
    onto the snapshot and compares identical, so the guard reports nothing.
    """
    before = pd.Series(["007", "008", "009"], index=[0, 1, 2])
    shuffled = pd.Series(["009", "007", "008"], index=[2, 0, 1])
    assert len(shuffled) == len(before)
    assert before.loc[shuffled.index].equals(shuffled), (
        "premise: realignment would hide this reorder"
    )
    assert _series_identical(before, shuffled) is not None

    df = pd.DataFrame({"cust_id": before})
    snapshot = snapshot_protected(df, ("cust_id",))
    with pytest.raises(ProtectedColumnError, match="cust_id"):
        verify_protected(pd.DataFrame({"cust_id": shuffled}), snapshot)


def test_row_drops_with_a_unique_index_still_realign_by_label():
    """The legitimate case cmp#6 protects must keep working."""
    before = pd.Series(["007", "008", "009"], index=[0, 1, 2])
    assert _series_identical(before, before.loc[[0, 2]]) is None


# -- one-sided missingness is a difference, whatever ``==`` says ------------


def test_a_missing_cell_swapped_for_a_permissive_object_is_a_violation():
    """Kills guard bool#2 (``if left_na or right_na`` -> ``and``).

    With ``and``, a cell that is missing on one side only skips the early
    return and is compared with ``==``. A value whose ``__eq__`` returns
    ``True`` then reads as *equal to a missing value*, and the guard passes a
    protected column whose missing cell was replaced by a real value — the
    guard missing a violation, which is strictly worse than a false alarm.

    This refutes the equivalence claimed for this operator in
    ``tests/test_policy_guard_mutants.py``: a 1296-pair differential over
    ``None``/``nan``/``NaT``/``pd.NA``/``NaT64``/``Decimal('NaN')``/ints/
    floats/bools/strings/bytes/containers/ndarrays/timestamps found the
    original and the mutant agreeing everywhere *except* on permissive-``__eq__``
    cells, where they disagree in both argument orders.
    """
    assert _cell_equal(float("nan"), _MatchesAnything()) is False
    assert _cell_equal(_MatchesAnything(), None) is False
    assert _cell_equal(pd.NaT, ANY) is False  # the stdlib's own wildcard cell

    before = pd.Series([np.nan, np.nan], index=[0, 0], dtype=object)
    after = pd.Series([_MatchesAnything()], index=[0], dtype=object)
    assert _series_identical(before, after) is not None


def test_two_missing_cells_are_still_equal_after_a_row_drop():
    """The early return's real job (nan == nan) must survive the tightening."""
    before = pd.Series([np.nan, "007", None], index=[0, 0, 0], dtype=object)
    assert _series_identical(before, pd.Series([None, "007"], index=[0, 0])) is None


# -- containers are values, not missing markers -----------------------------


def test_container_cells_are_never_treated_as_missing():
    """Kills guard const_bool#4 (``_is_missing`` fallback ``False`` -> ``True``).

    ``pd.isna`` returns an *array* for a container, so ``bool()`` raises and the
    fallback decides. Calling containers "missing" makes every pair of them
    compare equal (both sides missing), so a protected object column of lists
    or dicts could be rewritten wholesale without the guard noticing.
    """
    assert _is_missing([1, 2]) is False
    assert _is_missing({"a": 1}) is False
    assert _cell_equal([1, 2], [3, 4]) is False
    assert _cell_equal({"a": 1}, {"a": 2}) is False

    before = pd.Series([[1, 2], [3, 4], [5, 6]], index=[0, 0, 0], dtype=object)
    after = pd.Series([[9, 9], [5, 6]], index=[0, 0], dtype=object)
    assert _series_identical(before, after) is not None


# -- the audit record itself ------------------------------------------------


def test_a_violation_is_recorded_as_a_zero_count_high_risk_action():
    """Kills guard const_num#6 (``count=0`` -> ``1``) and const_bool#5
    (``human_review=True`` -> ``False``).

    The violation record is user-visible audit output: ``count`` is the number
    of cells the step *changed*, and the guard changes nothing — it refuses.
    A count of 1 would read as "the guard edited one cell". ``human_review``
    is what makes the record surface in review workflows; a guard violation is
    an executor bug and must never be filed as unremarkable.
    """
    df = pd.DataFrame({"rev": ["1000", "2000"]})
    snapshot = snapshot_protected(df, ("rev",))
    broken = pd.DataFrame({"rev": ["1000", "2,000"]})
    report = CleanReport()

    with pytest.raises(ProtectedColumnError, match="rev"):
        verify_protected(broken, snapshot, report)

    (action,) = [a for a in report.actions if a.step == "guard"]
    assert action.count == 0
    assert action.human_review is True
    assert action.risk == "high"
    assert action.status == "skipped"
    assert action.column == "rev"
    assert action.metadata["protected_column_violation"]["column"] == "rev"


def test_the_clean_verification_is_recorded_as_a_zero_count_action():
    """Kills guard const_num#5 (``count=0`` -> ``1``).

    The success record states a guarantee that was *checked*; it never changed
    a cell, so a non-zero count would misreport an untouched column as edited
    (and the number would be the column count, not a cell count).
    """
    df = pd.DataFrame({"rev": ["1000", "2000"], "sku": ["007", "008"]})
    snapshot = snapshot_protected(df, ("rev", "sku"))
    report = CleanReport()

    verify_protected(df, snapshot, report)

    (action,) = [a for a in report.actions if a.step == "guard"]
    assert action.count == 0
    assert action.description == "verified 2 protected column(s) byte-identical"
    assert action.metadata["protected_columns"] == ["rev", "sku"]


def test_a_report_object_without_add_cannot_break_the_raise():
    """Kills guard bool#7 (``report is not None and hasattr(...)`` -> ``or``).

    Both halves are required: ``or`` calls ``add`` on any non-``None`` object,
    so a caller passing a report-shaped object that does not implement ``add``
    gets an ``AttributeError`` from inside the recorder instead of the
    :class:`ProtectedColumnError` that says the data was corrupted — the guard
    still stops the frame, but the diagnosis is destroyed.
    """
    df = pd.DataFrame({"rev": ["1000", "2000"]})
    snapshot = snapshot_protected(df, ("rev",))
    broken = pd.DataFrame({"rev": ["1000", "9999"]})

    with pytest.raises(ProtectedColumnError, match="rev"):
        verify_protected(broken, snapshot, _ReportWithoutAdd())
    with pytest.raises(ProtectedColumnError, match="dropped"):
        verify_protected(pd.DataFrame({"other": [1, 2]}), snapshot, _ReportWithoutAdd())
    # ``None`` stays the documented no-report path.
    with pytest.raises(ProtectedColumnError, match="rev"):
        verify_protected(broken, snapshot, None)
