"""A repeated index label must not change what cleaning does.

Two steps selected rows by index label -- the formatted-number rescue in
``steps/dtypes.py`` and the coercion-casualty scan in ``engine/missing.py``.
``.loc`` expands a repeated label to every row carrying it, so both raised on
ordinary input. The pipeline now runs on a positional index and restores the
caller's labels, the same treatment ``validate_domain`` already applied.

Every test here is a *pair*: identical data under a unique index and under a
duplicated one. The assertion is that the two agree, so the tests keep their
meaning if the implementation changes again.
"""
from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd

#: The formatted-number rescue is only reached above the sampling threshold,
#: so a smaller frame would pass whether or not the bug were present.
ROWS = 40


def _pair(data: dict[str, list]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same data twice: unique labels, then labels repeated once each."""
    unique = pd.DataFrame(data, index=[f"u{i}" for i in range(ROWS)])
    repeated = pd.DataFrame(data, index=[f"r{i}" for i in range(ROWS // 2)] * 2)
    return unique, repeated


MONEY = {"v": [f"{i}.00" for i in range(ROWS - 1)] + ["$1,234.56"]}
#: The last value must fail to parse on *every* supported pandas. "5 March 2021"
#: was used first and is a casualty only on pandas 2, which infers one format
#: for the column; pandas 1.x infers per value and parses it, so there was no
#: casualty and the casualty-scan tests passed without reaching the code.
DATES = {"d": [f"2021-01-{i % 28 + 1:02d}" for i in range(ROWS - 1)] + ["not a date"]}


def test_a_repeated_label_does_not_break_the_formatted_number_rescue():
    """Regression: ``ValueError: cannot set using a list-like indexer ...``.

    ``_rescue_formatted`` assigned with ``parsed.loc[rescued.index]``. With
    ``r0`` appearing twice the left-hand side was longer than the right and
    pandas raised, so ``fd.clean(df)`` failed outright on default settings.
    """
    unique, repeated = _pair(MONEY)
    expected = fd.clean(unique)
    got = fd.clean(repeated)
    assert str(got.data["v"].dtype) == str(expected.data["v"].dtype) == "float64"
    assert got.data["v"].iloc[-1] == expected.data["v"].iloc[-1] == 1234.56


def test_a_repeated_label_does_not_break_the_coercion_casualty_scan():
    """Regression: ``IndexError: boolean index did not match indexed array``.

    ``_quarantined_rows`` narrowed with ``df[col].loc[rows].isna()``. The
    repeated label made ``.loc`` return two rows for a one-row mask.
    """
    unique, repeated = _pair(DATES)
    expected = fd.clean(unique)
    got = fd.clean(repeated)
    assert str(got.data["d"].dtype) == str(expected.data["d"].dtype)
    assert got.data["d"].isna().sum() == expected.data["d"].isna().sum()


def test_suggest_plan_also_survives_a_repeated_label():
    """``suggest_plan`` reaches the same rescue, so it crashed too."""
    unique, repeated = _pair(MONEY)
    assert fd.suggest_plan(repeated).column_plans == fd.suggest_plan(unique).column_plans


def test_the_callers_index_comes_back_unchanged():
    """The pipeline works positionally, but that must not leak to the caller."""
    _, repeated = _pair(MONEY)
    labels = list(repeated.index)
    result = fd.clean(repeated)
    assert list(result.data.index) == labels


def test_the_input_frame_is_not_mutated():
    """``preserve_original=True`` still holds on the positional path."""
    _, repeated = _pair(MONEY)
    before = repeated.copy(deep=True)
    fd.clean(repeated)
    pd.testing.assert_frame_equal(repeated, before)


def test_report_row_keys_are_the_callers_labels_not_positions():
    """A row key the caller cannot look up would be worse than no key."""
    _, repeated = _pair(DATES)
    result = fd.clean(repeated)
    report = result.report()
    rows = report.coerced_rows.get("d", ())
    assert rows, "the unparseable date should be recorded as a casualty"
    assert set(rows) <= set(repeated.index)
    assert set(report.coerced_cells.get("d", {})) <= set(repeated.index)


def test_dropped_rows_keep_their_original_labels():
    """Row-removing steps run positionally; survivors map back one-to-one."""
    frame = pd.DataFrame(
        {"a": [1, 1, 2, 3] * 10, "b": ["x", "x", "y", "z"] * 10},
        index=[f"k{i % 20}" for i in range(ROWS)],
    )
    result = fd.clean(frame, drop_duplicates=True)
    assert len(result.data) < len(frame)
    assert set(result.data.index) <= set(frame.index)


def test_reset_index_is_unaffected():
    """An explicit ``reset_index`` still yields positions, as on a unique index."""
    _, repeated = _pair(MONEY)
    result = fd.clean(repeated, reset_index=True)
    assert isinstance(result.data.index, pd.RangeIndex)


def test_a_quarantined_cell_is_recorded_once_per_row_not_per_label():
    """The casualty scan is keyed by label; only one row is actually a casualty.

    Row 0 and row 20 share the label ``r0``, and only position 39 fails to
    parse. The scan raised ``IndexError`` here before the fix. What it must
    now produce is the same single casualty the unique-index control produces
    -- not two, and not a different row.
    """
    data = {"d": [f"2021-01-{i % 28 + 1:02d}" for i in range(ROWS)]}
    data["d"][ROWS - 1] = "not a date"          # the casualty, at position 39
    unique, repeated = _pair(data)
    expected = fd.clean(unique)
    got = fd.clean(repeated)

    assert expected.data["d"].isna().sum() == 1, "control: exactly one casualty"
    assert got.data["d"].isna().sum() == 1, "the label twin must survive"
    assert got.data["d"].tolist() == expected.data["d"].tolist()


@pytest.mark.parametrize("data", [MONEY, DATES], ids=["money", "dates"])
def test_a_unique_index_takes_the_unchanged_path(data):
    """The fix must be inert when the index is already unique."""
    unique, _ = _pair(data)
    before = unique.copy(deep=True)
    result = fd.clean(unique)
    assert list(result.data.index) == list(unique.index)
    pd.testing.assert_frame_equal(unique, before)
