"""The audit record of a datetime parse that pandas can only keep as ``object``.

Datetime strings carrying *different* UTC offsets (or mixing offset-bearing
and offset-free values) have no common ``datetime64`` dtype. pandas parses
every cell to its own timestamp, each keeping its own offset, and returns them
in an ``object`` column. That parse is faithful and lossless, and ``fix_dtypes``
delivers it -- but it used to record the action as ``"converted to object"``,
which reads as though nothing was converted (FD2-015).

These tests pin the record, not the data: the description now says the
values were parsed to timestamps and why they stay ``object``, while the cell
values, the dtype, the risk and the confidence stay exactly what they were.
Every other column keeps its ``"converted to <dtype>"`` description.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.report import CleanReport
from freshdata.steps.dtypes import (
    fix_dtypes,
    refine_numeric_after_semantic,
    suggest_conversion,
)

MIXED_OFFSETS = [
    "2021-01-05 00:00:00+01:00",
    "2021-01-06 00:00:00+02:00",
    "2021-01-07 00:00:00+03:00",
    "2021-01-08 00:00:00+04:00",
]

#: Enough distinct values that one junk cell stays under the 95% threshold.
MIXED_OFFSETS_WITH_JUNK = [
    f"2021-01-{d:02d} 00:00:00+0{d % 5}:00" for d in range(1, 26)
] + ["garbage"]

NAIVE_AND_AWARE = [
    "2021-01-05 00:00:00",
    "2021-01-06 00:00:00+01:00",
    "2021-01-07 00:00:00",
    "2021-01-08 00:00:00",
]

DIFFERENT_OFFSETS = (
    "parsed to timestamps; kept as object because the values carry different "
    "UTC offsets, so no single datetime64 dtype can hold them"
)
NAIVE_MIXED_WITH_AWARE = (
    "parsed to timestamps; kept as object because the values mix "
    "timezone-aware and timezone-naive timestamps, so no single datetime64 "
    "dtype can hold them"
)
ONE_CASUALTY = " (1 unparseable value(s) set to missing)"


def _fix(values):
    """``fix_dtypes`` on one column named ``t``: ``(frame, report)``."""
    report = CleanReport()
    frame = fix_dtypes(pd.DataFrame({"t": values}), CleanConfig(), report)
    return frame, report


def _only_action(report: CleanReport):
    [action] = [a for a in report.actions if a.step == "fix_dtypes"]
    return action


# ---------------------------------------------------------------------------
# The record says what happened
# ---------------------------------------------------------------------------


def test_mixed_offsets_are_recorded_as_parsed_timestamps_kept_as_object():
    frame, report = _fix(MIXED_OFFSETS)
    assert str(frame["t"].dtype) == "object"
    action = _only_action(report)
    assert action.description == DIFFERENT_OFFSETS
    assert action.column == "t"
    assert action.count == 4


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(
            ["2021-01-05T10:00:00+01:00", "2021-03-05T10:00:00+01:00",
             "2021-07-05T10:00:00+02:00", "2021-08-05T10:00:00+02:00"],
            id="dst-changeover",
        ),
        pytest.param(
            ["2021-01-05T10:00:00Z", "2021-01-06T10:00:00Z",
             "2021-01-07T10:00:00+01:00", "2021-01-08T10:00:00Z"],
            id="z-and-offset",
        ),
        pytest.param(
            ["2021-01-05 00:00:00.123456+01:00", "2021-01-06 00:00:00.5+02:00",
             "2021-01-07 00:00:00+03:00", "2021-01-08 00:00:00+04:00"],
            id="fractional-seconds",
        ),
    ],
)
def test_every_different_offset_shape_gets_the_same_record(values):
    frame, report = _fix(values)
    assert str(frame["t"].dtype) == "object"
    assert _only_action(report).description == DIFFERENT_OFFSETS


def test_mixed_offsets_in_a_string_dtype_column_get_the_same_record():
    frame, report = _fix(pd.Series(MIXED_OFFSETS, dtype="string"))
    assert str(frame["t"].dtype) == "object"
    assert _only_action(report).description == DIFFERENT_OFFSETS


def test_naive_mixed_with_aware_values_names_that_reason():
    """Same mechanism, different cause: some values have no offset at all."""
    frame, report = _fix(NAIVE_AND_AWARE)
    assert str(frame["t"].dtype) == "object"
    assert _only_action(report).description == NAIVE_MIXED_WITH_AWARE


def test_the_public_clean_report_carries_the_record():
    out, report = fd.clean(pd.DataFrame({"t": MIXED_OFFSETS}),
                           return_report=True, verbose=False)
    assert str(out["t"].dtype) == "object"
    assert [a.description for a in report.actions if a.step == "fix_dtypes"] == [
        DIFFERENT_OFFSETS
    ]


def test_risk_and_confidence_are_not_changed_by_the_new_record():
    action = _only_action(_fix(MIXED_OFFSETS)[1])
    assert action.description == DIFFERENT_OFFSETS
    assert (action.risk, action.confidence) == ("low", 1.0)


def test_description_helper_falls_back_to_a_generic_reason():
    """No probed input reaches this branch; it keeps the record truthful anyway."""
    from freshdata.steps.dtypes import _describe_conversion  # noqa: PLC0415

    dates = pd.Series([dt.date(2021, 1, 5), dt.date(2021, 1, 6)], dtype=object)
    assert _describe_conversion("datetime", dates) == (
        "parsed to timestamps; kept as object because no single datetime64 "
        "dtype can hold them"
    )
    # A non-datetime target keeps the plain dtype description.
    assert _describe_conversion("numeric", dates) == "converted to object"


# ---------------------------------------------------------------------------
# (c) casualties keep their suffix and their warning
# ---------------------------------------------------------------------------


def test_unparseable_suffix_is_kept_after_the_new_description():
    frame, report = _fix(MIXED_OFFSETS_WITH_JUNK)
    assert str(frame["t"].dtype) == "object"
    assert _only_action(report).description == DIFFERENT_OFFSETS + ONE_CASUALTY
    assert report.coerced_cells == {"t": {25: "garbage"}}
    [warning] = report.warnings
    assert warning.startswith(
        "column 't': 1 value(s) could not be parsed as timestamps and were set "
        "to missing — e.g. 'garbage' (row 25)."
    )


def test_naive_and_aware_casualty_keeps_its_suffix():
    values = [
        f"2021-02-{d:02d} 00:00:00" + ("+01:00" if d % 2 else "") for d in range(1, 26)
    ] + ["garbage"]
    _, report = _fix(values)
    assert _only_action(report).description == NAIVE_MIXED_WITH_AWARE + ONE_CASUALTY


# ---------------------------------------------------------------------------
# (a) controls: every datetime64 conversion keeps its description exactly
# ---------------------------------------------------------------------------


def test_single_offset_column_keeps_its_description():
    values = [v[:-6] + "+01:00" for v in MIXED_OFFSETS]
    frame, report = _fix(values)
    dtype = frame["t"].dtype
    assert isinstance(dtype, pd.DatetimeTZDtype)
    # ``UTC+01:00`` on pandas 2, ``pytz.FixedOffset(60)`` on pandas 1.x.
    assert _only_action(report).description == f"converted to {dtype}"


def test_naive_column_keeps_its_description():
    frame, report = _fix([v[:10] for v in MIXED_OFFSETS])
    assert str(frame["t"].dtype) == "datetime64[ns]"
    assert _only_action(report).description == "converted to datetime64[ns]"


def test_naive_column_with_a_casualty_keeps_its_description_and_warning():
    values = [f"2021-01-{d:02d}" for d in range(1, 26)] + ["garbage"]
    frame, report = _fix(values)
    assert str(frame["t"].dtype) == "datetime64[ns]"
    assert _only_action(report).description == (
        "converted to datetime64[ns]" + ONE_CASUALTY
    )
    [warning] = report.warnings
    assert "could not be parsed as datetime64[ns] and were set to missing" in warning


def test_single_offset_column_with_a_casualty_keeps_its_description():
    values = [f"2021-01-{d:02d} 00:00:00+01:00" for d in range(1, 26)] + ["garbage"]
    frame, report = _fix(values)
    dtype = frame["t"].dtype
    assert isinstance(dtype, pd.DatetimeTZDtype)
    assert _only_action(report).description == f"converted to {dtype}" + ONE_CASUALTY


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1", "2", "3", "4.5"], "converted to float64"),
        (["yes", "no", "yes", "no"], "converted to bool"),
    ],
)
def test_non_datetime_conversions_keep_their_description(values, expected):
    assert _only_action(_fix(values)[1]).description == expected


# ---------------------------------------------------------------------------
# (b) the data is exactly what it was before the record changed
# ---------------------------------------------------------------------------


def test_mixed_offset_cells_are_unchanged():
    """Pinned against main's output: one timestamp per cell, own offset kept."""
    frame, _ = _fix(MIXED_OFFSETS)
    column = frame["t"]
    assert str(column.dtype) == "object"
    assert [cell.isoformat() for cell in column] == [
        "2021-01-05T00:00:00+01:00",
        "2021-01-06T00:00:00+02:00",
        "2021-01-07T00:00:00+03:00",
        "2021-01-08T00:00:00+04:00",
    ]
    assert all(isinstance(cell, dt.datetime) for cell in column)
    # Exactly the series ``suggest_conversion`` produced: the record is the
    # only thing fix_dtypes adds.
    target, converted, _ = suggest_conversion(pd.Series(MIXED_OFFSETS), CleanConfig())
    assert target == "datetime"
    assert column.equals(converted)
    assert [type(c) for c in column] == [type(c) for c in converted]


def test_mixed_offset_cells_with_a_casualty_are_unchanged():
    frame, _ = _fix(MIXED_OFFSETS_WITH_JUNK)
    column = frame["t"]
    assert str(column.dtype) == "object"
    assert [cell.isoformat() for cell in column.iloc[:-1]] == [
        f"2021-01-{d:02d}T00:00:00+0{d % 5}:00" for d in range(1, 26)
    ]
    assert column.iloc[-1] is pd.NaT


# ---------------------------------------------------------------------------
# The post-semantic numeric retry cannot produce this situation
# ---------------------------------------------------------------------------


def test_semantic_numeric_retry_leaves_parsed_timestamps_alone():
    """``refine_numeric_after_semantic`` only ever yields int64/Int64/float64.

    Even handed the object column of timestamps directly, nothing parses as a
    number, so it declines and records nothing -- its own "converted to ...
    after semantic repair" description never sees an object result.
    """
    frame, _ = _fix(MIXED_OFFSETS)
    report = CleanReport()
    out = refine_numeric_after_semantic(frame.copy(), ["t"], CleanConfig(), report)
    assert report.actions == []
    assert out["t"].equals(frame["t"])
