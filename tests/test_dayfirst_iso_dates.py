"""``dayfirst=True`` must not reinterpret unambiguous ISO-8601 dates.

``dayfirst`` exists to resolve *ambiguous* short dates such as ``05/12/2021``.
An ISO-8601 date is ``YYYY-MM-DD`` by definition, so there is nothing for it to
resolve. Before this fix ``fd.clean(df, dayfirst=True)`` silently returned
``2021-05-01`` for the input ``2021-01-05``.

The cause was pandas 2's format inference, which picks ONE format for a whole
column from its first value and, with ``dayfirst=True``, reads an ISO date as
``%Y-%d-%m``. pandas 1.x infers per value and was never affected.

The corruption was data-dependent, which is what made it dangerous: it is
silent only while every day is <= 12. A day >= 13 makes the guessed format fail
on that value, dropping the parse share below ``datetime_threshold`` so the
mixed-format retry replaces the column with the correct reading. The same
column therefore read correctly or incorrectly depending on values it happened
to contain, or on an unrelated threshold.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd

ISO = ["2021-01-05", "2021-02-11", "2021-03-09"]


def _dates(frame, column="v"):
    return [str(v.date()) for v in frame[column]]


def test_iso_dates_are_not_reinterpreted_when_dayfirst_is_true():
    """The regression. Before the fix this returned 2021-05-01 and friends."""
    out = fd.clean(pd.DataFrame({"v": ISO}), dayfirst=True, verbose=False)
    assert _dates(out) == ISO


@pytest.mark.parametrize("dayfirst", [True, False, "auto", None])
def test_iso_dates_read_the_same_whatever_dayfirst_says(dayfirst):
    """``dayfirst`` is not a question ISO-8601 input can answer differently."""
    kwargs = {} if dayfirst is None else {"dayfirst": dayfirst}
    out = fd.clean(pd.DataFrame({"v": ISO}), verbose=False, **kwargs)
    assert _dates(out) == ISO


def test_dayfirst_still_does_its_actual_job_on_ambiguous_slash_dates():
    """The fix must not disarm ``dayfirst`` where it is genuinely needed."""
    ambiguous = ["05/12/2021", "06/11/2021"]
    day_first = fd.clean(pd.DataFrame({"v": ambiguous}), dayfirst=True, verbose=False)
    assert _dates(day_first) == ["2021-12-05", "2021-11-06"]

    month_first = fd.clean(pd.DataFrame({"v": ambiguous}), dayfirst=False, verbose=False)
    assert _dates(month_first) == ["2021-05-12", "2021-06-11"]


def test_the_reading_no_longer_depends_on_whether_a_day_exceeds_twelve():
    """The data-dependence that made the corruption hard to notice.

    Nineteen consecutive January dates. Before the fix these read correctly at
    the default threshold but became 2021-01-01, 2021-02-01, 2021-03-01 ... at
    ``datetime_threshold=0.5``, losing seven values to ``NaT``.
    """
    dates = [f"2021-01-{d:02d}" for d in range(1, 20)]

    default = fd.clean(pd.DataFrame({"v": dates}), dayfirst=True, verbose=False)
    lowered = fd.clean(
        pd.DataFrame({"v": dates}), dayfirst=True, datetime_threshold=0.5, verbose=False
    )

    assert _dates(default) == dates
    assert _dates(lowered) == dates
    assert not default["v"].isna().any()
    assert not lowered["v"].isna().any()


def test_a_column_mixing_iso_and_ambiguous_forms_reads_each_by_its_own_shape():
    mixed = ["2021-01-05", "06/11/2021", "2021-03-09"]
    out = fd.clean(pd.DataFrame({"v": mixed}), dayfirst=True, verbose=False)
    assert _dates(out) == ["2021-01-05", "2021-11-06", "2021-03-09"]
