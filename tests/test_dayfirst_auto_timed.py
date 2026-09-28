"""A time of day must not turn "refuse to guess" into "guess month-first".

``dayfirst="auto"`` (the default) promises that day/month-ambiguous values
are quarantined for review, never read in an inferred order (``config.py``).
The detector matched only a bare ``DD/MM/YYYY``: ``_AMBIGUOUS_DATE`` was
anchored at ``$`` straight after the year, so ``"05/01/2021 00:00"`` did not
match, was not quarantined, and reached the parser with ``dayfirst=False`` --
read as 1 May, silently, reported as a plain ``converted to datetime64``.
Appending a time of day flipped the library's behaviour on the same date.

A first fix widened the anchor to allow a list of time suffixes. Six more
shapes still slipped through it (a comma before the time, a weekday prefix,
a trailing period, ``10h``, ``at 10:00``), because any list of decorations is
incomplete. The detector now searches for the date token anywhere in the
value, so it fails closed: the worst case is over-quarantine, which is
reviewable, never a silent guess.

The finance domain validator already treats these values as ambiguous (its
own pattern ends ``(?!\\d)``); this brings the core dtype step in line.
"""
from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.steps.dtypes import _ambiguous_day_month

#: Every value has both tokens <= 12 and day != month: genuinely ambiguous.
DATES = ["05/01/2021", "06/02/2021", "07/03/2021", "08/04/2021"]
#: Decorations pandas parses around a date. The first nine were the obvious
#: ones; the last six defeated a first fix that listed allowed suffixes
#: instead of searching for the date token -- which is why the detector now
#: searches. ``{}`` is where the date goes.
SHAPES = [
    "{} 00:00", "{} 00:00:00", "{} 00:00:00.123", "{} 3:30 PM", "{} 3:30PM",
    "{} 00:00+01:00", "{} 00:00 +0100", "{} 00:00Z", "{}T00:00",
    "{}, 10:00", "Mon {}", "Mon {} 10:00", "{}.", "{} 10h", "{} at 10:00",
]


def _clean(values, **options):
    return fd.clean(pd.DataFrame({"t": values * 6}), verbose=False, **options)


def _is_month_first_guess(col: pd.Series) -> bool:
    first = col.dropna().iloc[0] if col.notna().any() else None
    return hasattr(first, "month") and first.month == 5 and first.day == 1


@pytest.mark.parametrize("shape", SHAPES)
def test_decoration_around_a_date_does_not_defeat_the_auto_quarantine(shape):
    """A decorated column must be treated exactly as the bare one is: not guessed."""
    bare = _clean(DATES).data["t"]
    decorated = _clean([shape.format(d) for d in DATES]).data["t"]

    assert not _is_month_first_guess(bare), "control: bare dates are never guessed"
    assert not _is_month_first_guess(decorated), (
        f"{shape.format(DATES[0])!r} was silently read as 1 May under dayfirst='auto'"
    )
    assert str(decorated.dtype) == str(bare.dtype)


def test_an_unambiguous_timed_value_still_parses():
    """13 cannot be a month, so ``13/01/2021 10:00`` is certain: 13 January."""
    values = ["13/01/2021 10:00", "14/02/2021 11:00", "15/03/2021 12:00", "16/04/2021 13:00"]
    col = _clean(values).data["t"]
    assert str(col.dtype).startswith("datetime64")
    assert col.iloc[0] == pd.Timestamp(2021, 1, 13, 10, 0)


@pytest.mark.parametrize("dayfirst,month,day", [(True, 1, 5), (False, 5, 1)])
def test_an_explicit_dayfirst_still_resolves_timed_values(dayfirst, month, day):
    """The quarantine is for "auto" only; an explicit order is honoured."""
    col = _clean([d + " 00:00" for d in DATES], dayfirst=dayfirst).data["t"]
    assert str(col.dtype).startswith("datetime64")
    assert (col.iloc[0].month, col.iloc[0].day) == (month, day)


def test_in_a_mixed_column_only_the_ambiguous_timed_values_are_quarantined():
    """One ``13/01`` must not decide how ``05/01`` is read -- per value, not per column."""
    certain = ["13/01/2021 10:00", "14/02/2021 11:00", "15/03/2021 12:00"]
    ambiguous = ["05/01/2021 10:00"]
    result = fd.clean(pd.DataFrame({"t": certain * 10 + ambiguous}), verbose=False)
    col = result.data["t"]
    report = result.report()

    assert str(col.dtype).startswith("datetime64")
    assert col.iloc[0] == pd.Timestamp(2021, 1, 13, 10, 0)
    assert pd.isna(col.iloc[-1]), "the ambiguous value must be held, not read as 1 May"
    assert "05/01/2021 10:00" in report.coerced_cells.get("t", {}).values()
    notes = [a.description for a in report.actions if a.column == "t"]
    assert any("day/month-ambiguous" in n for n in notes), notes


def test_a_range_is_ambiguous_when_its_second_date_is():
    """A certain first date must not vouch for an ambiguous second one."""
    assert _ambiguous_day_month("13/01/2021 - 05/01/2021")
    assert not _ambiguous_day_month("13/01/2021 - 25/01/2021")


@pytest.mark.parametrize("value", [
    "2021-05-01", "2021-05-01 10:00:00", "2021-05-01T10:00:00+01:00", "20210501",
    "105/01/2021", "05/01/2021123", "13/01/2021 00:00", "01/01/2021 00:00",
])
def test_the_search_does_not_invent_ambiguity(value):
    """ISO, compact and digit-bounded lookalikes, and genuinely certain dates."""
    assert not _ambiguous_day_month(value)
