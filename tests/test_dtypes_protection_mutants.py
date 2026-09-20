"""Protection and text-shape boundaries in the dtype step that nothing else pins.

Three decision points in :mod:`freshdata.steps.dtypes` are reachable but
untested elsewhere in the suite:

* ``refine_numeric_after_semantic`` — the post-semantic numeric retry. It owns
  a second copy of the dtype-level leading-zero veto and its own dtype gate,
  and no test calls it directly (the cleaner only reaches it when a *numeric*
  semantic repair was applied earlier in the same run).
* ``_looks_dateish`` — the "nothing usable to look at" arm of the cheap date
  pre-screen, which must answer *no*, not *yes*.
* ``_record_coerced`` — the day/month-ambiguity note, for a coercion casualty
  that is not text at all.

The last test proves by execution that ``_fits_int64``'s exception guard
cannot be reached from ``_finalize_numeric``.

Everything here asserts current behaviour; nothing in ``src/`` is modified.
"""

import datetime as dt
import inspect
import sys
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd

from freshdata._util import PANDAS_MAJOR
from freshdata.config import CleanConfig
from freshdata.report import CleanReport
from freshdata.steps import dtypes as dtypes_mod
from freshdata.steps.dtypes import (
    _INT64_MAX,
    _INT64_MIN,
    _finalize_numeric,
    _fits_int64,
    _looks_dateish,
    fix_dtypes,
    refine_numeric_after_semantic,
    suggest_conversion,
)


def _refine(values, **options):
    """Run the post-semantic numeric retry on a one-column frame."""
    df = pd.DataFrame({"v": values})
    report = CleanReport()
    out = refine_numeric_after_semantic(df, ["v"], CleanConfig(**options), report)
    return out["v"], report


# --------------------------------------------------------------------------
# refine_numeric_after_semantic: which columns it is allowed to touch
# --------------------------------------------------------------------------


def test_refine_numeric_retry_accepts_a_nullable_string_column():
    """The retry's dtype gate is object *or* StringDtype, never both at once.

    A column the semantic stage repaired can be ``string`` dtype rather than
    ``object``; requiring both dtypes would silently skip every such column
    (no ``StringDtype`` is also an object dtype).
    """
    column, report = _refine(pd.Series(["1", "2", "3", "4", "5", "6"], dtype="string"))
    assert column.dtype == "int64"
    assert column.tolist() == [1, 2, 3, 4, 5, 6]
    assert [a.step for a in report.actions] == ["fix_dtypes"]


def test_refine_numeric_retry_accepts_a_plain_object_column():
    """The same retry on the object-dtype half of that gate."""
    column, report = _refine(pd.Series(["1", "2", "3", "4", "5", "6"], dtype=object))
    assert column.dtype == "int64"
    assert column.tolist() == [1, 2, 3, 4, 5, 6]
    assert "after semantic repair" in report.actions[0].description


# --------------------------------------------------------------------------
# refine_numeric_after_semantic: the leading-zero veto it carries itself
# --------------------------------------------------------------------------


ZIPS = ["01234", "02115", "03301", "04401", "05501"]


def test_refine_numeric_retry_keeps_zero_padded_ids_as_text():
    """The retry re-applies the dtype-level leading-zero veto.

    Without it a zero-padded identifier column that the semantic stage touched
    would be converted to ``int64`` after the first dtype pass had already
    protected it, dropping the padding.
    """
    column, report = _refine(pd.Series(ZIPS, dtype=object))
    assert column.tolist() == ZIPS
    assert report.actions == []


def test_refine_numeric_retry_honours_the_leading_zero_opt_out():
    """The veto is *gated on the flag*, not unconditional.

    Counterpart to the test above: with ``preserve_leading_zeros=False`` the
    very same column must convert, which is what makes the flag meaningful.
    """
    column, report = _refine(pd.Series(ZIPS, dtype=object), preserve_leading_zeros=False)
    assert column.dtype == "int64"
    assert column.tolist() == [1234, 2115, 3301, 4401, 5501]
    assert "after semantic repair" in report.actions[0].description


# --------------------------------------------------------------------------
# _looks_dateish: "nothing to look at" must mean no
# --------------------------------------------------------------------------


def test_dateish_screen_rejects_a_sample_with_no_usable_text_at_all():
    """No inspectable string means "not date-shaped", not "date-shaped".

    A mixed column of ``datetime.date`` objects and an integer offers the
    pre-screen nothing to read. Answering *yes* would hand the column to
    ``to_datetime``, which reinterprets the bare integer as a nanosecond
    epoch and fabricates a 1970 timestamp for it.
    """
    values = pd.Series(
        [dt.date(2021, 1, 5), dt.date(2021, 2, 6), dt.date(2021, 3, 7), 42],
        dtype=object,
    )
    assert pd.api.types.infer_dtype(values, skipna=True) == "mixed-integer"
    assert _looks_dateish(values) is False
    assert suggest_conversion(values, CleanConfig()) == ("none", None, 0)

    report = CleanReport()
    out = fix_dtypes(pd.DataFrame({"v": values}), CleanConfig(), report)
    assert out["v"].tolist() == values.tolist()
    assert report.actions == []
    # What the screen prevents: 42 would have become
    # 1970-01-01T00:00:00.000000042. Illustrative only, and pandas 1.x coerces
    # the bare int to NaT instead, so it is asserted on pandas 2+ only.
    if PANDAS_MAJOR >= 2:
        assert pd.to_datetime(values, errors="coerce").iloc[3] == pd.Timestamp(42)


def test_dateish_screen_rejects_a_sample_whose_text_is_all_too_long_to_read():
    """The other way the inspectable set empties out: every value over the cap.

    Each value here is a perfectly parseable timestamp, so a *yes* would
    convert the column; the screen still has to say no because it never looked
    at any of them.
    """
    too_long = [f"2021-01-0{i}T14:30:00.123456+00:00{' ' * 9}" for i in range(1, 6)]
    assert {len(v) for v in too_long} == {41}
    values = pd.Series(too_long, dtype=object)
    assert _looks_dateish(values) is False
    assert suggest_conversion(values, CleanConfig()) == ("none", None, 0)
    # The values themselves are not the problem -- only their length is.
    # ``format="mixed"`` was added in pandas 2.0, so this is asserted there.
    if PANDAS_MAJOR >= 2:
        assert pd.to_datetime(values, format="mixed", errors="coerce").notna().all()


# --------------------------------------------------------------------------
# _record_coerced: the day/month-ambiguity note
# --------------------------------------------------------------------------


def _iso_dates_plus(extra):
    """19 unambiguous ISO dates plus one trailing value, as object dtype."""
    values = [f"2021-{m:02d}-{m:02d}" for m in range(1, 13)]
    values += [f"2022-{m:02d}-{m:02d}" for m in range(1, 8)]
    return pd.Series([*values, extra], dtype=object)


def test_a_non_text_casualty_is_never_reported_as_an_ambiguous_date():
    """Only *text* can carry a day/month reading.

    A ``datetime.time`` cell in a date column coerces to ``NaT`` and is
    recorded as a casualty, but "pass dayfirst=True or dayfirst=False" is
    useless advice for it, so the note must not fire.
    """
    values = _iso_dates_plus(dt.time(12, 0))
    report = CleanReport()
    out = fix_dtypes(pd.DataFrame({"v": values}), CleanConfig(), report)

    assert pd.api.types.is_datetime64_any_dtype(out["v"].dtype)
    assert report.coerced_cells["v"] == {19: dt.time(12, 0)}
    assert [a.description for a in report.actions] == [
        "converted to datetime64[ns] (1 unparseable value(s) set to missing)"
    ]
    assert not any("ambiguous" in a.description for a in report.actions)


def test_a_text_casualty_that_is_ambiguous_still_raises_the_note():
    """Counterpart: a genuinely ambiguous date string does raise the note."""
    values = _iso_dates_plus("01/02/2023")
    report = CleanReport()
    out = fix_dtypes(pd.DataFrame({"v": values}), CleanConfig(), report)

    assert pd.api.types.is_datetime64_any_dtype(out["v"].dtype)
    assert report.coerced_cells["v"] == {19: "01/02/2023"}
    notes = [a for a in report.actions if "ambiguous" in a.description]
    assert len(notes) == 1
    assert notes[0].count == 1
    assert "dayfirst" in notes[0].rationale


# --------------------------------------------------------------------------
# _fits_int64: the exception guard is unreachable from _finalize_numeric
# --------------------------------------------------------------------------


def _adversarial_numeric_series():
    """Series built to break ``int(min)`` / ``int(max)`` if anything can.

    Deliberately not a pool of ordinary scalars: non-finite floats in every
    float container pandas offers (numpy, ``float32``, masked ``Float64``,
    object), integers far outside int64 in both directions,
    ``Decimal``/``Fraction`` payloads, the unsigned extreme, and the exact
    int64 bounds.
    """
    inf, ninf, nan = float("inf"), float("-inf"), float("nan")
    return [
        ("float64 with +inf", pd.Series([1.0, inf])),
        ("float64 with -inf", pd.Series([1.0, ninf])),
        ("float64 all inf", pd.Series([inf, inf])),
        ("float64 inf and nan", pd.Series([inf, nan, 1.0])),
        ("float32 with inf", pd.Series(np.array([1.0, np.float32("inf")], dtype="float32"))),
        ("Float64 with inf", pd.Series([1.0, inf], dtype="Float64")),
        ("Float64 inf only", pd.Series([inf], dtype="Float64")),
        ("Float64 NA and inf", pd.Series([pd.NA, inf], dtype="Float64")),
        ("object with inf", pd.Series([1, inf], dtype=object)),
        ("object with -inf", pd.Series([1, ninf], dtype=object)),
        ("object int beyond int64", pd.Series([2**70, 1], dtype=object)),
        ("object int below int64", pd.Series([-(2**70), 1], dtype=object)),
        ("object Decimal", pd.Series([Decimal("1"), Decimal("2")], dtype=object)),
        ("object Fraction", pd.Series([Fraction(2, 1), Fraction(4, 1)], dtype=object)),
        ("object bool", pd.Series([True, False], dtype=object)),
        ("int64 exact bounds", pd.Series([_INT64_MIN, _INT64_MAX])),
        ("uint64 max", pd.Series([np.iinfo(np.uint64).max], dtype="uint64")),
        ("Int64 with NA", pd.Series([1, pd.NA], dtype="Int64")),
        ("float64 near overflow", pd.Series([1e308, 1.0])),
        ("float64 exactly 2**63", pd.Series([float(2**63)])),
        ("float64 exactly -2**63", pd.Series([float(-(2**63))])),
        ("empty float64", pd.Series([], dtype="float64")),
    ]


def _fits_int64_with_flipped_guard(nonnull):
    """``_fits_int64`` with its exception guard answering the opposite way."""
    try:
        return int(nonnull.min()) >= _INT64_MIN and int(nonnull.max()) <= _INT64_MAX
    except (OverflowError, ValueError):
        return True


def test_int64_fit_guard_never_runs_and_cannot_change_finalize_numeric():
    """Execution proof that ``_fits_int64``'s ``except`` arm is dead code here.

    ``_finalize_numeric`` consults ``_fits_int64`` only after
    ``(nonnull % 1 == 0).all()`` has held, and that is false for every
    non-finite value (``inf % 1`` is ``nan``), so ``int(min)``/``int(max)``
    only ever see finite, exactly convertible numbers.

    Two independent checks, both by execution rather than argument:

    1. a line trace over the whole adversarial pool records zero hits on the
       guard's ``return`` line, while recording many hits on the function;
    2. swapping in a ``_fits_int64`` whose guard answers the opposite way
       leaves every ``_finalize_numeric`` result identical.
    """
    pool = _adversarial_numeric_series()
    source_file = inspect.getsourcefile(_fits_int64)
    lines, first_line = inspect.getsourcelines(_fits_int64)
    guard_line = first_line + len(lines) - 1
    assert lines[-1].strip() in ("return False", "return True")

    hits = {}

    def tracer(frame, event, _arg):
        if frame.f_code.co_filename != source_file:
            return None
        if event == "line":
            hits[frame.f_lineno] = hits.get(frame.f_lineno, 0) + 1
        return tracer

    real, flipped, reached = {}, {}, 0
    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        for name, series in pool:
            before = sum(hits.values())
            real[name] = _finalize_numeric(series)
            if sum(hits.values()) > before:
                reached += 1
    finally:
        sys.settrace(previous)

    original = dtypes_mod._fits_int64
    dtypes_mod._fits_int64 = _fits_int64_with_flipped_guard
    try:
        for name, series in pool:
            flipped[name] = _finalize_numeric(series)
    finally:
        dtypes_mod._fits_int64 = original

    # The pool really does drive the code under test (not a vacuous proof).
    assert reached >= 10, f"only {reached} of {len(pool)} series ran _finalize_numeric"
    assert hits.get(guard_line, 0) == 0

    for name, _series in pool:
        assert real[name].dtype == flipped[name].dtype, name
        assert real[name].astype(object).equals(flipped[name].astype(object)), name


def test_integral_values_are_always_exactly_convertible_to_int():
    """The property that guard unreachability rests on, checked value by value.

    For every adversarial cell: if it passes ``v % 1 == 0`` then ``int(v)``
    succeeds. Cells whose ``% 1`` raises never reach the guard either, because
    ``_finalize_numeric`` evaluates that test first and does not catch it.
    """
    inf, ninf, nan = float("inf"), float("-inf"), float("nan")
    cells = [
        inf, ninf, nan, 0.0, -0.0, 1.0, -1.0, 1e308, -1e308, float(2**63),
        float(-(2**63)), 2**70, -(2**70), np.float32("inf"), np.float64("inf"),
        np.float64("nan"), np.int64(_INT64_MIN), np.int64(_INT64_MAX),
        np.uint64(np.iinfo(np.uint64).max), True, False,
        Decimal("1"), Decimal("1.5"), Fraction(2, 1), Fraction(1, 2),
    ]
    integral, convertible, non_integral = 0, 0, 0
    for cell in cells:
        series = pd.Series([cell], dtype=object)
        try:
            is_integral = bool((series % 1 == 0).all())
        except Exception:
            continue  # `% 1` raised: _finalize_numeric propagates, guard unreached
        if not is_integral:
            non_integral += 1
            continue
        integral += 1
        int(series.min())  # must not raise — this is the claim under test
        int(series.max())
        convertible += 1
    assert integral >= 12
    assert non_integral >= 5  # the non-finite cells really are screened out
    assert convertible == integral
