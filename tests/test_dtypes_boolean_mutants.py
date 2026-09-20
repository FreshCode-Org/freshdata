"""Type inference: the flags that decide when a column is *silently* retyped.

Every test here pins one decision that ``steps/dtypes.py`` makes with a
boolean literal or a subset/identity comparison — the switches that say
"this text is a boolean vocabulary", "this cell matches the formatted-number
pattern", "this column contains separator noise", "this separator is a
literal, not a regex". They are the cheap-looking constants whose flip turns
a conservative *leave it as text* into a silent conversion (or the reverse),
which is exactly the class of change no assertion in ``test_dtypes.py``
notices.

Companion to ``test_dtypes.py``: that file covers the happy paths and the
threshold/quarantine behaviour; this one covers the boundaries those tests
step around.
"""

import datetime as dt

import pandas as pd
import pytest

import freshdata as fd
from freshdata._util import sample_series
from freshdata.config import CleanConfig
from freshdata.steps.dtypes import (
    _BOOL_WORDS,
    _finalize_numeric,
    _number_format,
    _rescue_formatted,
    _to_numeric_or_none,
    suggest_conversion,
)


def clean1(values, **options):
    """Clean a single-column frame and return the resulting column."""
    return fd.clean(pd.DataFrame({"v": values}), **options)["v"]


def is_string(dtype) -> bool:
    return pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)


# ── boolean vocabulary: the subset test is a *subset-or-equal* test ──────────


def test_column_holding_the_entire_boolean_vocabulary_is_still_boolean():
    """A column whose values are *exactly* the true/false vocabulary — all
    eight spellings, nothing else — is the most boolean column there is. The
    detection asks ``values <= _BOOL_WORDS``; a strict ``<`` would reject the
    one column that uses the whole vocabulary while accepting every subset of
    it, so the conversion would depend on which spellings happen to appear.
    """
    words = ["true", "t", "yes", "y", "false", "f", "no", "n"]
    assert {w.casefold() for w in words} == _BOOL_WORDS  # the equality case

    s = clean1(words)
    assert s.dtype == bool
    assert s.tolist() == [True, True, True, True, False, False, False, False]


def test_full_vocabulary_with_mixed_case_and_a_missing_cell_still_boolean():
    """The equality case again, through ``casefold`` and with a missing cell
    so the nullable ``boolean`` dtype (not ``bool``) is the result."""
    words = ["TRUE", "T", "Yes", "Y", "False", "f", "NO", "n", None]
    s = clean1(words, drop_empty_rows=False)
    assert s.dtype == "boolean"
    assert s.tolist()[:8] == [True, True, True, True, False, False, False, False]
    assert pd.isna(s.iloc[8])


# ── the decimal separator is replaced literally, never as a regex ────────────


@pytest.mark.parametrize("separator", ["|", "^"])
def test_decimal_separator_is_replaced_literally_not_as_a_regex(separator):
    """``cleanup`` rewrites the locale decimal separator to ``"."`` with
    ``regex=False``. The separator is caller-supplied and may be a regex
    metacharacter: ``"|"`` as a pattern matches the empty string at every
    position and ``"^"`` matches the start, so a regex replace would sprinkle
    dots through the value instead of swapping the separator.
    """
    _, _, cleanup = _number_format(CleanConfig(decimal=separator, thousands=","))
    values = [f"1{separator}5", f"2{separator}25", f"3{separator}75", f"10{separator}0"]
    assert cleanup(pd.Series(values, dtype=object)).tolist() == [
        "1.5", "2.25", "3.75", "10.0",
    ]

    s = clean1(values, decimal=separator)
    assert s.dtype == "float64"
    assert s.tolist() == [1.5, 2.25, 3.75, 10.0]


def test_comma_decimal_locale_is_unaffected_by_the_literal_replace():
    """The ordinary European locale keeps working: ``","`` is not a
    metacharacter, so it is the metacharacter separators above that pin the
    ``regex=False`` flag, not this case."""
    s = clean1(["1.234,56", "2.000,00", "3.500,75"], decimal=",", thousands=".")
    assert s.dtype == "float64"
    assert s.tolist() == [1234.56, 2000.0, 3500.75]


# ── a cell that is not readable text is neither "noise" nor a "match" ────────


def test_unreadable_cells_do_not_count_as_separator_noise():
    """The formatted-number second pass is only attempted when the *sample*
    actually contains a currency/separator character (``na=False``: a cell
    with no text view is not evidence of noise).

    ``b"\\xff"`` is a non-UTF-8 BLOB, so the text view of the column has a
    missing cell (see ``_text_view``). Counting that cell as noise starts the
    formatted-number pass on a column that has no separators at all, and
    ``"+ 12"`` — which plain ``to_numeric`` rejects — is then rewritten to a
    number, so the BLOB is quarantined as an unparseable casualty. The column
    must stay text instead.
    """
    values = [b"\xff", "+ 12", "+ 34", "+ 56"]
    target, converted, n_coerced = suggest_conversion(
        pd.Series(values, dtype=object), CleanConfig(numeric_threshold=0.7)
    )
    assert (target, converted, n_coerced) == ("none", None, 0)

    out, report = fd.clean(
        pd.DataFrame({"v": values}), numeric_threshold=0.7, return_report=True,
    )
    assert is_string(out["v"].dtype)
    assert out["v"].tolist() == values  # the BLOB is neither decoded nor nulled
    assert "v" not in report.coerced_cells


def test_unreadable_cell_is_not_a_formatted_number_straggler():
    """``_rescue_formatted`` re-parses the values a plain ``to_numeric``
    nulled, but only those that match the formatted-number pattern. A
    non-UTF-8 cell has no text view at all (``_text_view`` yields a missing
    cell for it), and a missing cell is not a match — so for a column whose
    only casualties are unreadable/word values there is nothing to rescue and
    the parse is handed back untouched, not rewritten.
    """
    formatted_re, _, cleanup = _number_format(CleanConfig())
    values = ["1", "2", "abc", b"\xff"]

    s = pd.Series(values, dtype=object)
    parsed = _to_numeric_or_none(s)
    # Untouched means untouched: nothing matched, so the same object comes
    # back rather than a rewritten copy.
    assert _rescue_formatted(s, parsed, formatted_re, cleanup) is parsed

    # Second, independent check that no write happens: with repeated index
    # labels a write through ``.loc`` cannot succeed, so a rescue attempt on
    # the unreadable cell would raise instead of returning the parse.
    dup = pd.Series(values, index=[0, 0, 1, 1], dtype=object)
    dup_parsed = _to_numeric_or_none(dup)
    out = _rescue_formatted(dup, dup_parsed, formatted_re, cleanup)
    assert out.tolist()[:2] == [1.0, 2.0]
    assert out.isna().tolist() == [False, False, True, True]

    # Control: a real formatted straggler in the same shape *is* rescued.
    with_straggler = pd.Series(["1", "2", "$1,234.56", b"\xff"], dtype=object)
    rescued = _rescue_formatted(
        with_straggler, _to_numeric_or_none(with_straggler), formatted_re, cleanup,
    )
    assert rescued.tolist()[:3] == [1.0, 2.0, 1234.56]


def test_missing_cells_are_not_formatted_number_matches():
    """A missing cell has no text, so it cannot match the formatted-number
    pattern (``fillna(False)``). If it did, a column where *no* value matches
    would still enter the formatted-number pass, and a sample-based rejection
    would be silently overturned by the presence of a missing value.

    The column below is deliberately one the cheap sample pre-screen rejects:
    ``sample_size=1`` draws the single unparseable value, so the full-column
    parse is never attempted. ``"1e5"`` parses as a number but never matches
    the formatted-number pattern (no exponent in it), so the only "match" a
    mutated fill could produce is the missing cell.
    """
    s = pd.Series(["1e5"] * 19 + ["$x", None], dtype="string")
    nonnull = s.dropna()

    def seed_sampling(value):
        return next(
            r for r in range(50) if sample_series(nonnull, 1, r).tolist() == [value]
        )

    # Sample the unparseable value: the pre-screen declines the column, and
    # the missing cell must not overturn that by posing as a match.
    declined = CleanConfig(sample_size=1, random_state=seed_sampling("$x"))
    assert suggest_conversion(s, declined) == ("none", None, 0)

    # Control, same column: sample a parseable value and the column converts,
    # so it is the sample — not the missing cell — that decides.
    accepted = CleanConfig(sample_size=1, random_state=seed_sampling("1e5"))
    target, converted, _ = suggest_conversion(s, accepted)
    assert target == "numeric"
    assert converted.tolist()[0] == 100000


# ── defects found while writing the tests above: pinned, not fixed ──────────


def test_duplicate_index_labels_currently_break_the_formatted_number_rescue():
    """Pinned defect — current behaviour, deliberately NOT fixed here.

    ``_rescue_formatted`` writes the rescued values back with
    ``parsed.loc[rescued.index] = rescued.to_numpy()``. When the frame's index
    repeats a label, ``.loc`` expands that label to *every* row carrying it, so
    the assignment length stops matching the values and the whole clean raises
    ``ValueError`` instead of converting the column. The identical frame with a
    unique index converts, which is what makes this a defect rather than a
    documented limitation.
    """
    values = [str(i) for i in range(19)] + ["$1,234.56"]
    repeated = pd.DataFrame({"v": values}, index=list(range(10)) * 2)
    with pytest.raises(ValueError, match="list-like indexer"):
        fd.clean(repeated, verbose=False)

    s = clean1(values)  # control: same values, unique index
    assert s.dtype == "float64"
    assert s.tolist()[-1] == 1234.56


def test_complex_value_beside_text_currently_crashes_the_numeric_finalizer():
    """Pinned defect — current behaviour, deliberately NOT fixed here.

    ``_finalize_numeric``'s integrality check is ``nonnull % 1 == 0``, and
    ``complex`` has no ``%``, so a complex value that reaches the finalizer
    raises ``TypeError`` instead of the column being left alone. A column of
    complex values *alone* is declined earlier (``infer_dtype`` reports
    ``"complex"``, which is not a text-ish kind); one complex value beside text
    makes the column ``"mixed"``, which is how it gets there.

    Asserted against ``_finalize_numeric`` directly rather than through
    ``fd.clean``. The end-to-end route is **order-dependent**: the same
    ``fd.clean(...)`` call raises in a fresh process but does not after certain
    other work has happened in the same process, so an ``fd.clean``-level
    ``pytest.raises`` here was genuinely flaky under randomised test ordering.
    The unit-level behaviour is stable, and it is the defect. See
    ``test_the_end_to_end_route_to_this_crash_is_order_dependent``.
    """
    # Three spellings of the same failure across pandas/numpy versions: object
    # dtype falls back to Python's ``%`` ("unsupported operand type(s)"), a
    # numpy complex array hits the ufunc ("ufunc 'remainder' not supported"),
    # and pandas 1.x raises its own ("can't mod complex numbers"). The point is
    # that it raises at all.
    with pytest.raises(TypeError, match="remainder|unsupported operand|mod complex"):
        _finalize_numeric(pd.Series([complex(1, 2), 3], dtype=object))

    # An all-complex column is declined before reaching the finalizer, which is
    # the behaviour the mixed column should have as well.
    out = fd.clean(pd.DataFrame({"v": [complex(1, 2), complex(3, 4)]}), verbose=False)
    assert str(out["v"].dtype) == "complex128"


def test_the_end_to_end_route_to_this_crash_is_order_dependent():
    """Records an unexplained order dependence rather than hiding it.

    In a fresh process ``fd.clean`` on a complex-beside-text column raises the
    ``TypeError`` above. After some other pandas work in the same process it
    does not — the column stops being routed to the numeric path. Both
    ``pd.api.types.infer_dtype`` (always ``"mixed"``) and ``_finalize_numeric``
    (always raises) were checked and are stable, so the divergence is upstream
    of the finalizer in ``clean``'s routing. The mechanism was not identified.

    This test asserts only what is stable in either state: the frame is
    returned unchanged, or the call raises ``TypeError`` — never a silently
    coerced column that has lost the complex value.
    """
    frame = pd.DataFrame({"v": [complex(1, 2), "abc", "3"]})
    try:
        out = fd.clean(frame, verbose=False)
    except TypeError as exc:
        assert any(m in str(exc) for m in ("remainder", "unsupported operand", "mod complex"))
        return
    assert out["v"].tolist() == [complex(1, 2), "abc", "3"]

def test_date_objects_with_a_missing_cell_still_normalize_to_datetime64():
    """``infer_dtype(s, skipna=True)`` is what makes a column of ``date``
    objects report as ``"date"`` even when a cell is missing. Inferring with
    the missing cell included reports ``"mixed"``, which sends the column
    through the *text* heuristics — and those look for date-shaped strings,
    find none (these are ``date`` objects, not text), and leave the column as
    object. One missing cell would then decide the dtype of the column.
    """
    values = [dt.date(2021, 1, 5), None, dt.date(2021, 3, 9)]
    s = clean1(values, drop_empty_rows=False)
    assert str(s.dtype).startswith("datetime64")
    assert s.isna().tolist() == [False, True, False]
    assert s.tolist()[0] == pd.Timestamp("2021-01-05")

    values = [dt.datetime(2021, 1, 5, 9, 30), pd.NaT, dt.datetime(2021, 3, 9, 10, 0)]
    s = clean1(values, drop_empty_rows=False)
    assert str(s.dtype).startswith("datetime64")
    assert s.tolist()[2] == pd.Timestamp("2021-03-09 10:00")
