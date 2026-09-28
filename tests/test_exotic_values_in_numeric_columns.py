"""Exotic Python values in an otherwise numeric or textual object column.

FD2-012: one ``complex`` value beside text crashed ``fd.clean`` on default
settings. pandas' ``to_numeric`` returns a complex result as soon as one cell
is complex, and it never writes a text, bytes or bool cell into that complex
buffer -- those cells come back as whatever the last same-sized buffer freed
by numpy held. The garbage counted as "parsed", the column passed the numeric
threshold, and the real-number finalizer's ``% 1`` raised ``TypeError``.

The numeric targets here are real dtypes, so a complex value is not a number
this step can hold. It is treated as any other unparseable straggler: it
counts against ``numeric_threshold``; when the column still converts it is
quarantined (set to missing, original kept in ``report.coerced_cells``); when
the column does not convert it is left exactly as it was, and the
type-contamination warning names it.
"""

import fractions

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.report import CleanReport
from freshdata.steps.dtypes import (
    _to_numeric_or_none,
    refine_numeric_after_semantic,
    suggest_conversion,
)

COMPLEX_VALUES = [
    pytest.param(complex(1, 2), id="complex"),
    pytest.param(np.complex128(1 + 2j), id="np.complex128"),
    pytest.param(np.complex64(1 + 2j), id="np.complex64"),
]

NUMERIC_STRINGS = [str(i) for i in range(1, 21)]  # 20 of 21 parse: >= 0.95


def _clean(values, **options):
    out, report = fd.clean(pd.DataFrame({"v": values}), return_report=True,
                           verbose=False, drop_duplicates=False, **options)
    return out["v"], report


def _poison_numpy_buffer_cache(n: int, value: complex) -> None:
    """Free an ``n``-cell complex128 buffer holding *value*.

    numpy hands small freed buffers straight back to the next allocation of the
    same size, so a pandas call that reads an unwritten slot of its complex
    buffer sees *value*. This makes the pandas garbage deterministic instead of
    depending on whatever ran earlier in the process.
    """
    buf = np.full(n, value, dtype=np.complex128)
    del buf


# ── the reported crash ───────────────────────────────────────────────────────


@pytest.mark.parametrize("value", COMPLEX_VALUES)
def test_complex_beside_text_is_left_intact(value):
    """The FD2-012 reproduction: the column is not numeric (one of three cells
    parses), so it is declined and returned exactly as given."""
    column, report = _clean([value, "abc", "3"])

    assert column.dtype == object
    assert column.tolist() == [value, "abc", "3"]
    assert column.iloc[0] == complex(1, 2)
    assert "v" not in report.coerced_cells


@pytest.mark.parametrize("fill", [complex(7, 7), complex(np.nan, np.nan)])
def test_parse_beside_a_complex_value_does_not_read_pandas_garbage(fill):
    """The cause. Whatever the freed buffer held -- a plausible number or NaN --
    the text cells are parsed as text: ``"abc"`` is missing, ``"3"`` is 3.0,
    and the complex cell is unparseable. Before the fix the result was
    complex128 and ``"abc"``/``"3"`` both came back as *fill*."""
    s = pd.Series([complex(1, 2), "abc", "3"], dtype=object)
    _poison_numpy_buffer_cache(len(s), fill)
    parsed = _to_numeric_or_none(s)

    assert parsed is not None
    assert parsed.dtype == np.float64
    assert parsed.isna().tolist() == [True, True, False]
    assert parsed.iloc[2] == 3.0


def test_result_does_not_depend_on_what_ran_earlier_in_the_process():
    """The order dependence recorded in FD2-012: a NaN-filled freed buffer made
    the garbage fail the threshold (no crash), a 7+7j one made ``"abc"`` look
    parsed (crash). Both states must now give the same, intact frame."""
    frame = pd.DataFrame({"v": [complex(1, 2), "abc", "3"]})
    results = []
    for fill in (complex(np.nan, np.nan), complex(7, 7)):
        _poison_numpy_buffer_cache(len(frame), fill)
        results.append(fd.clean(frame, verbose=False)["v"].tolist())
    assert results == [[complex(1, 2), "abc", "3"]] * 2


# ── a complex straggler in a numeric column is quarantined, not lost ─────────


@pytest.mark.parametrize("value", COMPLEX_VALUES)
def test_complex_straggler_in_a_numeric_column_is_quarantined(value):
    """20 numeric strings and one complex value clear the 0.95 threshold, so the
    column converts. The complex value cannot live in Int64: it is set to
    missing, its original is in ``coerced_cells``, and a warning says so."""
    column, report = _clean([value, *NUMERIC_STRINGS])

    assert str(column.dtype) == "Int64"
    assert column.isna().tolist() == [True] + [False] * 20
    assert column.iloc[1:].tolist() == list(range(1, 21))
    assert report.coerced_cells["v"] == {0: value}
    assert report.coerced_rows["v"] == (0,)
    assert any("could not be parsed as Int64" in w and "(1+2j) (row 0)" in w
               for w in report.warnings)


def test_complex_straggler_is_handled_exactly_like_other_unparseable_values():
    """Alignment with the existing straggler path: a ``Fraction`` (which pandas
    cannot parse either) in the same position gets the same dtype, the same
    missing cell and the same ``coerced_cells`` entry."""
    for values in ([None, *NUMERIC_STRINGS], [None, "1", "2", "3", "4"]):
        with_complex = _clean([complex(1, 2), *values[1:]])
        with_fraction = _clean([fractions.Fraction(1, 3), *values[1:]])
        assert str(with_complex[0].dtype) == str(with_fraction[0].dtype)
        assert with_complex[0].isna().tolist() == with_fraction[0].isna().tolist()
        assert (set(with_complex[1].coerced_cells.get("v", {}))
                == set(with_fraction[1].coerced_cells.get("v", {})))


def test_complex_below_threshold_leaves_the_column_and_is_named_in_the_warning():
    """Four numbers and one complex value: 80% parse, below 0.95, so nothing is
    converted and the contamination warning points at the complex row."""
    column, report = _clean([complex(1, 2), "1", "2", "3", "4"])

    assert column.dtype == object
    assert column.tolist() == [complex(1, 2), "1", "2", "3", "4"]
    assert "v" not in report.coerced_cells
    assert any("looks numeric (80% of values parse)" in w and "(1+2j) (row 0)" in w
               and "values were NOT silently coerced" in w for w in report.warnings)


def test_complex_beside_python_numbers_in_an_object_column_is_left_intact():
    """No text at all: pandas parses this correctly (complex128), and the
    finalizer then crashed on ``% 1``. One of five is not real -> declined."""
    values = pd.Series([complex(1, 2), 1, 2, 3, 4.5], dtype=object)
    column, report = _clean(values)

    assert column.dtype == object
    assert column.tolist() == [complex(1, 2), 1, 2, 3, 4.5]
    assert "v" not in report.coerced_cells


def test_lower_threshold_quarantines_the_complex_value_beside_text():
    """With ``numeric_threshold=0.3`` the FD2-012 column converts; the complex
    value and the word are both casualties, recorded with their originals."""
    column, report = _clean([complex(1, 2), "abc", "3"], numeric_threshold=0.3)

    assert str(column.dtype) == "Int64"
    assert column.isna().tolist() == [True, True, False]
    assert column.iloc[2] == 3
    assert report.coerced_cells["v"] == {0: complex(1, 2), 1: "abc"}


def test_profile_preview_matches_cleaning():
    """``suggest_conversion`` is shared with ``fd.profile``; it must report the
    same decision instead of crashing."""
    config = CleanConfig()
    beside_text = pd.Series([complex(1, 2), "abc", "3"], dtype=object)
    assert suggest_conversion(beside_text, config) == ("none", None, 0)

    target, converted, n_coerced = suggest_conversion(
        pd.Series([complex(1, 2), *NUMERIC_STRINGS], dtype=object), config)
    assert (target, str(converted.dtype), n_coerced) == ("numeric", "Int64", 1)


def test_post_semantic_retry_declines_a_complex_casualty():
    """``refine_numeric_after_semantic`` only quarantines numeric-*shaped* text.
    A complex value is not, so the retry leaves the column untouched."""
    df = pd.DataFrame({"v": [complex(1, 2), "1", "2", "3", "4", "5"]})
    report = CleanReport()
    out = refine_numeric_after_semantic(df, ["v"], CleanConfig(), report)

    assert out["v"].tolist() == [complex(1, 2), "1", "2", "3", "4", "5"]
    assert "v" not in report.coerced_cells


# ── controls: columns without a complex value convert exactly as before ──────


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        (["1", "2", "3"], "int64", [1, 2, 3]),
        (["1.5", "2", "-3"], "float64", [1.5, 2.0, -3.0]),
        (["1e3", "2", "-0.5"], "float64", [1000.0, 2.0, -0.5]),
        ([str(i) for i in range(19)] + ["$1,234.56"], "float64",
         [float(i) for i in range(19)] + [1234.56]),
        (["inf", "1", "2"], "float64", [float("inf"), 1.0, 2.0]),
    ],
)
def test_ordinary_numeric_text_converts_as_before(values, dtype, expected):
    column, report = _clean(values)
    assert str(column.dtype) == dtype
    assert column.tolist() == expected
    assert "v" not in report.coerced_cells


def test_ordinary_text_straggler_is_still_quarantined_as_before():
    column, report = _clean(["abc", *NUMERIC_STRINGS])
    assert str(column.dtype) == "Int64"
    assert column.iloc[1:].tolist() == list(range(1, 21))
    assert report.coerced_cells["v"] == {0: "abc"}


def test_all_complex_object_column_is_still_declined_before_parsing():
    """``infer_dtype`` reports ``"complex"`` and the column never reaches the
    numeric path -- unchanged by this fix."""
    column, report = _clean(pd.Series([complex(1, 2), complex(3, 4)], dtype=object))
    assert column.tolist() == [complex(1, 2), complex(3, 4)]
    assert "v" not in report.coerced_cells
