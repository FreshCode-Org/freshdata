import datetime as dt

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.steps.dtypes import _finalize_numeric, _to_numeric_or_none


def clean1(values, **options):
    """Clean a single-column frame and return the resulting column."""
    out = fd.clean(pd.DataFrame({"v": values}), **options)
    return out["v"]


def is_string(dtype) -> bool:
    return pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)


def test_integer_strings_become_int64():
    s = clean1(["1", "2", "3"])
    assert s.dtype == "int64"
    assert s.tolist() == [1, 2, 3]


def test_integer_strings_with_missing_become_nullable_int():
    s = clean1(["1", None, "3"], drop_empty_rows=False)
    assert s.dtype == "Int64"


def test_float_strings_become_float64():
    s = clean1(["1.5", "2.25", "-3.0e2"])
    assert s.dtype == "float64"
    assert s.tolist() == [1.5, 2.25, -300.0]


def test_currency_and_thousands_separators():
    s = clean1(["$1,200.50", "$2,000", "€3,500.75", "900"])
    assert s.dtype == "float64"
    assert s.tolist() == [1200.50, 2000.0, 3500.75, 900.0]


def test_junk_column_stays_text():
    s = clean1(["1", "2", "x", "y"])
    assert is_string(s.dtype)


def test_threshold_boundary():
    # conservative strategy: the NaN coerced from "junk" must survive so the
    # conversion threshold itself is observable.
    mostly = [str(i) for i in range(19)] + ["junk"]  # 19/20 = 0.95 -> convert
    s = clean1(mostly, strategy="conservative")
    assert s.dtype == "Int64"
    assert s.isna().sum() == 1

    below = [str(i) for i in range(18)] + ["junk"]  # 18/19 < 0.95 -> keep text
    s = clean1(below)
    assert is_string(s.dtype)


def test_coerced_values_are_reported():
    df = pd.DataFrame({"v": [str(i) for i in range(19)] + ["junk"]})
    _, report = fd.clean(df, return_report=True)
    [action] = [a for a in report if a.step == "fix_dtypes"]
    assert "unparseable" in action.description


def test_boolean_vocabulary():
    assert clean1(["yes", "no", "YES", "No"]).dtype == bool
    assert clean1(["true", "false", "T", "f"]).dtype == bool
    s = clean1(["y", None, "n"], drop_empty_rows=False)
    assert s.dtype == "boolean"


def test_boolean_objects_get_boolean_dtype():
    s = clean1([True, False, None], drop_empty_rows=False)
    assert s.dtype == "boolean"


def test_non_boolean_words_stay_text():
    assert is_string(clean1(["yes", "no", "maybe"]).dtype)


def test_zero_one_strings_become_numeric_not_boolean():
    s = clean1(["0", "1", "1", "0"])
    assert s.dtype == "int64"


def test_iso_dates_become_datetime():
    s = clean1(["2021-01-05", "2021-02-11", "2021-03-09"])
    assert str(s.dtype).startswith("datetime64")


def test_mixed_date_formats_become_datetime():
    s = clean1(["2021-01-05", "05/30/2021", "March 9, 2021"])
    assert str(s.dtype).startswith("datetime64")
    assert s.isna().sum() == 0


def test_words_never_attempt_datetime():
    s = clean1(["alpha", "beta", "gamma"])
    assert is_string(s.dtype)


def test_relative_date_words_never_silently_use_real_date():
    # A column of otherwise-parseable dates plus "today" must NOT be
    # auto-converted here: pd.to_datetime("today") resolves to the real
    # wall-clock date, and fix_dtypes has no reference_date to consult (only
    # the semantic layer's DatePhraseExpert does, gated explicitly on one).
    s = clean1(["2026-01-01", "2026-02-01", "2026-03-01", "today"])
    assert is_string(s.dtype)
    assert "today" in s.tolist()


def test_id_like_strings_stay_text():
    s = clean1(["A123", "B456", "C789"])
    assert is_string(s.dtype)


def test_compact_digit_strings_become_numeric_not_datetime():
    s = clean1(["20210105", "20210211", "20210309"])
    assert s.dtype == "int64"


def test_fix_dtypes_can_be_disabled():
    s = clean1(["1", "2", "3"], fix_dtypes=False)
    assert is_string(s.dtype)


def test_numeric_threshold_is_configurable():
    s = clean1(["1", "2", "junk", "4"], numeric_threshold=0.7)
    assert s.dtype == "Int64"
    assert s.isna().sum() == 1


def test_existing_typed_columns_untouched():
    df = pd.DataFrame(
        {
            "i": np.array([1, 2, 3], dtype="int32"),
            "f": [1.5, 2.5, 3.5],
            "d": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
        }
    )
    out = fd.clean(df)
    assert out["i"].dtype == "int32"
    assert out["f"].dtype == "float64"
    assert str(out["d"].dtype).startswith("datetime64")


def test_date_objects_normalized_to_datetime64():
    s = clean1([dt.date(2021, 1, 5), dt.date(2021, 2, 11), dt.date(2021, 3, 9)])
    assert str(s.dtype).startswith("datetime64")


@pytest.mark.parametrize("huge", [["9" * 25, "8" * 25]])
def test_huge_integers_stay_float_not_overflow(huge):
    s = clean1(huge)
    assert s.dtype == "float64"


def test_finalize_numeric_int64_boundaries_no_overflow_no_demotion():
    """Regression for #34: exact int64 boundary handling.

    float64 cannot represent 2**63 - 1; it rounds up to 2**63, so any
    float-space threshold either rejects legitimate values or admits an
    overflowing one.  The largest float64 below 2**63 is 2**63 - 1024 and
    must convert exactly; float(2**63) must stay float64, never wrap.
    """
    top = float(2**63 - 1024)
    out = _finalize_numeric(pd.Series([top, 1.0]))
    assert str(out.dtype) == "int64"
    assert int(out.iloc[0]) == 2**63 - 1024

    out = _finalize_numeric(pd.Series([float(2**63), 1.0]))
    assert str(out.dtype) == "float64"

    out = _finalize_numeric(pd.Series([top, None]))
    assert str(out.dtype) == "Int64"
    assert int(out.iloc[0]) == 2**63 - 1024


def test_finalize_numeric_int64_min_not_rejected_by_abs_asymmetry():
    """int64's range is asymmetric: -2**63 is representable, +2**63 is not.
    A magnitude-only guard rejected the legitimate minimum."""
    bottom = float(-(2**63))
    out = _finalize_numeric(pd.Series([bottom, 0.0]))
    assert str(out.dtype) == "int64"
    assert int(out.iloc[0]) == -(2**63)

    out = _finalize_numeric(pd.Series([float(-(2**64)), 0.0]))
    assert str(out.dtype) == "float64"


def test_huge_integer_strings_still_stay_float_end_to_end():
    """Pipeline-level guard: values beyond int64 parse to float64, exactly
    as before the boundary fix."""
    s = clean1(["18446744073709551616", "1"])  # 2**64
    assert s.dtype == "float64"


def test_unsafe_scientific_exponents_are_quarantined_before_pandas_parse():
    """Malformed exponents must not reach pandas' vulnerable numeric parser."""
    values = pd.Series(["1", "1e3000000000", "3"])
    parsed = _to_numeric_or_none(values)
    assert parsed is not None
    assert parsed.iloc[0] == 1
    assert pd.isna(parsed.iloc[1])
    assert parsed.iloc[2] == 3


def test_unsafe_exponent_guard_handles_mixed_and_boundary_payloads():
    """The vectorized candidate scan must match the per-value guard exactly:
    non-strings pass through, E308 stays parseable, E309 and an unparseable
    exponent are masked, and safe exponents survive."""
    values = pd.Series(
        ["1E308", "1e309", "2.5e-309", "1e+10", b"1e999", 7, None, "1" + "0" * 40]
    )
    parsed = _to_numeric_or_none(values)
    assert parsed is not None
    assert parsed.iloc[0] == 1e308
    assert pd.isna(parsed.iloc[1])  # exponent 309 > 308: masked pre-parse
    assert pd.isna(parsed.iloc[2])  # -309 out of range: masked pre-parse
    assert parsed.iloc[3] == 1e10
    assert pd.isna(parsed.iloc[4])  # bytes are not a string: pandas coerces to NaN
    assert parsed.iloc[5] == 7
    assert pd.isna(parsed.iloc[6])
    assert parsed.iloc[7] == 1e40


def test_unsafe_exponent_guard_handles_stringless_object_columns():
    # An object column with no strings at all (pandas .str refuses these)
    # has no unsafe tokens; it must parse instead of raising AttributeError.
    values = pd.Series([1.5, 2.5, None], dtype=object)
    parsed = _to_numeric_or_none(values)
    assert parsed is not None
    assert parsed.iloc[0] == 1.5
    assert parsed.iloc[1] == 2.5


def test_unsafe_exponent_guard_handles_nullable_string_dtype():
    values = pd.Series(["1", "1e3000000000", None, "2e3"], dtype="string")
    parsed = _to_numeric_or_none(values)
    assert parsed is not None
    assert parsed.iloc[0] == 1
    assert pd.isna(parsed.iloc[1])
    assert pd.isna(parsed.iloc[2])
    assert parsed.iloc[3] == 2000.0


def test_relative_date_words_blocked_regardless_of_case_and_whitespace():
    # Default cleaning strips surrounding whitespace (clean_strings), so the
    # value may come back trimmed — but it must stay text, never a resolved
    # wall-clock date.
    for word in ("  TODAY ", "Yesterday", "tomorrow\t"):
        s = clean1(["2026-01-01", "2026-02-01", "2026-03-01", word])
        assert is_string(s.dtype)
        assert word.strip() in {str(v).strip() for v in s.tolist()}


def test_relative_date_word_in_unhashable_company_still_blocks_conversion():
    # pd.unique raises TypeError on unhashable cells; the fallback scan must
    # still find the relative-date word.
    s = clean1(["2026-01-01", "2026-02-01", ["not", "hashable"], "today"])
    assert "today" in s.tolist()
