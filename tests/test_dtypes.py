import datetime as dt
import re
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.steps.dtypes import (
    _finalize_numeric,
    _has_unsafe_scientific_exponent,
    _to_numeric_or_none,
)


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
    non-strings pass through, every exponent that cannot overflow pandas' C int
    (in range, subnormal or out of range) parses exactly as pandas parses it,
    and only a ten-digit exponent is masked."""
    values = pd.Series(
        ["1E308", "1e309", "2.5e-309", "1e+10", b"1e999", 7, None, "1" + "0" * 40,
         "1e1000000000x"]
    )
    parsed = _to_numeric_or_none(values)
    assert parsed is not None
    # Exponents of at most three digits: safe to hand raw pandas in-process.
    expected = pd.to_numeric(values.iloc[:-1], errors="coerce")
    pd.testing.assert_series_equal(parsed.iloc[:-1], expected)
    assert parsed.iloc[0] == 1e308
    assert parsed.iloc[2] > 0  # subnormal: kept, not masked
    assert parsed.iloc[7] == 1e40
    assert pd.isna(parsed.iloc[-1])  # ten exponent digits: masked pre-parse


def test_dtype_inference_keeps_subnormal_and_underflow_values():
    values = pd.Series(["4.9e-324", "1e-320", "1e-310", "5e-400", "2.2e-308", "1.5"])
    parsed = _to_numeric_or_none(values)
    assert parsed is not None
    pd.testing.assert_series_equal(parsed, pd.to_numeric(values, errors="coerce"))
    assert parsed.notna().all()
    s = clean1(values.tolist(), drop_duplicates=False)
    assert s.dtype == "float64"
    assert s.notna().all()


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


# Cells pandas < 3 reads as scientific notation whose exponent overflows a C
# int (pandas-dev/pandas#62617). Each of the first four segfaulted
# ``pd.to_numeric(errors="coerce")`` on Linux x86_64 with pandas 2.3.3; the
# hex ones have the shape of hash-masked values, which is how CI hit them.
_EXPONENT_OVERFLOW_TOKENS = [
    "81e3104049863b72",
    "4e492493924924",
    "1e3104049863",
    "1e2147483648",
    "  -7.5E+99999999999xyz",
    ".5e-3104049863 tail",
]
_C_INT_EXPONENT = re.compile(r"\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?(\d{1,17})")


def _overflows_c_int_exponent(token: str) -> bool:
    """Mirror pandas' parser: up to 17 exponent digits read into a C int.

    This mirrors ``precise_xstrtod`` as reported in pandas-dev/pandas#62617.
    If pandas ever reads more digits, a token this filter lets through could
    still crash a raw ``pd.to_numeric`` call, which is why parity baselines
    run in a child interpreter."""
    match = _C_INT_EXPONENT.match(token)
    return match is not None and int(match.group(1)) > 2**31 - 1


@pytest.mark.parametrize("token", [*_EXPONENT_OVERFLOW_TOKENS, b"81e3104049863b72"])
def test_exponent_overflow_prefix_is_flagged(token):
    assert _has_unsafe_scientific_exponent(token)


@pytest.mark.parametrize(
    "token",
    ["1e308", "1e308abc", "12e3", "a1e3104049863", "e999", "1e",
     "1e309", "4.9e-324", "1e999999999", "-1e-999999999", "1e0000000001"],
)
def test_in_range_or_non_leading_exponents_are_not_flagged(token):
    assert not _has_unsafe_scientific_exponent(token)


def test_exponent_overflow_tokens_never_reach_pandas_parser():
    """Runs in a child interpreter: on pandas < 3 an unguarded token kills the
    process with SIGSEGV, which must fail this test instead of the whole run."""
    code = textwrap.dedent(
        f"""
        import pandas as pd
        import freshdata as fd
        from freshdata.steps.dtypes import _to_numeric_or_none

        tokens = {_EXPONENT_OVERFLOW_TOKENS!r}
        parsed = _to_numeric_or_none(pd.Series(["1", *tokens, "3"], dtype=object))
        assert parsed.iloc[0] == 1 and parsed.iloc[-1] == 3
        assert parsed.iloc[1:-1].isna().all()
        parsed = _to_numeric_or_none(pd.Series([b"81e3104049863b72", "2"], dtype=object))
        assert pd.isna(parsed.iloc[0]) and parsed.iloc[1] == 2
        parsed = _to_numeric_or_none(pd.Series(tokens, dtype="string"))
        assert parsed.isna().all()
        out = fd.clean(pd.DataFrame({{"token": tokens, "n": range(len(tokens))}}))
        assert out["token"].notna().all()
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])
    assert proc.stdout.strip().endswith("ok")


def test_prefix_exponent_guard_matches_pandas_on_every_safe_token():
    """Parity: the guard changes nothing on any token pandas can parse without
    overflowing its exponent accumulator. In-range, subnormal and out-of-range
    values parse as pandas parses them, and the masked ten-digit exponents here
    are ones pandas rejects anyway (the rest would crash it). The
    raw pandas baseline runs in a child interpreter, so a token that still
    crashes pandas fails this test instead of killing the whole run."""
    rng = np.random.default_rng(20260915)
    hexdigits = np.array(list("0123456789abcdef"))
    tokens = ["".join(rng.choice(hexdigits, 16)) for _ in range(4000)]
    tokens += ["1e880f3f8de2590b", "1e309abc", "2.5e-309x", "7e123456789z", "1e308x"]
    tokens += ["1e308", " 1e5 ", "12e3", "e999", "1e", "1e+", "3.5", "abc", None]
    # Subnormal, underflow and out-of-range values: parsed exactly as pandas.
    tokens += ["4.9e-324", "1e-320", "1e-310", "5e-400", "1e309", "1e400", "-1e400"]
    tokens += ["2.2e-308", "7e123456789", "1e999999999", "1e0000000001"]
    # Ten exponent digits (masked) that pandas rejects anyway.
    tokens += ["1e1000000000abc", "5e2000000000x", "4e1000000000", "2.5e-1999999999 tail"]
    safe = [t for t in tokens if t is None or not _overflows_c_int_exponent(t)]
    masked = [t for t in safe if t is not None and _has_unsafe_scientific_exponent(t)]
    assert len(masked) >= 4  # the guard is exercised
    code = textwrap.dedent(
        """
        import ast
        import sys

        import pandas as pd
        from freshdata.steps.dtypes import _to_numeric_or_none

        safe = ast.literal_eval(sys.stdin.read())
        for dtype in (object, "string"):
            values = pd.Series(safe, dtype=dtype)
            expected = pd.to_numeric(values, errors="coerce")
            parsed = _to_numeric_or_none(values)
            assert parsed is not None
            pd.testing.assert_series_equal(parsed, expected)
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        input=repr(safe),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])
    assert proc.stdout.strip().endswith("ok")


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


def test_ambiguous_numeric_date_is_quarantined_not_interpreted():
    """One '01/02/2025' (day/month both <= 12, no dayfirst evidence) must not
    be silently interpreted: the clear dates convert, and the ambiguous value
    is coerced to missing and recorded for review (TruthBench finance
    trade_date)."""
    values = ["2026-01-15"] * 6 + ["01/02/2025"]
    out, rep = fd.clean(pd.DataFrame({"v": values}), return_report=True,
                        drop_duplicates=False)
    assert str(out["v"].dtype).startswith("datetime64")
    assert out["v"].isna().sum() == 1  # the ambiguous value is quarantined
    coerced = rep.coerced_cells.get("v", {})
    assert any(str(v) == "01/02/2025" for v in coerced.values())


def test_partial_iso_date_is_quarantined_not_fabricated():
    """'2025-01' has no day; pandas would invent day=01. The clear dates
    convert and the partial value is quarantined for review instead of being
    fabricated (TruthBench healthcare event_date)."""
    values = ["2026-01-15"] * 6 + ["2025-01"]
    out, rep = fd.clean(pd.DataFrame({"v": values}), return_report=True,
                        drop_duplicates=False)
    assert str(out["v"].dtype).startswith("datetime64")
    assert out["v"].isna().sum() == 1
    coerced = rep.coerced_cells.get("v", {})
    assert any(str(v) == "2025-01" for v in coerced.values())


def test_sibling_votes_do_not_resolve_ambiguity():
    """Audit P1-2: a '05/30/2021' sibling must NOT decide how '01/02/2021' is
    read — one export can mix conventions. Unambiguous values parse on their
    own merits; ambiguous ones are quarantined for review."""
    values = ["05/30/2021", "01/02/2021", "03/04/2021", "06/20/2021"]
    out, rep = fd.clean(pd.DataFrame({"v": values}), return_report=True,
                        drop_duplicates=False)
    s = out["v"]
    assert str(s.dtype).startswith("datetime64")
    assert pd.Timestamp("2021-05-30") in list(s)
    assert pd.Timestamp("2021-06-20") in list(s)
    coerced = set(map(str, rep.coerced_cells.get("v", {}).values()))
    assert coerced == {"01/02/2021", "03/04/2021"}


def test_explicit_dayfirst_resolves_ambiguity():
    values = ["01/02/2021", "03/04/2021", "05/06/2021", "07/08/2021"]
    s = clean1(values, dayfirst=True, drop_duplicates=False)
    assert str(s.dtype).startswith("datetime64")
    assert pd.Timestamp("2021-02-01") in list(s)


def test_unparseable_garbage_still_coerces_not_blocks():
    """Plainly-invalid dates keep the existing quarantine behavior: the column
    converts and the garbage is coerced with originals preserved."""
    values = [f"2021-01-{d:02d}" for d in range(1, 20)] + ["not a date"]
    out, rep = fd.clean(
        pd.DataFrame({"v": values}), return_report=True, drop_duplicates=False,
    )
    assert str(out["v"].dtype).startswith("datetime64")
    assert rep.coerced_cells.get("v")


def test_time_range_strings_are_not_parsed_as_datetimes():
    """'2026-01-15 09:00-17:00' is a delivery window, not a timestamp;
    pandas misreads the '-17:00' as a UTC offset. The column must stay text
    (TruthBench logistics delivery_window)."""
    values = ["2026-01-15 09:00-17:00"] * 6 + ["2026-01-15 23:30-2026-01-16 01:00"]
    s = clean1(values, drop_duplicates=False)
    assert is_string(s.dtype)
    assert "2026-01-15 09:00-17:00" in s.tolist()


def test_plain_datetimes_still_convert_next_to_a_time():
    s = clean1(["2026-01-15 09:00", "2026-02-01 10:30", "2026-03-05 11:45"],
               drop_duplicates=False)
    assert str(s.dtype).startswith("datetime64")


# ── #447 / #451: object cells the pipeline must not choke on ────────────────────


def test_clean_keeps_undecodable_bytes_and_cleans_the_rest():
    # Regression (#447): pandas 2 decodes bytes when casting to StringDtype, so
    # one non-UTF-8 cell (a DB BLOB) aborted the whole clean. pandas 1.5
    # returned the frame with the cell untouched; both do that now.
    df = pd.DataFrame({"a": ["$12", b"\xff"], "n": [1, 2]})
    out = fd.clean(df, verbose=False)
    assert out["a"].tolist()[1] == b"\xff"
    assert out["n"].tolist() == [1, 2]


def test_clean_still_parses_ascii_bytes_columns():
    df = pd.DataFrame({"a": [b"ab", b"cd"], "n": [1, 2]})
    assert fd.clean(df, verbose=False)["a"].tolist() == [b"ab", b"cd"]


@pytest.mark.parametrize("extra", [{"y": [0, 0, 0, 0]}, {}])
def test_clean_accepts_booleans_mixed_with_nat(extra):
    # Regression (#451): BooleanArray rejects pd.NaT as a missing value, so
    # ["", NaT, False] raised TypeError("Need to pass bool-like values") — but
    # only when the frame had a second column, which changed inference order.
    df = pd.DataFrame({"x": [None, None, pd.NaT, False], **extra})
    out = fd.clean(df, verbose=False, drop_empty_rows=False)
    assert out["x"].isna().tolist()[:3] == [True, True, True]
    assert bool(out["x"].tolist()[3]) is False


def test_boolean_columns_without_missing_values_still_convert():
    df = pd.DataFrame({"x": [True, False, True, False]})
    assert str(fd.clean(df, verbose=False)["x"].dtype) in {"bool", "boolean"}
