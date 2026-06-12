import datetime as dt

import numpy as np
import pandas as pd
import pytest

import freshdata as fd


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
    _, report = fd.clean(df, report=True)
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
