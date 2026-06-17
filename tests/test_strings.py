import numpy as np
import pandas as pd

import freshdata as fd


def test_whitespace_stripped_object_and_string_dtype():
    df = pd.DataFrame(
        {
            "obj": [" x ", "y\t", "z"],
            "str": pd.array([" a", "b ", "c"], dtype="string"),
        }
    )
    out = fd.clean(df)
    assert out["obj"].tolist() == ["x", "y", "z"]
    assert out["str"].tolist() == ["a", "b", "c"]


def test_internal_whitespace_preserved():
    df = pd.DataFrame({"city": ["New  York ", " San Francisco"]})
    out = fd.clean(df)
    assert out["city"].tolist() == ["New  York", "San Francisco"]


def test_mixed_type_column_numbers_survive():
    df = pd.DataFrame({"mix": [1, " keep ", 2.5, None]})
    out = fd.clean(df, fix_dtypes=False, drop_empty_rows=False)
    assert out["mix"].tolist()[:3] == [1, "keep", 2.5]


def test_sentinels_are_case_insensitive():
    df = pd.DataFrame({"v": ["NULL", "n/a", "None", "ok", "-", "#REF!"]})
    out = fd.clean(df, drop_empty_rows=False, drop_duplicates=False)
    assert out["v"].isna().sum() == 5
    assert out["v"].dropna().tolist() == ["ok"]


def test_extra_sentinels():
    df = pd.DataFrame({"v": ["unknown", "ok", "UNKNOWN "]})
    out = fd.clean(df, extra_sentinels=("unknown",), drop_empty_rows=False,
                   drop_duplicates=False)
    assert out["v"].isna().sum() == 2


def test_sentinel_only_when_entire_cell_matches():
    df = pd.DataFrame({"v": ["banana", "nathan", "na"]})
    out = fd.clean(df, drop_empty_rows=False)
    assert out["v"].isna().sum() == 1  # only the bare "na"


def test_steps_can_be_disabled():
    df = pd.DataFrame({"v": [" x ", "N/A"]})
    out = fd.clean(df, strip_whitespace=False, normalize_sentinels=False)
    assert out["v"].tolist() == [" x ", "N/A"]


def test_empty_string_becomes_missing():
    df = pd.DataFrame({"v": ["", "  ", "x"]})
    out = fd.clean(df, drop_empty_rows=False, drop_duplicates=False)
    assert out["v"].isna().sum() == 2


def test_empty_and_blank_values_are_reported_in_clean_report():
    df = pd.DataFrame({"v": ["", " ", "N/A", "x"]})
    _, report = fd.clean(
        df, return_report=True, drop_empty_rows=False, drop_duplicates=False
    )
    actions = [a for a in report if a.step == "normalize_sentinels"]
    assert len(actions) == 1
    assert actions[0].count == 3


def test_backslash_n_sentinel_is_normalized_as_missing():
    df = pd.DataFrame({"v": ["", " \\N ", "\\N", "x"]})
    out = fd.clean(df, drop_empty_rows=False, drop_duplicates=False)
    assert out["v"].isna().sum() == 3
    assert pd.isna(out["v"].iloc[2])


def test_partial_backslash_n_values_are_preserved():
    df = pd.DataFrame({"v": ["A\\N42", "\\N42", "42\\N", "x"]})
    out = fd.clean(df, drop_empty_rows=False, drop_duplicates=False)
    assert out["v"].tolist() == ["A\\N42", "\\N42", "42\\N", "x"]


def test_unhashable_values_pass_through():
    df = pd.DataFrame({"v": [[1, 2], [3], None], "w": ["a", "b", "c"]})
    out = fd.clean(df)
    assert out["v"].iloc[0] == [1, 2]
    assert not np.any(out["w"].isna())
