import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata._util import PANDAS_MAJOR


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


def _plain(values):
    return [None if pd.isna(v) else v for v in values]


def _column_steps(report, column):
    return [(a.step, a.count) for a in report if a.column == column]


@pytest.mark.skipif(PANDAS_MAJOR < 2, reason="pd.ArrowDtype strings need pandas >= 2")
@pytest.mark.parametrize(
    "values",
    [
        [" a ", "N/A", "3", "4"],  # text: strip + sentinel
        ["1", " 2 ", "N/A", "4"],  # numeric-looking: fix_dtypes converts it
        ["2024-01-01", " 2024-02-01", None, "2024-03-01"],  # dates
    ],
)
def test_arrow_string_column_cleans_like_string_pyarrow(values):
    pa = pytest.importorskip("pyarrow")
    arrow = pd.DataFrame(
        {"s": pd.Series(values, dtype=pd.ArrowDtype(pa.string())), "k": [1.0, 2.0, 3.0, 4.0]}
    )
    string = arrow.astype({"s": "string[pyarrow]"})
    out_arrow, report_arrow = fd.clean(arrow, return_report=True, verbose=False)
    out_string, report_string = fd.clean(string, return_report=True, verbose=False)
    assert _plain(out_arrow["s"].astype(object)) == _plain(out_string["s"].astype(object))
    assert _column_steps(report_arrow, "s") == _column_steps(report_string, "s")
    assert ("strip_whitespace", 1) in _column_steps(report_arrow, "s")


def test_categorical_text_is_normalized_and_stays_categorical():
    cat = pd.Categorical(
        [" a ", "N/A", "b", "null", "a"],
        categories=["a", " a ", "N/A", "b", "null"],
        ordered=True,
    )
    df = pd.DataFrame({"c": cat, "k": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out, report = fd.clean(
        df, strategy="conservative", return_report=True, verbose=False,
        drop_empty_rows=False, drop_duplicates=False,
    )
    assert isinstance(out["c"].dtype, pd.CategoricalDtype)
    assert out["c"].cat.ordered
    assert list(out["c"].cat.categories) == ["a", "b"]  # " a " merged, sentinels gone
    assert _plain(out["c"]) == ["a", None, "b", None, "a"]
    counts = dict(_column_steps(report, "c"))
    assert counts["strip_whitespace"] == 1
    assert counts["normalize_sentinels"] == 2


def test_categorical_values_match_object_column():
    values = [" a ", "N/A", "b", "null"]
    cat = pd.DataFrame({"c": pd.Categorical(values), "k": [1.0, 2.0, 3.0, 4.0]})
    out_cat = fd.clean(cat, verbose=False)
    out_obj = fd.clean(cat.astype({"c": object}), verbose=False)
    assert isinstance(out_cat["c"].dtype, pd.CategoricalDtype)
    assert _plain(out_cat["c"].astype(object)) == _plain(out_obj["c"])


def test_clean_handles_missing_values_next_to_a_container_cell():
    # Regression (#448): the strip pass counted repairs with stripped.ne(s),
    # whose flex comparison hands object columns to NumPy, which calls bool()
    # on pd.NA != pd.NA and raises "boolean value of NA is ambiguous". A list
    # cell keeps the column object-dtype, which is how JSON data arrives.
    df = pd.DataFrame({"a": [pd.NA, []], "keep": [1, 2]})
    out = fd.clean(df, verbose=False)
    assert out["keep"].tolist() == [1, 2]
    assert [] in out["a"].tolist()


@pytest.mark.parametrize("cell", [[], {"k": 1}, {1, 2}, (1,)])
def test_text_repair_leaves_container_cells_untouched(cell):
    df = pd.DataFrame({"a": [cell, "  padded  ", pd.NA], "n": [1, 2, 3]})
    out = fd.clean(df, verbose=False, drop_empty_rows=False)
    values = out["a"].tolist()
    assert values[0] == cell  # containers are never rewritten
    assert values[1] == "padded"  # ordinary text is still stripped


def test_repair_counts_ignore_untouched_container_cells():
    df = pd.DataFrame({"a": [[1], "  x  ", "  y  "], "n": [1, 2, 3]})
    _, report = fd.clean(df, verbose=False, return_report=True, drop_empty_rows=False)
    stripped = [
        a for a in report.actions if a.step == "strip_whitespace" and a.column == "a"
    ]
    assert stripped and all(a.count == 2 for a in stripped)  # the list cell is not counted
