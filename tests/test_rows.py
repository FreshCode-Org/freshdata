import pandas as pd
import pytest

import freshdata as fd


def test_empty_rows_dropped_and_reported():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
    out, report = fd.clean(df, return_report=True)
    assert len(out) == 2
    assert any(a.step == "drop_empty_rows" and a.count == 1 for a in report)


def test_empty_columns_dropped():
    df = pd.DataFrame({"a": [1, 2], "b": [None, None]})
    out = fd.clean(df)
    assert list(out.columns) == ["a"]


def test_empty_pruning_can_be_disabled():
    df = pd.DataFrame({"a": [1, None], "b": [None, None]})
    out = fd.clean(df, drop_empty_rows=False, drop_empty_columns=False)
    assert out.shape == (2, 2)


def test_constant_columns_kept_by_default_dropped_on_request():
    df = pd.DataFrame({"a": [1, 2, 3], "c": ["same", "same", "same"]})
    assert "c" in fd.clean(df).columns
    out = fd.clean(df, drop_constant_columns=True)
    assert list(out.columns) == ["a"]


def test_duplicates_dropped_keep_first():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out = fd.clean(df, drop_duplicates=True)
    assert len(out) == 2
    assert out.index.tolist() == [0, 2]  # original labels kept by default


def test_duplicate_subset():
    df = pd.DataFrame({"id": [1, 1, 2], "note": ["a", "b", "c"]})
    out = fd.clean(df, drop_duplicates=True, duplicate_subset=("id",))
    assert len(out) == 2


def test_duplicate_subset_unknown_column_raises():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="duplicate_subset"):
        fd.clean(df, duplicate_subset=("nope",))


def test_unhashable_rows_skip_duplicates_with_note():
    # Multi-column duplicated() raises TypeError on unhashable cells; the
    # step must skip with a note instead of crashing.
    df = pd.DataFrame({"v": [[1], [1]], "w": [1, 2]})
    out, report = fd.clean(df, return_report=True)
    assert len(out) == 2  # nothing dropped
    assert any(a.step == "drop_duplicates" and "unhashable" in a.description
               for a in report)


def test_reset_index_opt_in():
    df = pd.DataFrame({"a": [1, 1, 2]})
    out = fd.clean(df, drop_duplicates=True, reset_index=True)
    assert out.index.tolist() == [0, 1]


def test_typed_duplicates_found_after_conversion():
    # "1.0" and "1" are different strings but the same number; duplicates are
    # detected after dtype fixing, so these rows collapse.
    df = pd.DataFrame({"v": ["1.0", "1", "2"]})
    out = fd.clean(df, drop_duplicates=True)
    assert out["v"].tolist() == [1, 2]


def test_leading_zero_ids_are_preserved_not_coerced():
    # "01"/"007"/ZIP codes are identifiers — coercing them to int would destroy
    # the padding, so they are kept as text (preserve_leading_zeros, default).
    df = pd.DataFrame({"v": ["01", "007", "02115"]})
    out = fd.clean(df, drop_duplicates=False)
    assert out["v"].tolist() == ["01", "007", "02115"]
    # opting out restores numeric coercion
    out2 = fd.clean(df, drop_duplicates=False, preserve_leading_zeros=False)
    assert out2["v"].tolist() == [1, 7, 2115]
