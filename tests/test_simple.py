"""Tests for the minimal function-first Core surface (freshdata/simple.py)."""

import numpy as np
import pandas as pd
import pytest

import freshdata as fd

# --------------------------------------------------------------------------- #
# fill_missing
# --------------------------------------------------------------------------- #

def test_fill_missing_auto_numeric_uses_median_and_categorical_uses_mode():
    df = pd.DataFrame({"n": [1.0, 2.0, np.nan, 4.0], "c": ["a", "a", None, "b"]})
    out = fd.fill_missing(df)
    assert out["n"].tolist() == [1.0, 2.0, 2.0, 4.0]  # median of 1,2,4 == 2
    assert out["c"].tolist() == ["a", "a", "a", "b"]  # mode == "a"


def test_fill_missing_mean_and_median_and_mode():
    df = pd.DataFrame({"n": [1.0, 2.0, np.nan, 5.0]})
    assert fd.fill_missing(df, method="mean")["n"].iloc[2] == pytest.approx(8 / 3)
    assert fd.fill_missing(df, method="median")["n"].iloc[2] == 2.0
    cat = pd.DataFrame({"c": ["x", "x", None, "y"]})
    assert fd.fill_missing(cat, method="mode")["c"].iloc[2] == "x"


def test_fill_missing_constant_requires_value():
    df = pd.DataFrame({"n": [1.0, np.nan]})
    with pytest.raises(ValueError, match="constant"):
        fd.fill_missing(df, method="constant")
    out = fd.fill_missing(df, method="constant", value=-1)
    assert out["n"].tolist() == [1.0, -1.0]


def test_fill_missing_ffill_and_bfill():
    df = pd.DataFrame({"n": [1.0, np.nan, np.nan, 4.0]})
    assert fd.fill_missing(df, method="ffill")["n"].tolist() == [1.0, 1.0, 1.0, 4.0]
    assert fd.fill_missing(df, method="bfill")["n"].tolist() == [1.0, 4.0, 4.0, 4.0]


def test_fill_missing_column_subset_and_skips_full_columns():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    out = fd.fill_missing(df, columns="a", method="constant", value=0)
    assert out["a"].tolist() == [1.0, 0.0]
    assert out["b"].isna().sum() == 1  # untouched


def test_fill_missing_mean_on_non_numeric_is_skipped():
    df = pd.DataFrame({"c": ["a", None]})
    out = fd.fill_missing(df, method="mean")
    assert out["c"].isna().sum() == 1  # mean undefined -> left as-is


def test_fill_missing_all_null_mode_column_left_alone():
    df = pd.DataFrame({"c": [None, None]})
    out = fd.fill_missing(df, method="mode")
    assert out["c"].isna().sum() == 2


def test_fill_missing_inplace_semantics():
    df = pd.DataFrame({"n": [1.0, np.nan]})
    snapshot = df.copy(deep=True)
    out_copy = fd.fill_missing(df, method="constant", value=0)
    pd.testing.assert_frame_equal(df, snapshot)  # not mutated
    assert out_copy is not df
    out_inplace = fd.fill_missing(df, method="constant", value=0, inplace=True)
    assert out_inplace is df
    assert df["n"].tolist() == [1.0, 0.0]


def test_fill_missing_bad_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        fd.fill_missing(pd.DataFrame({"n": [1]}), method="nope")


def test_fill_missing_verbose(capsys):
    fd.fill_missing(pd.DataFrame({"n": [1.0, np.nan]}), method="median", verbose=True)
    assert "fill_missing" in capsys.readouterr().out


def test_resolve_columns_missing_raises():
    with pytest.raises(KeyError, match="columns not found"):
        fd.fill_missing(pd.DataFrame({"a": [1]}), columns="ghost")


# --------------------------------------------------------------------------- #
# detect_outliers / remove_outliers
# --------------------------------------------------------------------------- #

def test_detect_outliers_iqr_flags_extreme():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
    mask = fd.detect_outliers(df)
    assert mask.dtype == bool
    assert mask.tolist() == [False, False, False, False, False, True]
    assert list(mask.index) == list(df.index)


def test_detect_outliers_zscore_uses_default_factor_three():
    df = pd.DataFrame({"x": [10, 10, 10, 10, 20]})
    # Tight threshold flags the 20; the default factor (3.0) is too loose to.
    assert bool(fd.detect_outliers(df, method="zscore", threshold=1.0).iloc[-1]) is True
    assert fd.detect_outliers(df, method="zscore").sum() == 0


def test_detect_outliers_threshold_override():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 12]})
    assert fd.detect_outliers(df, threshold=5.0).sum() == 0
    assert fd.detect_outliers(df, threshold=0.1).sum() >= 1


def test_detect_outliers_constant_column_no_flags():
    df = pd.DataFrame({"x": [5, 5, 5, 5]})
    assert fd.detect_outliers(df).sum() == 0


def test_detect_outliers_ignores_non_numeric_and_seed_noop():
    df = pd.DataFrame({"c": ["a", "b", "c"], "x": [1, 2, 100]})
    m1 = fd.detect_outliers(df, seed=1)
    m2 = fd.detect_outliers(df, seed=999)
    pd.testing.assert_series_equal(m1, m2)  # seed does not change result


def test_detect_outliers_ignores_boolean_columns():
    # Regression: is_numeric_dtype is True for bool, which triggered a
    # "numpy boolean subtract" TypeError during IQR math on bool columns.
    df = pd.DataFrame({"x": [1, 2, 3, 100], "flag": [True, False, True, False]})
    mask = fd.detect_outliers(df)
    assert mask.tolist() == [False, False, False, True]
    assert len(fd.remove_outliers(df)) == 3


def test_detect_outliers_bad_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        fd.detect_outliers(pd.DataFrame({"x": [1]}), method="nope")


def test_detect_outliers_verbose(capsys):
    fd.detect_outliers(pd.DataFrame({"x": [1, 2, 3, 1000]}), verbose=True)
    assert "detect_outliers" in capsys.readouterr().out


def test_remove_outliers_preserves_index_and_copies():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]}, index=[10, 11, 12, 13, 14, 15])
    snapshot = df.copy(deep=True)
    out = fd.remove_outliers(df)
    assert out.index.tolist() == [10, 11, 12, 13, 14]  # 15 dropped, index kept
    pd.testing.assert_frame_equal(df, snapshot)  # original untouched
    assert out is not df


def test_remove_outliers_inplace():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
    out = fd.remove_outliers(df, inplace=True)
    assert out is df
    assert len(df) == 5


def test_remove_outliers_bad_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        fd.remove_outliers(pd.DataFrame({"x": [1]}), method="nope")


def test_remove_outliers_verbose(capsys):
    fd.remove_outliers(pd.DataFrame({"x": [1, 2, 3, 1000]}), verbose=True)
    assert "remove_outliers" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# resolve_duplicates
# --------------------------------------------------------------------------- #

def test_resolve_duplicates_first_last_drop():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
    first = fd.resolve_duplicates(df, columns="a", method="first")
    assert first["b"].tolist() == ["x", "z"]
    last = fd.resolve_duplicates(df, columns="a", method="last")
    assert last["b"].tolist() == ["y", "z"]
    dropped = fd.resolve_duplicates(df, columns="a", method="drop")
    assert dropped["b"].tolist() == ["z"]  # both members of the dup group removed


def test_resolve_duplicates_whole_row_default():
    df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "y"]})
    out = fd.resolve_duplicates(df)
    assert len(out) == 2  # rows 0 and 1 identical -> one dropped


def test_resolve_duplicates_preserves_order():
    df = pd.DataFrame({"a": [3, 1, 3, 2]})
    out = fd.resolve_duplicates(df, columns="a")
    assert out["a"].tolist() == [3, 1, 2]


def test_resolve_duplicates_inplace_and_copy():
    df = pd.DataFrame({"a": [1, 1, 2]})
    snapshot = df.copy(deep=True)
    fd.resolve_duplicates(df, columns="a")  # copy path
    pd.testing.assert_frame_equal(df, snapshot)
    out = fd.resolve_duplicates(df, columns="a", inplace=True)
    assert out is df
    assert len(df) == 2


def test_resolve_duplicates_bad_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        fd.resolve_duplicates(pd.DataFrame({"a": [1]}), method="nope")


def test_resolve_duplicates_verbose(capsys):
    fd.resolve_duplicates(pd.DataFrame({"a": [1, 1]}), verbose=True)
    assert "resolve_duplicates" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# group_aggregate
# --------------------------------------------------------------------------- #

def test_group_aggregate_str_agg_numeric_default_columns():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 3, 10], "w": [2, 4, 20]})
    out = fd.group_aggregate(df, by="g", agg="sum")
    assert set(out.columns) == {"g", "v", "w"}
    row_a = out[out["g"] == "a"].iloc[0]
    assert row_a["v"] == 4 and row_a["w"] == 6


def test_group_aggregate_list_agg():
    df = pd.DataFrame({"g": ["a", "a"], "v": [1, 3]})
    out = fd.group_aggregate(df, by="g", agg=["min", "max"], columns="v")
    # MultiIndex columns: ("v","min"), ("v","max")
    assert out[("v", "min")].iloc[0] == 1
    assert out[("v", "max")].iloc[0] == 3


def test_group_aggregate_dict_agg():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 3, 10], "w": [5, 7, 9]})
    out = fd.group_aggregate(df, by="g", agg={"v": "sum", "w": "mean"})
    row_a = out[out["g"] == "a"].iloc[0]
    assert row_a["v"] == 4 and row_a["w"] == 6


def test_group_aggregate_multi_key():
    df = pd.DataFrame({"g1": ["a", "a", "a"], "g2": ["x", "x", "y"], "v": [1, 2, 3]})
    out = fd.group_aggregate(df, by=["g1", "g2"], agg="sum")
    assert len(out) == 2
    assert out.index.tolist() == [0, 1]  # index reset


def test_group_aggregate_requires_by():
    with pytest.raises(ValueError, match="by is required"):
        fd.group_aggregate(pd.DataFrame({"v": [1]}), by=[])


def test_group_aggregate_missing_by_raises():
    with pytest.raises(KeyError, match="group columns not found"):
        fd.group_aggregate(pd.DataFrame({"v": [1]}), by="ghost")


def test_group_aggregate_verbose(capsys):
    df = pd.DataFrame({"g": ["a", "b"], "v": [1, 2]})
    fd.group_aggregate(df, by="g", verbose=True)
    assert "group_aggregate" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# pipeline_clean
# --------------------------------------------------------------------------- #

def test_pipeline_clean_end_to_end_and_no_mutation():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan, 1000.0]})
    snapshot = df.copy(deep=True)
    out = fd.pipeline_clean(df)
    pd.testing.assert_frame_equal(df, snapshot)  # copy by default
    assert out["x"].isna().sum() == 0          # filled
    assert 1000.0 not in out["x"].tolist()     # outlier removed
    assert out["x"].duplicated().sum() == 0    # duplicates resolved


def test_pipeline_clean_is_deterministic():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan, 1000.0]})
    pd.testing.assert_frame_equal(fd.pipeline_clean(df), fd.pipeline_clean(df))


def test_pipeline_clean_inplace_returns_same_object():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan, 1000.0]})
    out = fd.pipeline_clean(df, inplace=True)
    assert out is df


def test_pipeline_clean_column_scoping():
    df = pd.DataFrame({"x": [1.0, np.nan, 2.0], "y": [np.nan, np.nan, np.nan]})
    out = fd.pipeline_clean(df, columns="x")
    assert out["x"].isna().sum() == 0
    assert out["y"].isna().sum() == out["y"].shape[0]  # y untouched
