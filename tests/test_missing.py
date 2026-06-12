import pandas as pd

import freshdata as fd

# NOTE: single-column fixtures interact with row-level steps (a missing cell
# makes an all-missing row; repeated values make duplicate rows), so these
# tests disable those steps to exercise imputation in isolation.
KEEP_ROWS = {"drop_empty_rows": False, "drop_duplicates": False}


def test_imputation_is_off_with_conservative_strategy():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})
    out = fd.clean(df, strategy="conservative", **KEEP_ROWS)
    assert out["a"].isna().sum() == 1
    assert out["b"].isna().sum() == 1


def test_remaining_nans_are_always_explained_by_default():
    # 1 of 3 missing on a tiny frame: the engine preserves rather than guesses,
    # but it must say so in the report — NaNs are never left silently.
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})
    out, report = fd.clean(df, report=True, **KEEP_ROWS)
    assert out["a"].isna().sum() == 1
    explained = {a.column for a in report if a.step == "missing" and a.rationale}
    assert {"a", "b"} <= explained


def test_median_imputation():
    df = pd.DataFrame({"a": [1.0, None, 3.0, 100.0]})
    out = fd.clean(df, impute="median", **KEEP_ROWS)
    assert out["a"].isna().sum() == 0
    assert out["a"].iloc[1] == 3.0  # median of [1, 3, 100]


def test_mean_imputation():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    out = fd.clean(df, impute="mean", **KEEP_ROWS)
    assert out["a"].iloc[1] == 2.0


def test_auto_uses_median_for_numbers_mode_for_text():
    df = pd.DataFrame({"n": [1.0, None, 5.0, 5.0], "t": ["x", "x", None, "y"]})
    out = fd.clean(df, impute="auto")
    assert out["n"].iloc[1] == 5.0
    assert out["t"].iloc[2] == "x"


def test_mean_skips_text_columns_with_note():
    df = pd.DataFrame({"t": ["x", None, "y"]})
    out, report = fd.clean(df, impute="mean", report=True, **KEEP_ROWS)
    assert out["t"].isna().sum() == 1  # unchanged
    assert any(a.step == "impute" and "skipped" in a.description for a in report)


def test_fractional_median_casts_integer_column():
    df = pd.DataFrame({"a": ["1", "2", None, None]})
    out, report = fd.clean(df, impute="median", report=True, **KEEP_ROWS)
    assert out["a"].dtype == "float64"  # median 1.5 cannot live in Int64
    assert out["a"].isna().sum() == 0
    assert any("cast to float64" in a.description for a in report)


def test_all_missing_column_is_skipped():
    df = pd.DataFrame({"a": [1, 2], "b": [None, None]})
    out = fd.clean(df, impute="auto", drop_empty_columns=False)
    assert out["b"].isna().all()


def test_imputation_counts_reported():
    df = pd.DataFrame({"a": [1.0, None, None, 4.0]})
    _, report = fd.clean(df, impute="median", report=True, **KEEP_ROWS)
    [action] = [a for a in report if a.step == "impute"]
    assert action.count == 2


def test_datetime_mode_imputation_via_auto():
    df = pd.DataFrame(
        {"d": pd.to_datetime(["2021-01-01", "2021-01-01", None, "2021-02-01"])}
    )
    out = fd.clean(df, impute="auto", **KEEP_ROWS)
    assert out["d"].isna().sum() == 0
    assert out["d"].iloc[2] == pd.Timestamp("2021-01-01")


def test_boolean_mode_imputation_via_auto():
    df = pd.DataFrame({"b": pd.array([True, True, None, False], dtype="boolean")})
    out = fd.clean(df, impute="auto", **KEEP_ROWS)
    assert out["b"].isna().sum() == 0
    assert bool(out["b"].iloc[2]) is True
