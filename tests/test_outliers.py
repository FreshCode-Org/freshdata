import warnings

import pandas as pd

import freshdata as fd

BASE = [10.0, 11.0, 12.0, 11.0, 10.0, 12.0, 11.0, 10.0, 12.0, 11.0]

# Single-column fixtures with repeated values form duplicate rows; disable
# deduplication so detection bounds are computed on the intended data.
ISOLATE = {"drop_duplicates": False}


def test_outliers_untouched_with_conservative_strategy():
    df = pd.DataFrame({"v": BASE + [1000.0]})
    out = fd.clean(df, strategy="conservative", **ISOLATE)
    assert out["v"].max() == 1000.0


def test_outliers_capped_by_default():
    df = pd.DataFrame({"v": BASE + [1000.0]})
    out = fd.clean(df, strategy="aggressive", **ISOLATE)
    assert out["v"].max() < 1000.0  # aggressive strategy winsorizes by default


def test_outliers_flagged_by_balanced_default():
    df = pd.DataFrame({"v": BASE + [1000.0]})
    out = fd.clean(df, **ISOLATE)
    assert out["v"].max() == 1000.0
    assert "v_outlier" in out.columns


def test_clip_iqr():
    df = pd.DataFrame({"v": BASE + [1000.0]})
    out, report = fd.clean(df, outliers="clip", return_report=True, **ISOLATE)
    assert out["v"].max() < 1000.0
    [action] = [a for a in report if a.step == "outliers"]
    assert action.count == 1 and "clipped" in action.description


def test_clip_zscore():
    df = pd.DataFrame({"v": BASE * 5 + [10_000.0]})
    out = fd.clean(df, outliers="clip", outlier_method="zscore", **ISOLATE)
    assert out["v"].max() < 10_000.0


def test_flag_adds_boolean_column_and_keeps_data():
    df = pd.DataFrame({"v": BASE + [1000.0]})
    out = fd.clean(df, outliers="flag", **ISOLATE)
    assert out["v"].max() == 1000.0  # data untouched
    assert out["v_outlier"].dtype == bool
    assert out["v_outlier"].sum() == 1
    assert bool(out["v_outlier"].iloc[-1])


def test_flag_name_collision_avoided():
    df = pd.DataFrame({"v": BASE + [1000.0], "v_outlier": ["x"] * 11})
    out = fd.clean(df, outliers="flag", **ISOLATE)
    assert "v_outlier_2" in out.columns


def test_integer_columns_stay_integer_after_clip():
    df = pd.DataFrame({"v": [int(x) for x in BASE] + [1000]})
    out = fd.clean(df, outliers="clip", **ISOLATE)
    assert out["v"].dtype == "int64"
    assert out["v"].max() < 1000


def test_constant_and_boolean_columns_skipped():
    df = pd.DataFrame({"c": [5.0] * 11, "b": [True, False] * 5 + [True],
                       "v": BASE + [1000.0]})
    out, report = fd.clean(df, outliers="clip", return_report=True, **ISOLATE)
    assert out["c"].tolist() == [5.0] * 11
    assert all(a.column == "v" for a in report if a.step == "outliers")


def test_custom_factor():
    df = pd.DataFrame({"v": BASE + [14.0]})
    loose = fd.clean(df, outliers="clip", outlier_factor=10.0, **ISOLATE)
    assert loose["v"].max() == 14.0  # wide fences: nothing clipped


def test_infinite_values_do_not_poison_fences_or_leak_warnings():
    # ±inf used to make the IQR quantiles NaN/infinite: detection silently
    # returned no fences and numpy's quantile interpolation leaked a
    # RuntimeWarning to the caller. Fences must come from the finite bulk,
    # with the infinities themselves detected as outliers.
    df = pd.DataFrame({"v": BASE + [float("inf"), float("-inf")]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = fd.clean(df, outliers="flag", **ISOLATE)
    assert "v_outlier" in out.columns
    assert out["v_outlier"].sum() == 2
    assert out.loc[out["v"] == float("inf"), "v_outlier"].all()


def test_infinite_values_flagged_by_zscore_path_too():
    df = pd.DataFrame({"v": BASE * 5 + [float("inf")]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = fd.clean(df, outliers="flag", outlier_method="zscore", **ISOLATE)
    assert "v_outlier" in out.columns
    assert bool(out["v_outlier"].iloc[-1])


def test_all_infinite_column_is_skipped_not_crashed():
    df = pd.DataFrame({"v": [float("inf"), float("-inf")] * 6})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out, report = fd.clean(df, outliers="clip", return_report=True, **ISOLATE)
    # no finite bulk -> no fences -> column left alone
    assert not [a for a in report if a.step == "outliers"]
