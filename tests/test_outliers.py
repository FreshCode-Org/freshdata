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
    out, report = fd.clean(df, outliers="clip", report=True, **ISOLATE)
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
    out, report = fd.clean(df, outliers="clip", report=True, **ISOLATE)
    assert out["c"].tolist() == [5.0] * 11
    assert all(a.column == "v" for a in report if a.step == "outliers")


def test_custom_factor():
    df = pd.DataFrame({"v": BASE + [14.0]})
    loose = fd.clean(df, outliers="clip", outlier_factor=10.0, **ISOLATE)
    assert loose["v"].max() == 14.0  # wide fences: nothing clipped
