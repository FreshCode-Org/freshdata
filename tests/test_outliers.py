import math
import warnings

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.steps.outliers import (
    MEANAD_TO_IQR,
    ZERO_IQR_MAX_SHARE,
    ZERO_IQR_NOTE,
    detection_bounds,
    detection_fences,
)

BASE = [10.0, 11.0, 12.0, 11.0, 10.0, 12.0, 11.0, 10.0, 12.0, 11.0]

# Single-column fixtures with repeated values form duplicate rows; disable
# deduplication so detection bounds are computed on the intended data.
ISOLATE = {"drop_duplicates": False}


def test_outliers_untouched_with_conservative_strategy():
    df = pd.DataFrame({"v": BASE + [1000.0]})
    out = fd.clean(df, strategy="conservative", **ISOLATE)
    assert out["v"].max() == 1000.0


def test_outliers_flagged_by_aggressive_default():
    # Audit P1-6: "auto" never rewrites values — aggressive flags, like balanced.
    df = pd.DataFrame({"v": BASE + [1000.0]})
    out = fd.clean(df, strategy="aggressive", **ISOLATE)
    assert out["v"].max() == 1000.0
    assert "v_outlier" in out.columns


def test_outliers_capped_on_explicit_request():
    df = pd.DataFrame({"v": BASE + [1000.0]})
    out = fd.clean(df, strategy="aggressive", outlier_action="cap", **ISOLATE)
    assert out["v"].max() < 1000.0  # capping is explicit-only


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


# -- zero IQR on a non-constant column ---------------------------------------
# Q1 == Q3 used to return no fences, silently disabling IQR detection on
# mostly-zero columns with real spikes (while zscore still flagged one).

SPIKES = [0.0] * 95 + [1000.0, 5000.0, -800.0]
QUIET = {**ISOLATE, "verbose": False}


def _zero_iqr_fence(values, factor=1.5):
    mean_ad = float(np.abs(np.asarray(values)).mean())  # median is 0
    return factor * (MEANAD_TO_IQR * mean_ad)


def _outlier_actions(report):
    return [a for a in report if a.step == "outliers"]


@pytest.mark.parametrize("method", ["iqr", "auto"])
def test_zero_iqr_spikes_are_flagged(method):
    df = pd.DataFrame({"x": SPIKES})
    out, report = fd.clean(df, outliers="flag", outlier_method=method,
                           return_report=True, **QUIET)
    assert out["x"].tolist() == SPIKES  # data untouched
    assert out.loc[out["x_outlier"], "x"].tolist() == [1000.0, 5000.0, -800.0]
    [action] = _outlier_actions(report)
    assert action.count == 3 and report.outliers_handled == 3
    assert ZERO_IQR_NOTE in action.description


@pytest.mark.parametrize("method", ["iqr", "auto"])
def test_zero_iqr_spikes_are_clipped_to_fallback_fences(method):
    df = pd.DataFrame({"x": SPIKES})
    out, report = fd.clean(df, outliers="clip", outlier_method=method,
                           return_report=True, **QUIET)
    fence = _zero_iqr_fence(SPIKES)
    assert out["x"].tolist() == [0.0] * 95 + [fence, fence, -fence]
    [action] = _outlier_actions(report)
    assert action.count == 3 and ZERO_IQR_NOTE in action.description


def test_zero_iqr_small_deviations_are_not_outliers():
    # The spikes widen the fences, so 2 and 3 stay inliers.
    values = [0.0] * 95 + [1000.0, 5000.0, 2.0, 3.0, -800.0]
    out = fd.clean(pd.DataFrame({"x": values}), outliers="flag", **QUIET)
    assert out.loc[out["x_outlier"], "x"].tolist() == [1000.0, 5000.0, -800.0]


def test_zero_iqr_engine_actions_record_the_fallback():
    df = pd.DataFrame({"x": SPIKES})
    flagged, report = fd.clean(df, return_report=True, **QUIET)  # balanced default
    assert int(flagged["x_outlier"].sum()) == 3
    assert ZERO_IQR_NOTE in _outlier_actions(report)[0].description

    capped = fd.clean(df, outlier_action="cap", **QUIET)
    fence = _zero_iqr_fence(SPIKES)
    assert capped["x"].max() == fence and capped["x"].min() == -fence

    removed = fd.clean(df, outlier_action="remove", **QUIET)
    assert removed["x"].tolist() == [0.0] * 95


def test_zero_iqr_integer_column_stays_integer_after_clip():
    df = pd.DataFrame({"x": [int(v) for v in SPIKES]})
    out = fd.clean(df, outliers="clip", **QUIET)
    assert out["x"].dtype == "int64"
    assert out["x"].max() == math.ceil(_zero_iqr_fence(SPIKES))


def test_zero_iqr_nullable_column_with_missing_values():
    # The second column keeps the missing-x row from being an empty row.
    df = pd.DataFrame({"x": pd.array([int(v) for v in SPIKES] + [None], dtype="Int64"),
                       "k": [f"r{i}" for i in range(len(SPIKES) + 1)]})
    out = fd.clean(df, outliers="flag", strategy="conservative", **QUIET)  # no engine imputation
    assert int(out["x_outlier"].sum()) == 3
    assert out["x"].isna().iloc[-1] and not bool(out["x_outlier"].iloc[-1])


@pytest.mark.parametrize(
    "values",
    [
        # 30% non-zero, split around zero so the IQR is still zero.
        [0.0] * 70 + [float(s * i) for i in range(1, 16) for s in (-1, 1)],
        # 8% small non-zero values: a second mode, not rare spikes.
        [0.0] * 92 + [float(i) for i in range(1, 9)],
    ],
    ids=["two_sided_30pct", "one_sided_8pct"],
)
def test_zero_iqr_second_mode_is_not_flagged_wholesale(values):
    s = pd.Series(values)
    assert s.quantile(0.25) == s.quantile(0.75)  # the case under test
    df = pd.DataFrame({"x": values})
    for options in ({"outliers": "flag"}, {"outliers": "clip"}, {},
                    {"outlier_action": "cap"}):
        out, report = fd.clean(df, return_report=True, **options, **QUIET)
        assert not _outlier_actions(report), options
        assert out["x"].tolist() == values
        assert "x_outlier" not in out.columns


def test_zero_iqr_fallback_share_limit_is_inclusive():
    at_limit = [0.0] * 95 + [1000.0] * 5  # 5% flagged: allowed
    over = [0.0] * 94 + [1000.0] * 6  # 6% flagged: a second mode
    assert detection_fences(pd.Series(at_limit), "iqr", 1.5) is not None
    assert detection_fences(pd.Series(over), "iqr", 1.5) is None
    assert ZERO_IQR_MAX_SHARE == 0.05


def test_constant_column_still_untouched_by_every_action():
    df = pd.DataFrame({"c": [0.0] * 98})
    for options in ({"outliers": "flag"}, {"outliers": "clip"}, {},
                    {"outlier_action": "cap"}, {"outlier_action": "remove"}):
        out, report = fd.clean(df, return_report=True, **options, **QUIET)
        assert out["c"].tolist() == [0.0] * 98
        assert list(out.columns) == ["c"]
        assert not _outlier_actions(report), options
    assert detection_fences(pd.Series([7.0] * 10 + [np.nan]), "iqr", 1.5) is None


def test_non_zero_iqr_fences_are_unchanged():
    rng = np.random.default_rng(3)
    s = pd.Series(np.r_[rng.lognormal(0, 1, 500), 400.0])
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    expected = (float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1)))
    assert detection_fences(s, "iqr", 1.5) == (*expected, "")
    assert detection_bounds(s, "iqr", 1.5) == expected
    _, report = fd.clean(pd.DataFrame({"v": s}), outliers="flag",
                         return_report=True, **QUIET)
    [action] = _outlier_actions(report)
    assert action.description.endswith("(iqr, factor 1.5)")


def test_zero_iqr_profile_reports_the_spikes():
    prof = fd.profile(pd.DataFrame({"x": SPIKES}))
    [col] = [c for c in prof.columns if c.name == "x"]
    assert "3 potential outlier(s) (iqr)" in col.issues
