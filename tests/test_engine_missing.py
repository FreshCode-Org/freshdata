"""Missing-value decision engine: bands, roles, indicators, KNN."""

import numpy as np
import pandas as pd
import pytest

import freshdata as fd

# The engine acts beyond the low band only on frames with >= 30 rows; tests
# build 100-row frames so every band is reachable. Row-level default steps
# are disabled where they would interfere with the column under test.
N = 100
ISOLATE = {"drop_duplicates": False, "drop_empty_rows": False, "verbose": False}
AGGRESSIVE = {**ISOLATE, "strategy": "aggressive"}


def base_frame(rng=None):
    rng = rng or np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(50, 5, N),
        "z": rng.normal(0, 1, N),
    })


def with_missing(values, n_missing, rng_seed=1):
    s = pd.Series(list(values))
    idx = np.random.default_rng(rng_seed).choice(N, size=n_missing, replace=False)
    s.iloc[idx] = None
    return s


def engine_actions(report, column):
    return [a for a in report if a.step == "missing" and a.column == column]


# -- low missingness ----------------------------------------------------------

def test_low_band_normal_numeric_uses_mean():
    rng = np.random.default_rng(0)
    df = base_frame(rng)
    df["v"] = with_missing(rng.normal(100, 10, N), 3)
    out, report = fd.clean(df, report=True, **ISOLATE)
    assert out["v"].isna().sum() == 0
    [action] = engine_actions(report, "v")
    assert "mean" in action.description
    assert action.confidence > 0.8


def test_low_band_skewed_numeric_uses_median():
    rng = np.random.default_rng(0)
    df = base_frame(rng)
    df["v"] = with_missing(rng.lognormal(3, 1.5, N), 3)
    out, report = fd.clean(df, report=True, outlier_action=None, **ISOLATE)
    assert out["v"].isna().sum() == 0
    [action] = engine_actions(report, "v")
    assert "median" in action.description
    assert "skew" in action.rationale or "outlier" in action.rationale


def test_low_band_categorical_mode_with_clear_majority():
    df = base_frame()
    df["c"] = with_missing(["red"] * 80 + ["blue"] * 20, 4)
    out, report = fd.clean(df, report=True, **ISOLATE)
    [action] = engine_actions(report, "c")
    assert "mode" in action.description
    assert (out["c"] == "red").sum() >= 80


def test_low_band_categorical_no_majority_uses_unknown():
    values = [f"cat_{i % 4}" for i in range(N)]  # 25% each — no majority
    df = base_frame()
    df["c"] = with_missing(values, 4)
    out, report = fd.clean(df, report=True, **ISOLATE)
    assert (out["c"] == "Unknown").sum() == 4
    [action] = engine_actions(report, "c")
    assert "Unknown" in action.description


def test_low_band_datetime_ffill_when_time_ordered():
    df = base_frame()
    dates = pd.date_range("2024-01-01", periods=N, freq="D").to_series().reset_index(drop=True)
    dates.iloc[10] = None
    df["when"] = dates
    out, report = fd.clean(df, report=True, **ISOLATE)
    assert out["when"].isna().sum() == 0
    [action] = engine_actions(report, "when")
    assert "forward/backward" in action.description


def test_datetime_without_time_order_is_preserved():
    rng = np.random.default_rng(3)
    df = base_frame(rng)
    shuffled = pd.Series(pd.date_range("2024-01-01", periods=N, freq="D")
                         ).sample(frac=1, random_state=0).reset_index(drop=True)
    shuffled.iloc[5] = None
    df["when"] = shuffled
    out, report = fd.clean(df, report=True, **ISOLATE)
    assert out["when"].isna().sum() == 1
    [action] = engine_actions(report, "when")
    assert "preserved" in action.description
    assert "time order" in action.rationale


# -- medium missingness -------------------------------------------------------

def test_medium_band_numeric_uses_median_by_default():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"v": with_missing(rng.normal(10, 2, N), 20)})
    out, report = fd.clean(df, report=True, **ISOLATE)
    assert out["v"].isna().sum() == 0
    [action] = engine_actions(report, "v")
    assert "median" in action.description or "KNN" in action.description


def test_medium_band_knn_used_with_correlated_features():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, N)
    df = pd.DataFrame({
        "a": x + rng.normal(0, 0.1, N),
        "b": 2 * x + rng.normal(0, 0.1, N),
        "v": with_missing(3 * x + rng.normal(0, 0.1, N), 15),
    })
    out, report = fd.clean(df, report=True, **AGGRESSIVE)
    assert out["v"].isna().sum() == 0
    [action] = engine_actions(report, "v")
    assert "KNN" in action.description
    assert action.rationale  # never a blind fill


def test_medium_band_categorical_sentinel_without_dominant_value():
    values = [f"cat_{i % 5}" for i in range(N)]
    df = base_frame()
    df["c"] = with_missing(values, 20)
    out, report = fd.clean(df, report=True, **ISOLATE)
    assert (out["c"] == "Missing").sum() == 20
    [action] = [a for a in engine_actions(report, "c") if "filled" in a.description]
    assert action.risk == "medium"


# -- high and extreme missingness ---------------------------------------------

def test_high_band_uninformative_column_is_dropped():
    rng = np.random.default_rng(0)
    df = base_frame(rng)
    df["sparse"] = with_missing(rng.normal(0, 1, N), 45)
    out, report = fd.clean(df, report=True, **AGGRESSIVE)
    assert "sparse" not in out.columns
    assert "sparse" in report.columns_dropped
    [action] = engine_actions(report, "sparse")
    assert "dropped" in action.description and action.rationale


def test_high_band_preserved_column_is_kept_and_imputed_with_warning():
    rng = np.random.default_rng(0)
    df = base_frame(rng)
    df["sparse"] = with_missing(rng.normal(0, 1, N), 45)
    out, report = fd.clean(df, report=True, preserve_columns=("sparse",), **AGGRESSIVE)
    assert "sparse" in out.columns
    assert out["sparse"].isna().sum() == 0
    assert any("sparse" in w for w in report.warnings)
    fill = [a for a in engine_actions(report, "sparse") if a.count]
    assert fill and fill[-1].risk == "high" and fill[-1].confidence <= 0.5


def test_extreme_band_dropped_by_default():
    df = base_frame()
    df["gone"] = with_missing(["x"] * N, 70)
    out, report = fd.clean(df, report=True, **AGGRESSIVE)
    assert "gone" not in out.columns
    assert any("gone" in r for r in report.recommendations)


def test_extreme_band_target_column_is_never_dropped_or_filled():
    df = base_frame()
    df["target"] = with_missing([0, 1] * (N // 2), 70)
    out, report = fd.clean(df, report=True, **ISOLATE)
    assert "target" in out.columns
    assert out["target"].isna().sum() == 70
    [action] = engine_actions(report, "target")
    assert "preserved" in action.description
    assert any("target" in w for w in report.warnings)


# -- roles ---------------------------------------------------------------------

def test_id_column_is_never_imputed():
    df = base_frame()
    df["customer_id"] = with_missing(range(N), 3)
    out, report = fd.clean(df, report=True, **ISOLATE)
    assert out["customer_id"].isna().sum() == 3
    [action] = engine_actions(report, "customer_id")
    assert "preserved" in action.description
    assert any("customer_id" in r for r in report.recommendations)


def test_free_text_column_is_never_force_filled():
    texts = [f"this is a longer free text comment number {i} with details" for i in range(N)]
    df = base_frame()
    df["comment"] = with_missing(texts, 10)
    out, report = fd.clean(df, report=True, **ISOLATE)
    assert out["comment"].isna().sum() == 10
    [action] = engine_actions(report, "comment")
    assert "free-text" in action.rationale


def test_tiny_dataset_preserves_and_recommends_review():
    df = pd.DataFrame({"a": [1.0, None, None, 4.0, 5.0]})  # 40% missing, 5 rows
    out, report = fd.clean(df, report=True, **ISOLATE)
    assert out["a"].isna().sum() == 2
    [action] = engine_actions(report, "a")
    assert "too small" in action.rationale
    assert report.recommendations


# -- indicators -----------------------------------------------------------------

def test_indicator_added_when_missingness_is_informative():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, N)
    v = pd.Series(rng.normal(10, 1, N))
    v[x > 0.8] = None  # missing only where x is large: informative
    df = pd.DataFrame({"x": x, "v": v})
    out, report = fd.clean(df, report=True, **ISOLATE)
    assert "v_was_missing" in out.columns
    assert out["v_was_missing"].sum() == int(v.isna().sum())
    assert out["v"].isna().sum() == 0


def test_indicator_forced_on_and_off():
    rng = np.random.default_rng(0)
    df = base_frame(rng)
    df["v"] = with_missing(rng.normal(0, 1, N), 12)
    on = fd.clean(df, missing_indicators=True, **ISOLATE)
    assert "v_was_missing" in on.columns
    off = fd.clean(df, missing_indicators=False, **ISOLATE)
    assert "v_was_missing" not in off.columns


def test_explicit_impute_overrides_engine():
    rng = np.random.default_rng(0)
    df = base_frame(rng)
    df["sparse"] = with_missing(rng.normal(0, 1, N), 70)  # extreme band
    out = fd.clean(df, impute="median", **ISOLATE)
    assert "sparse" in out.columns  # engine drop suppressed by explicit choice
    assert out["sparse"].isna().sum() == 0
