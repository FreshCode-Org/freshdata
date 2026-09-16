"""Unit tests for the model selection router."""

import numpy as np
import pandas as pd
import pytest

from freshdata.config import CleanConfig
from freshdata.engine.context import build_context
from freshdata.engine.model_select import (
    _partner_info,
    rank_missing_models,
    select_outlier_action,
)


def _ctx(df, col, **config_kw):
    cfg = CleanConfig(**config_kw)
    return build_context(df, col, cfg)


def test_normal_numeric_low_missing_prefers_mean():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"v": rng.normal(10, 1, 100)})
    df.loc[:2, "v"] = np.nan
    ctx = _ctx(df, "v")
    sel = rank_missing_models(df, "v", ctx, CleanConfig(), mode="balanced")
    assert sel.primary.model_id == "mean"
    assert any(a.model_id == "median" for a in sel.alternatives)


def test_skewed_numeric_prefers_median():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"v": rng.lognormal(0, 1.5, 100)})
    df.loc[:2, "v"] = np.nan
    ctx = _ctx(df, "v")
    sel = rank_missing_models(df, "v", ctx, CleanConfig(), mode="balanced")
    assert sel.primary.model_id == "median"


def test_categorical_no_majority_prefers_sentinel():
    df = pd.DataFrame({"c": [f"x{i % 4}" for i in range(100)]})
    df.loc[:10, "c"] = np.nan
    ctx = _ctx(df, "c")
    sel = rank_missing_models(df, "c", ctx, CleanConfig(), mode="balanced")
    assert sel.primary.model_id == "sentinel"


def test_datetime_monotonic_prefers_time_fill():
    df = pd.DataFrame({
        "t": pd.date_range("2024-01-01", periods=100, freq="D"),
    })
    df.loc[5, "t"] = pd.NaT
    ctx = _ctx(df, "t")
    sel = rank_missing_models(df, "t", ctx, CleanConfig(), mode="balanced")
    assert sel.primary.model_id == "time_fill"


def test_aggressive_medium_missing_may_use_knn():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    v = pd.Series(3 * x)
    v.iloc[:15] = np.nan
    df = pd.DataFrame({
        "a": x,
        "b": 2 * x,
        "v": v,
    })
    ctx = _ctx(df, "v")
    sel = rank_missing_models(df, "v", ctx, CleanConfig(), mode="aggressive")
    assert sel.primary.model_id in ("knn", "median", "partner_median", "linear")


def test_balanced_disqualifies_knn():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    v = pd.Series(3 * x)
    v.iloc[:15] = np.nan
    df = pd.DataFrame({
        "a": x,
        "b": 2 * x,
        "v": v,
    })
    ctx = _ctx(df, "v")
    sel = rank_missing_models(df, "v", ctx, CleanConfig(), mode="balanced")
    assert sel.primary.model_id != "knn"
    knn = [c for c in sel.alternatives if c.model_id == "knn"]
    if knn:
        assert not knn[0].eligible


def test_domain_column_outlier_preserve():
    ctx = _ctx(pd.DataFrame({"fraud_score": [0.1, 0.9, 100.0]}), "fraud_score")
    action, choice = select_outlier_action(ctx, CleanConfig(), mode="balanced", share=0.33)
    assert action is None
    assert choice.model_id == "preserve"


def test_heavy_tail_prefers_flag():
    values = np.concatenate([np.random.default_rng(0).normal(0, 1, 300), [100.0]])
    ctx = _ctx(pd.DataFrame({"measurement": values}), "measurement")
    action, choice = select_outlier_action(ctx, CleanConfig(), mode="aggressive", share=0.20)
    assert action == "flag"
    assert choice.model_id == "flag"


def test_router_deterministic():
    df = pd.DataFrame({"v": [1.0, 2.0, np.nan, 4.0] * 25})
    ctx = _ctx(df, "v")
    a = rank_missing_models(df, "v", ctx, CleanConfig(), mode="balanced")
    b = rank_missing_models(df, "v", ctx, CleanConfig(), mode="balanced")
    assert a.primary.model_id == b.primary.model_id
    assert [x.model_id for x in a.alternatives] == [x.model_id for x in b.alternatives]


def test_partner_info_survives_degenerate_columns():
    """FDC-L3-029: a non-finite float beside a single-valued nullable Int64.

    On the py3.9 lane (pandas 1.5.3 / numpy 1.26.4) the raw ``corrwith`` blew up
    with ``AttributeError: 'float' object has no attribute 'shape'``; an unusable
    correlation must read as "no partner", never abort the caller.
    """
    df = pd.DataFrame({
        "f": [np.inf, -np.inf, 1e308, -1e308, 5e-324, np.nan, 3.0, -2.5],
        "i": pd.array([None] * 7 + [-9223372036854775775], dtype="Int64"),
    })
    partners, corr = _partner_info(df, "f")
    assert partners == []
    assert corr is not None and pd.isna(corr["i"])
    partners, corr = _partner_info(df, "i")
    assert partners == []
    assert corr is not None and pd.isna(corr["f"])


def test_partner_info_ignores_constant_and_all_inf_columns():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 60)
    df = pd.DataFrame({
        "v": x,
        "const": np.ones(60),
        "infs": np.full(60, np.inf),
        "one_finite": np.concatenate([[1.0], np.full(59, np.nan)]),
    })
    partners, corr = _partner_info(df, "v")
    assert partners == []
    assert corr is not None
    assert all(pd.isna(corr[c]) for c in ("const", "infs", "one_finite"))


def test_partner_info_matches_pandas_on_ordinary_numeric_data():
    """The guard must not move correlations for plain finite numeric columns."""
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 120)
    df = pd.DataFrame({
        "v": 3 * x + rng.normal(0, 0.2, 120),
        "close": 2 * x,
        "loose": x + rng.normal(0, 5, 120),
        "noise": rng.normal(0, 1, 120),
    })
    df.loc[:9, "v"] = np.nan
    others = ["close", "loose", "noise"]
    expected = df[others].corrwith(df["v"]).abs()
    partners, corr = _partner_info(df, "v")
    assert corr is not None
    pd.testing.assert_series_equal(corr[others], expected[others], check_names=False)
    assert partners == [c for c in others if expected[c] >= 0.4]
    assert "close" in partners


def test_nullable_int_partner_is_selected_like_its_float_twin():
    """An Int64 partner must score exactly like the same values as float64."""
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 80)
    values = np.round(3 * x).astype("int64")
    df_int = pd.DataFrame({"v": x, "p": pd.array(values, dtype="Int64")})
    df_float = pd.DataFrame({"v": x, "p": values.astype("float64")})
    int_partners, int_corr = _partner_info(df_int, "v")
    float_partners, float_corr = _partner_info(df_float, "v")
    assert int_partners == float_partners == ["p"]
    assert float(int_corr["p"]) == float(float_corr["p"])  # type: ignore[index]


def test_degenerate_partner_column_still_ranks_models():
    rng = np.random.default_rng(1)
    v = pd.Series(rng.normal(0, 1, 100))
    v.iloc[:15] = np.nan
    df = pd.DataFrame({"v": v, "i": pd.array([None] * 99 + [7], dtype="Int64")})
    ctx = _ctx(df, "v")
    sel = rank_missing_models(df, "v", ctx, CleanConfig(), mode="aggressive")
    assert sel.primary.model_id in ("mean", "median")
    assert all(c.model_id != "partner_median" for c in sel.alternatives)
