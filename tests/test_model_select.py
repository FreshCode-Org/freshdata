"""Unit tests for the model selection router."""

import numpy as np
import pandas as pd
import pytest

from freshdata.config import CleanConfig
from freshdata.engine.context import build_context
from freshdata.engine.model_select import rank_missing_models, select_outlier_action


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
