"""Outlier decision engine: methods, actions, preservation rules."""

import numpy as np
import pandas as pd
import pytest

import freshdata as fd

QUIET = {"drop_duplicates": False, "verbose": False}


def normal_with_spike(n=200, spike=1_000.0, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(10, 1, n)
    values[-1] = spike
    return values


def outlier_actions(report):
    return [a for a in report if a.step == "outliers"]


def test_default_action_caps_not_deletes():
    df = pd.DataFrame({"v": normal_with_spike()})
    out, report = fd.clean(df, report=True, **QUIET)
    assert len(out) == 200  # no rows deleted
    assert out["v"].max() < 1_000.0
    [action] = outlier_actions(report)
    assert "capped" in action.description
    assert "method=" in action.description  # detection method always logged
    assert report.outliers_handled >= 1


def test_auto_method_picks_zscore_for_normal_and_iqr_for_skewed():
    rng = np.random.default_rng(0)
    skewed = rng.lognormal(0, 1, 300)
    skewed[-1] = 10_000.0
    df = pd.DataFrame({"normalish": normal_with_spike(300),
                       "skewed": skewed})
    _, report = fd.clean(df, report=True, outlier_method="auto", **QUIET)
    methods = {a.column: a.description for a in outlier_actions(report)}
    assert "method=zscore" in methods["normalish"]
    assert "method=iqr" in methods["skewed"]


def test_remove_action_drops_rows():
    df = pd.DataFrame({"v": normal_with_spike()})
    out, report = fd.clean(df, report=True, outlier_action="remove", **QUIET)
    assert len(out) < 200
    assert out["v"].max() < 1_000.0
    [action] = outlier_actions(report)
    assert "removed" in action.description


def test_flag_action_keeps_data():
    df = pd.DataFrame({"v": normal_with_spike()})
    out = fd.clean(df, outlier_action="flag", **QUIET)
    assert out["v"].max() == 1_000.0
    assert out["v_outlier"].sum() >= 1


def test_action_none_detects_but_preserves():
    df = pd.DataFrame({"v": normal_with_spike()})
    out, report = fd.clean(df, report=True, outlier_action=None, **QUIET)
    assert out["v"].max() == 1_000.0
    [action] = outlier_actions(report)
    assert "preserved" in action.description and action.rationale


def test_domain_sensitive_columns_preserved():
    df = pd.DataFrame({"fraud_score": normal_with_spike()})
    out, report = fd.clean(df, report=True, **QUIET)
    assert out["fraud_score"].max() == 1_000.0
    [action] = outlier_actions(report)
    assert "preserved" in action.description
    assert any("fraud_score" in r for r in report.recommendations)


def test_id_and_target_columns_never_capped():
    spiky = normal_with_spike()
    df = pd.DataFrame({"user_id": spiky, "target": spiky.copy()})
    out, report = fd.clean(df, report=True, **QUIET)
    assert out["user_id"].max() == 1_000.0
    assert out["target"].max() == 1_000.0
    assert all("preserved" in a.description for a in outlier_actions(report))


def test_heavy_tailed_column_is_flagged_not_capped():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"v": rng.standard_cauchy(300)})  # ~25% outside fences
    out, report = fd.clean(df, report=True, **QUIET)
    assert "v_outlier" in out.columns
    assert any("heavy-tailed" in w for w in report.warnings)


def test_isolation_forest_method():
    pytest.importorskip("sklearn")
    df = pd.DataFrame({"v": normal_with_spike(300)})
    out, report = fd.clean(df, report=True, outlier_method="isolation_forest",
                           **QUIET)
    assert out["v"].max() < 1_000.0
    [action] = outlier_actions(report)
    assert "isolation_forest" in action.description


def test_isolation_forest_falls_back_without_enough_rows():
    df = pd.DataFrame({"v": normal_with_spike(50)})
    out, report = fd.clean(df, report=True, outlier_method="isolation_forest",
                           **QUIET)
    assert out["v"].max() < 1_000.0  # fell back to a fence method and capped
    [action] = outlier_actions(report)
    assert "method=" in action.description


def test_small_columns_skipped():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 100.0]})  # < 10 values: no stats
    out = fd.clean(df, **QUIET)
    assert out["v"].max() == 100.0
