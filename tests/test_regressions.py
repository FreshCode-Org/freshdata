"""Known v0.2 bugs — must stay fixed in v0.3+."""

import warnings

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from expectations import ALL_FIXTURES, load_fixture, resolve_column


def test_aqi_aqi_never_imputed_in_balanced():
    df = load_fixture("aqi_sample")
    out, report = fd.clean(df, return_report=True, verbose=False)
    raw = resolve_column(df, "aqi")
    assert out["aqi"].isna().sum() == df[raw].isna().sum()
    assert "aqi" in out.columns
    assert "aqi_bucket" in out.columns
    missing_actions = [a for a in report if a.step == "missing" and a.column == "aqi"]
    if missing_actions:
        assert "preserved" in missing_actions[0].description


def test_knn_emits_no_runtime_warnings():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(0, 1, n)
    v = pd.Series(3 * x + rng.normal(0, 0.01, n))
    v.iloc[rng.choice(n, 15, replace=False)] = np.nan
    df = pd.DataFrame({
        "a": x + rng.normal(0, 0.01, n),
        "b": 2 * x + rng.normal(0, 0.01, n),
        "v": v,
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        fd.clean(df, strategy="aggressive", drop_duplicates=False, verbose=False)


def test_balanced_never_drops_high_missing_column():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 100)})
    sparse = pd.Series(rng.normal(0, 1, 100))
    sparse.iloc[:45] = np.nan
    df["sparse"] = sparse
    out = fd.clean(df, drop_duplicates=False, drop_empty_rows=False, verbose=False)
    assert "sparse" in out.columns


def test_fraud_column_outliers_not_capped():
    df = load_fixture("fraud_signals")
    before_max = df["fraud_score"].max()
    out = fd.clean(df, verbose=False)
    assert out["fraud_score"].max() == before_max


def test_survey_notes_never_force_filled():
    df = load_fixture("survey_responses")
    before_missing = df["notes"].isna().sum()
    out = fd.clean(df, verbose=False)
    assert out["notes"].isna().sum() >= before_missing


def test_tiny_frame_engine_preserves_not_drops():
    df = load_fixture("tiny_cohort")
    out, report = fd.clean(df, return_report=True, verbose=False)
    assert "value" in out.columns
    assert out["value"].isna().sum() >= df["value"].isna().sum()


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_clean_idempotent_all_fixtures(fixture_name):
    df = load_fixture(fixture_name)
    once = fd.clean(df, verbose=False)
    twice = fd.clean(once, verbose=False)
    pd.testing.assert_frame_equal(once, twice)


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_clean_deterministic_all_fixtures(fixture_name):
    df = load_fixture(fixture_name)
    a = fd.clean(df, verbose=False)
    b = fd.clean(df, verbose=False)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_plan_matches_clean_actions(fixture_name):
    df = load_fixture(fixture_name)
    plan = fd.suggest_plan(df)
    _, report = fd.clean(df, return_report=True, verbose=False)
    for col, cp in plan.column_plans.items():
        if cp.missing is None:
            continue
        engine = [a for a in report if a.step == "missing" and a.column == col]
        if not engine:
            continue
        fill = [a for a in engine if a.model_id or "filled" in a.description]
        action = fill[-1] if fill else engine[-1]
        assert action.model_id == cp.missing.model_id or (
            cp.missing.model_id in action.description
            or cp.missing.model_id in action.rationale
        )


def test_profile_string_dtype_column():
    df = pd.DataFrame({"a": pd.array(["x", "y", None], dtype="string")})
    profile = fd.profile(df)
    assert profile.n_cols == 1


def test_model_router_deterministic():
    df = load_fixture("aqi_sample")
    a = fd.suggest_plan(df)
    b = fd.suggest_plan(df)
    assert a.to_dict() == b.to_dict()


def test_balanced_knn_not_primary():
    df = load_fixture("aqi_sample")
    plan = fd.suggest_plan(df)
    for cp in plan.column_plans.values():
        if cp.missing:
            assert cp.missing.model_id != "knn"
