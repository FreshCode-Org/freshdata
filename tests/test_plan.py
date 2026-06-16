"""Tests for suggest_plan and compare_plans."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import freshdata as fd
from expectations import ALL_FIXTURES, load_fixture


def test_suggest_plan_returns_clean_plan():
    df = load_fixture("aqi_sample")
    plan = fd.suggest_plan(df)
    assert isinstance(plan, fd.CleanPlan)
    assert plan.config.strategy == "balanced"
    assert "aqi" in plan.column_plans
    assert plan.column_plans["aqi"].missing.model_id == "preserve"


def test_plan_summary_and_alternatives():
    df = load_fixture("aqi_sample")
    plan = fd.suggest_plan(df)
    text = plan.summary()
    assert "aqi" in text
    alts = plan.alternatives()
    assert isinstance(alts, pd.DataFrame)
    assert "model_id" in alts.columns


def test_compare_plans_three_strategies():
    df = load_fixture("aqi_sample")
    table = fd.compare_plans(df)
    strategies = set(table["strategy"].unique())
    assert "balanced" in strategies and "aggressive" in strategies
    assert fd.suggest_plan(df, strategy="conservative").column_plans == {}


def test_plan_conservative_empty_engine():
    df = load_fixture("sales_export")
    plan = fd.suggest_plan(df, strategy="conservative")
    assert plan.column_plans == {}


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_plan_to_frame(fixture_name):
    df = load_fixture(fixture_name)
    frame = fd.suggest_plan(df).to_frame()
    assert isinstance(frame, pd.DataFrame)


def test_repair_plan_records_patches_and_serializes():
    df = pd.DataFrame({
        "Name": [" Ann ", "Bob", "Bob"],
        "Amount": ["$1,200.50", "-", "-"],
    })
    plan = fd.plan(df, mode="repair_safe")

    assert isinstance(plan, fd.RepairPlan)
    assert plan.config.strategy == "conservative"
    assert plan.patch_count > 0
    assert "source_fingerprint" in plan.to_dict()
    assert "patches" in plan.to_json()
    assert "freshdata repair plan" in plan.to_markdown()
    assert {"patch_id", "operation", "risk", "confidence"}.issubset(plan.to_frame().columns)


def test_repair_plan_apply_matches_safe_repair_and_rollback():
    df = pd.DataFrame({
        "name": [" Ann ", "Bob", "Bob"],
        "amount": ["$1,200.50", "-", "-"],
    })
    repaired, plan = fd.repair(df, mode="repair_safe", return_plan=True)
    expected = fd.clean(df, strategy="conservative", verbose=False)

    assert_frame_equal(repaired, expected)
    assert_frame_equal(plan.apply(), expected)
    assert_frame_equal(plan.rollback(), df)


def test_repair_reviewed_applies_only_approved_patch_ids():
    df = pd.DataFrame({"name": [" Ann ", "Bob"]})
    plan = fd.plan(df, mode="repair_safe")
    update = next(patch for patch in plan.patches if patch.operation == "update_cell")

    repaired = fd.repair(
        df,
        mode="repair_reviewed",
        approved_patch_ids={update.patch_id},
        strategy="conservative",
    )

    assert repaired.loc[0, "name"] == "Ann"
    assert repaired.loc[1, "name"] == "Bob"


def test_repair_reviewed_without_approvals_preserves_input():
    df = pd.DataFrame({"name": [" Ann "]})
    repaired = fd.repair(df, mode="repair_reviewed", strategy="conservative")

    assert_frame_equal(repaired, df)


def test_repair_plan_review_queue_and_dbt_export_shapes():
    df = pd.DataFrame({"value": [1, 2, 1000, None]})
    plan = fd.plan(df, mode="repair_aggressive")

    assert isinstance(plan.review_queue(), pd.DataFrame)
    assert {"unique_id", "column_name", "issue", "patch_id"}.issubset(
        plan.to_dbt_failures().columns
    )


def test_repair_plan_inspect_mode_does_not_propose_patches():
    df = pd.DataFrame({"name": [" Ann "]})
    plan = fd.plan(df, mode="inspect")

    assert plan.patch_count == 0
    assert_frame_equal(plan.apply(), df)
