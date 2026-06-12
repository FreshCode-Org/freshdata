"""Tests for suggest_plan and compare_plans."""

import pandas as pd
import pytest

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
