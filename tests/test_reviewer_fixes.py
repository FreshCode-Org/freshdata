import numpy as np
import pandas as pd
import polars as pl
import pytest
import freshdata as fd
from freshdata._core.models import ActionRecord, CleanConfig, CleanPlan, ColumnRule

def test_apply_plan_executes_custom_plan_actions():
    df = pd.DataFrame({" Name ": ["  Alice "], "Val": [None]})
    plan = CleanPlan(actions=(
        ActionRecord(stage="missing", column="Val", action="impute_median", parameters={"value": 42.0}),
    ))
    cleaned = fd.apply_plan(df, plan)
    assert " Name " in cleaned.columns
    assert cleaned[" Name "].iloc[0] == "  Alice "
    assert cleaned["Val"].iloc[0] == 42.0

def test_target_and_protected_columns_preserved_with_unnormalized_names():
    df = pd.DataFrame({
        "Target Col": [1.0, None, 3.0],
        "Protected Val": [10.0, None, 30.0],
        "Regular Val": [100.0, None, 300.0],
    })
    cfg = CleanConfig(
        normalize_names=True,
        target_column="Target Col",
        rules={"Protected Val": ColumnRule(protected=True)},
        default_impute="median",
    )
    cleaned = fd.clean(df, config=cfg)
    assert "target_col" in cleaned.columns
    assert "protected_val" in cleaned.columns
    assert "regular_val" in cleaned.columns
    assert cleaned["target_col"].isna().sum() == 1
    assert cleaned["protected_val"].isna().sum() == 1
    assert cleaned["regular_val"].isna().sum() == 0

def test_currency_parsing_preserves_negative_and_handles_leading_spaces():
    df = pd.DataFrame({
        "negative_curr": [" -$1,200.50 ", " $300.00 ", "-$50.25"],
        "mixed_text": ["Price is $5", "Free", "Contact us"],
    })
    cleaned = fd.clean(df)
    assert cleaned["negative_curr"].iloc[0] == -1200.50
    assert cleaned["negative_curr"].iloc[2] == -50.25
    assert cleaned["mixed_text"].iloc[1] == "Free"

def test_polars_backend_applies_imputation_and_clipping():
    df = pl.DataFrame({
        "num": [10.0, None, 30.0, 1000.0],
        "str_col": [" a ", " b ", " c ", " d "],
    })
    cfg = CleanConfig(
        rules={"num": ColumnRule(impute="median", outliers="clip")},
        default_outliers="clip",
    )
    plan = fd.suggest_plan(df, cfg)
    assert any(a.stage == "missing" for a in plan.actions)
    cleaned = fd.apply_plan(df, plan)
    lazy_df = df.lazy()
    lazy_cleaned = fd.apply_plan(lazy_df, plan)
    assert isinstance(lazy_cleaned, pl.LazyFrame)
    collected = lazy_cleaned.collect()
    assert collected["num"].null_count() == 0
    assert collected["str_col"].to_list() == ["a", "b", "c", "d"]

def test_missing_strategies_and_outlier_drop():
    df = pd.DataFrame({
        "mean_col": [10.0, None, 20.0, 10.0, 20.0, 30.0],
        "mode_col": ["cat", None, "cat", "dog", "cat", "dog"],
        "drop_col": [1.0, 2.0, 3.0, 4.0, None, 6.0],
        "outlier_col": [10.0, 11.0, 12.0, 10.0, 11.0, 1000.0],
    })
    cfg = CleanConfig(
        rules={
            "mean_col": ColumnRule(impute="mean"),
            "mode_col": ColumnRule(impute="mode"),
            "drop_col": ColumnRule(impute="drop"),
            "outlier_col": ColumnRule(outliers="drop"),
        }
    )
    plan = fd.suggest_plan(df, cfg)
    cleaned = fd.apply_plan(df, plan, cfg)
    assert cleaned["mean_col"].isna().sum() == 0
    assert cleaned["mode_col"].iloc[1] == "cat"
    assert len(cleaned) < len(df)

def test_polars_impute_mean_and_duplicates():
    df = pl.DataFrame({
        "num": [10.0, None, 20.0, 10.0],
        "tag": ["x", "y", "z", "x"],
    })
    cfg = CleanConfig(
        rules={"num": ColumnRule(impute="mean")},
        drop_duplicates=True,
    )
    plan = fd.suggest_plan(df, cfg)
    cleaned = fd.apply_plan(df, plan, cfg)
    assert len(cleaned) == 3
    lazy_cleaned = fd.apply_plan(df.lazy(), plan, cfg).collect()
    assert len(lazy_cleaned) == 3

