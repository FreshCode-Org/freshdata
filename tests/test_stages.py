import pandas as pd
import pytest
from freshdata._core.models import CleanConfig, ColumnRule
from freshdata._core.stages import (
    stage_column_names,
    stage_strings,
    stage_dtypes,
    stage_duplicates,
    stage_missing,
    stage_outliers,
)

def test_stage_column_names_pandas():
    df = pd.DataFrame({" First Name ": ["Alice"], "Spend ($)": [100]})
    cfg = CleanConfig(normalize_names=True)
    out_df, actions = stage_column_names(df, cfg)
    assert list(out_df.columns) == ["first_name", "spend"]
    assert len(actions) == 2

def test_stage_strings_pandas():
    df = pd.DataFrame({"city": [" New York  ", "", "  ", "Chicago"]})
    cfg = CleanConfig(strip_strings=True)
    out_df, actions = stage_strings(df, cfg)
    assert out_df["city"].tolist() == ["New York", None, None, "Chicago"]
    assert any(a.action == "strip_strings" for a in actions)

def test_stage_dtypes_currency_pandas():
    df = pd.DataFrame({"price": ["$1,200.50", "350.00", "$45.99"]})
    cfg = CleanConfig(fix_dtypes=True)
    out_df, actions = stage_dtypes(df, cfg)
    assert pd.api.types.is_numeric_dtype(out_df["price"])
    assert out_df["price"].iloc[0] == 1200.50

def test_stage_duplicates_pandas():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    out_drop, act_drop = stage_duplicates(df, CleanConfig(drop_duplicates=True))
    assert len(out_drop) == 2
    out_keep, act_keep = stage_duplicates(df, CleanConfig(drop_duplicates=False))
    assert len(out_keep) == 3

def test_stage_missing_protected():
    df = pd.DataFrame({"target": [1, None, 3], "val": [10.0, None, 30.0]})
    cfg = CleanConfig(
        target_column="target",
        rules={"val": ColumnRule(impute="median")},
    )
    out_df, actions = stage_missing(df, cfg)
    assert out_df["target"].isna().sum() == 1  # target untouched
    assert out_df["val"].isna().sum() == 0     # val imputed to 20.0
    assert out_df["val"].iloc[1] == 20.0

def test_stage_outliers_clip():
    df = pd.DataFrame({"x": [10, 11, 12, 10, 11, 100]})
    cfg = CleanConfig(
        rules={"x": ColumnRule(outliers="clip")},
        iqr_factor=1.5,
    )
    out_df, actions = stage_outliers(df, cfg)
    assert out_df["x"].max() < 100

def test_stage_outliers_short_series():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    cfg = CleanConfig(rules={"x": ColumnRule(outliers="clip")})
    out_df, actions = stage_outliers(df, cfg)
    assert len(actions) == 0
    assert out_df["x"].tolist() == [1.0, 2.0]

