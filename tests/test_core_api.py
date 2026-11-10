import pandas as pd
import polars as pl
import pytest
import freshdata as fd

EXPECTED_EXPORTS = {
    "ActionRecord",
    "CleanConfig",
    "Cleaner",
    "CleanPlan",
    "CleanReport",
    "ColumnProfile",
    "ColumnRule",
    "ProfileReport",
    "apply_plan",
    "clean",
    "profile",
    "suggest_plan",
}

def test_public_surface_has_exact_12_exports():
    exported = {k for k in dir(fd) if not k.startswith("_")}
    assert exported == EXPECTED_EXPORTS

def test_full_clean_workflow_pandas():
    df = pd.DataFrame({" Name ": ["  Alice ", "Bob "], "Income ($)": ["$50,000", "$60,000"]})
    report = fd.profile(df)
    assert report.row_count == 2
    plan = fd.suggest_plan(df)
    assert len(plan.actions) > 0
    cleaned, clean_report = fd.apply_plan(df, plan, return_report=True)
    assert list(cleaned.columns) == ["name", "income"]
    assert pd.api.types.is_numeric_dtype(cleaned["income"])

def test_clean_wrapper_matches_apply_plan():
    df = pd.DataFrame({" x ": [" a ", " b "]})
    out1 = fd.clean(df)
    plan = fd.suggest_plan(df)
    out2 = fd.apply_plan(df, plan)
    pd.testing.assert_frame_equal(out1, out2)

def test_cleaner_class_interface():
    cleaner = fd.Cleaner(fd.CleanConfig(normalize_names=True))
    df = pd.DataFrame({" A B ": [1, 2]})
    out = cleaner.clean(df)
    assert "a_b" in out.columns
