import types
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
    assert set(fd.__all__) == EXPECTED_EXPORTS
    exported = {
        k
        for k in dir(fd)
        if not k.startswith("_") and not isinstance(getattr(fd, k), types.ModuleType)
    }
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
    assert cleaner.profile(df).row_count == 2
    plan = cleaner.suggest_plan(df)
    assert cleaner.apply_plan(df, plan).columns.tolist() == ["a_b"]
    out, rep = cleaner.clean(df, return_report=True)
    assert "a_b" in out.columns
    assert rep.original_cols == 1

def test_polars_clean_workflow():
    df = pl.DataFrame({"msg": ["  hi  ", "there "]})
    plan = fd.suggest_plan(df)
    assert len(plan.actions) == 1
    cleaned = fd.clean(df)
    assert cleaned["msg"].to_list() == ["hi", "there"]

def test_stages_disabled_flags():
    df = pd.DataFrame({" Messy Name ": [" val "], "num": [10.0]})
    cfg = fd.CleanConfig(
        normalize_names=False,
        strip_strings=False,
        fix_dtypes=False,
        default_outliers="preserve",
    )
    cleaned = fd.clean(df, config=cfg)
    assert " Messy Name " in cleaned.columns
    assert cleaned[" Messy Name "].iloc[0] == " val "
