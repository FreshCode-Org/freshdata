"""Native impute / outlier steps: parity with the pandas reference.

These steps used to force a pandas fallback; the Polars and DuckDB backends now
run them natively. The *count* of affected rows must match the pandas reference
on an unambiguous fixture, and any value-level divergence (quantile
interpolation, mode tie-breaking) must be recorded in ``backend_differences``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd

pytest.importorskip("polars")
pytest.importorskip("duckdb")

NATIVE_ENGINES = ("polars", "duckdb")


def _clean(df, config, engine):
    out, rep = fd.clean(df.copy(), config=config, engine=engine, return_report=True)
    out = out if isinstance(out, pd.DataFrame) else out.to_pandas()
    return out, rep


def test_impute_outlier_action_tuples_match(numeric_df, impute_outlier_config):
    _, ref = _clean(numeric_df, impute_outlier_config, "pandas")
    ref_actions = [(a.step, a.column, a.count) for a in ref.actions]
    for engine in NATIVE_ENGINES:
        _, rep = _clean(numeric_df, impute_outlier_config, engine)
        actions = [(a.step, a.column, a.count) for a in rep.actions]
        assert actions == ref_actions, f"{engine} action mismatch"


def test_impute_fills_numeric_skips_non_numeric(numeric_df, impute_outlier_config):
    for engine in NATIVE_ENGINES:
        out, rep = _clean(numeric_df, impute_outlier_config, engine)
        # median fills the numeric "amount"; it is undefined for the "label"
        # string column, so that gap is preserved (matching the pandas step).
        assert int(out["amount"].isna().sum()) == 0
        assert int(out["label"].isna().sum()) == 1
        assert "amount" in rep.columns_imputed


def test_outlier_clip_counts_match_pandas(numeric_df, impute_outlier_config):
    ref_out, _ = _clean(numeric_df, impute_outlier_config, "pandas")
    for engine in NATIVE_ENGINES:
        out, _ = _clean(numeric_df, impute_outlier_config, engine)
        # The unambiguous 10_000 / 9_999 outliers get clipped down everywhere.
        assert out["amount"].max() < 1_000
        assert out["qty"].max() < 1_000
        assert float(out["amount"].max()) == pytest.approx(float(ref_out["amount"].max()))


def test_outlier_flag_adds_boolean_column(numeric_df):
    cfg = fd.CleanConfig(strategy="conservative", fix_dtypes=False,
                         outliers="flag", outlier_method="iqr")
    for engine in NATIVE_ENGINES:
        out, rep = _clean(numeric_df, cfg, engine)
        assert "amount_outlier" in out.columns
        assert out["amount_outlier"].dtype == bool
        assert int(out["amount_outlier"].sum()) == 1


def test_backend_and_differences_recorded(numeric_df, impute_outlier_config):
    for engine in NATIVE_ENGINES:
        _, rep = _clean(numeric_df, impute_outlier_config, engine)
        assert rep.backend == engine
        steps = {d["step"] for d in rep.backend_differences}
        assert {"impute", "outliers"} <= steps
        assert all(d["backend"] == engine for d in rep.backend_differences)


def test_isolation_forest_falls_back_with_metadata(numeric_df):
    cfg = fd.CleanConfig(strategy="conservative", fix_dtypes=False,
                         outliers="clip", outlier_method="isolation_forest")
    for engine in NATIVE_ENGINES:
        _, rep = _clean(numeric_df, cfg, engine)
        assert rep.backend == "pandas"  # delegated
        assert rep.fallback_events
        event = rep.fallback_events[0]
        assert event["backend"] == engine
        assert "fallback_reason" in event


def test_mean_impute_skips_non_numeric(numeric_df):
    cfg = fd.CleanConfig(strategy="conservative", fix_dtypes=False, impute="mean")
    for engine in NATIVE_ENGINES:
        _, rep = _clean(numeric_df, cfg, engine)
        impute_actions = [a for a in rep.actions if a.step == "impute"]
        # "label" is non-numeric: mean is undefined, so it is skipped (count 0).
        skipped = [a for a in impute_actions if a.column == "label"]
        assert skipped and skipped[0].count == 0


def test_imputed_values_close_to_pandas(numeric_df):
    """Median fill values track the pandas reference (linear interpolation)."""
    cfg = fd.CleanConfig(strategy="conservative", fix_dtypes=False, impute="median")
    ref, _ = _clean(numeric_df, cfg, "pandas")
    filled_at = numeric_df["amount"].isna()
    for engine in NATIVE_ENGINES:
        out, _ = _clean(numeric_df, cfg, engine)
        assert np.allclose(out.loc[filled_at, "amount"], ref.loc[filled_at, "amount"])


def test_mean_impute_parity(numeric_df):
    cfg = fd.CleanConfig(strategy="conservative", fix_dtypes=False, impute="mean")
    ref_out, ref = _clean(numeric_df, cfg, "pandas")
    filled_at = numeric_df["amount"].isna()
    ref_actions = [(a.step, a.column, a.count) for a in ref.actions]
    for engine in NATIVE_ENGINES:
        out, rep = _clean(numeric_df, cfg, engine)
        assert [(a.step, a.column, a.count) for a in rep.actions] == ref_actions
        assert np.allclose(out.loc[filled_at, "amount"], ref_out.loc[filled_at, "amount"])


def test_impute_handles_no_missing_and_all_null():
    # "full" has no gaps (skipped); "blank" is all-null (nothing to learn from).
    df = pd.DataFrame({
        "full": [1.0, 2.0, 3.0, 4.0],
        "gap": [1.0, np.nan, 3.0, np.nan],
        "blank": [np.nan, np.nan, np.nan, np.nan],
    })
    cfg = fd.CleanConfig(strategy="conservative", fix_dtypes=False,
                         impute="median", drop_empty_columns=False)
    _, ref = _clean(df, cfg, "pandas")
    ref_actions = [(a.step, a.column, a.count) for a in ref.actions]
    for engine in NATIVE_ENGINES:
        out, rep = _clean(df, cfg, engine)
        assert [(a.step, a.column, a.count) for a in rep.actions] == ref_actions
        assert int(out["gap"].isna().sum()) == 0


def test_outliers_constant_column_noop():
    # A constant numeric column has zero IQR spread -> no bounds, no action.
    df = pd.DataFrame({"const": [5.0] * 6, "varied": [1.0, 2.0, 3.0, 4.0, 5.0, 99.0]})
    cfg = fd.CleanConfig(strategy="conservative", fix_dtypes=False,
                         outliers="clip", outlier_method="iqr")
    for engine in NATIVE_ENGINES:
        out, rep = _clean(df, cfg, engine)
        outlier_cols = {a.column for a in rep.actions if a.step == "outliers"}
        assert "const" not in outlier_cols
        assert out["const"].nunique() == 1


def test_mode_impute_fills_non_numeric(numeric_df):
    cfg = fd.CleanConfig(strategy="conservative", fix_dtypes=False, impute="mode")
    for engine in NATIVE_ENGINES:
        out, rep = _clean(numeric_df, cfg, engine)
        # mode is defined for every dtype, so the string gap is filled.
        assert int(out["label"].isna().sum()) == 0
        assert "label" in rep.columns_imputed


def test_zscore_outlier_method_runs_natively(numeric_df):
    # A single extreme value inflates the std, so mean±3σ flags nothing — the
    # point here is that the zscore detector runs natively (no pandas fallback)
    # and records its statistics caveat.
    cfg = fd.CleanConfig(strategy="conservative", fix_dtypes=False,
                         outliers="flag", outlier_method="zscore")
    for engine in NATIVE_ENGINES:
        _, rep = _clean(numeric_df, cfg, engine)
        assert rep.backend == engine
        assert any(d["step"] == "outliers" for d in rep.backend_differences)
