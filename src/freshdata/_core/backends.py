# src/freshdata/_core/backends.py
from __future__ import annotations
from typing import Any
import pandas as pd
import polars as pl
from freshdata._core.models import (
    ActionRecord,
    CleanConfig,
    CleanPlan,
    ColumnProfile,
    ProfileReport,
)
from freshdata._core.stages import (
    stage_column_names,
    stage_dtypes,
    stage_duplicates,
    stage_missing,
    stage_outliers,
    stage_strings,
)

class PandasBackend:
    def profile(self, df: pd.DataFrame) -> ProfileReport:
        cols = {}
        for col in df.columns:
            s = df[col]
            nulls = int(s.isna().sum())
            role = "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
            cols[str(col)] = ColumnProfile(
                name=str(col),
                inferred_role=role,
                null_count=nulls,
                null_ratio=nulls / max(len(df), 1),
                unique_count=int(s.nunique()),
                sample_values=tuple(s.dropna().head(3).tolist()),
            )
        return ProfileReport(row_count=len(df), column_count=len(df.columns), columns=cols)

    def suggest_plan(self, df: pd.DataFrame, cfg: CleanConfig) -> CleanPlan:
        actions = []
        cur, act = stage_column_names(df, cfg)
        actions.extend(act)
        cur, act = stage_strings(cur, cfg)
        actions.extend(act)
        cur, act = stage_dtypes(cur, cfg)
        actions.extend(act)
        cur, act = stage_duplicates(cur, cfg)
        actions.extend(act)
        cur, act = stage_missing(cur, cfg)
        actions.extend(act)
        cur, act = stage_outliers(cur, cfg)
        actions.extend(act)
        return CleanPlan(actions=tuple(actions))

    def apply_plan(self, df: pd.DataFrame, plan: CleanPlan, cfg: CleanConfig) -> pd.DataFrame:
        cur, _ = stage_column_names(df, cfg)
        cur, _ = stage_strings(cur, cfg)
        cur, _ = stage_dtypes(cur, cfg)
        cur, _ = stage_duplicates(cur, cfg)
        cur, _ = stage_missing(cur, cfg)
        cur, _ = stage_outliers(cur, cfg)
        return cur

class PolarsBackend:
    def profile(self, df: pl.DataFrame | pl.LazyFrame) -> ProfileReport:
        schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
        cols = {}
        for name, dtype in schema.items():
            role = "numeric" if dtype.is_numeric() else "categorical"
            cols[name] = ColumnProfile(
                name=name,
                inferred_role=role,
                null_count=0,
                null_ratio=0.0,
                unique_count=0,
                sample_values=(),
            )
        row_count = 0 if isinstance(df, pl.LazyFrame) else len(df)
        return ProfileReport(row_count=row_count, column_count=len(schema), columns=cols)

    def suggest_plan(self, df: pl.DataFrame | pl.LazyFrame, cfg: CleanConfig) -> CleanPlan:
        schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
        actions = []
        if cfg.strip_strings:
            for name, dtype in schema.items():
                if dtype == pl.Utf8 or dtype == pl.String:
                    actions.append(ActionRecord("strings", name, "strip_strings", {}))
        return CleanPlan(actions=tuple(actions))

    def apply_plan(self, df: pl.DataFrame | pl.LazyFrame, plan: CleanPlan, cfg: CleanConfig) -> pl.DataFrame | pl.LazyFrame:
        exprs = []
        for a in plan.actions:
            if a.stage == "strings" and a.action == "strip_strings":
                exprs.append(pl.col(a.column).str.strip_chars())
        if exprs:
            return df.with_columns(exprs)
        return df

def resolve_backend(df: Any) -> PandasBackend | PolarsBackend:
    if isinstance(df, pd.DataFrame):
        return PandasBackend()
    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return PolarsBackend()
    raise TypeError(f"Unsupported frame type: {type(df).__name__}. freshdata supports pandas.DataFrame, polars.DataFrame, and polars.LazyFrame.")
