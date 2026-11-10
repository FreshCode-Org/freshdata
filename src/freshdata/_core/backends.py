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
    normalize_config,
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
        cfg_eff = normalize_config(cfg)
        cur, act = stage_column_names(df, cfg)
        actions.extend(act)
        cur, act = stage_strings(cur, cfg_eff)
        actions.extend(act)
        cur, act = stage_dtypes(cur, cfg_eff)
        actions.extend(act)
        cur, act = stage_duplicates(cur, cfg_eff)
        actions.extend(act)
        cur, act = stage_missing(cur, cfg_eff)
        actions.extend(act)
        cur, act = stage_outliers(cur, cfg_eff)
        actions.extend(act)
        return CleanPlan(actions=tuple(actions))

    def apply_plan(self, df: pd.DataFrame, plan: CleanPlan, cfg: CleanConfig) -> pd.DataFrame:
        out = df.copy()
        # Execute explicitly planned actions
        renames = {}
        for a in plan.actions:
            if a.stage == "column_names" and a.action == "normalize_name":
                renames[a.column] = a.parameters["to"]
        if renames:
            out = out.rename(columns=renames)

        for a in plan.actions:
            col = renames.get(a.column, a.column)
            if col not in out.columns:
                continue

            if a.stage == "strings" and a.action == "strip_strings":
                if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
                    out[col] = out[col].astype(object).apply(
                        lambda x: x.strip() if isinstance(x, str) and x.strip() else None
                    )
            elif a.stage == "dtypes" and a.action == "parse_currency":
                cleaned = out[col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
                out[col] = pd.to_numeric(cleaned, errors="coerce")
            elif a.stage == "duplicates" and a.action == "drop_duplicates":
                mask = ~out.duplicated().to_numpy()
                out = pd.DataFrame({col: out[col].to_numpy()[mask] for col in out.columns})
            elif a.stage == "missing":
                if a.action in ("impute_median", "impute_mean"):
                    val = a.parameters.get("value")
                    if val is not None:
                        out[col] = out[col].fillna(val)
                elif a.action == "impute_mode":
                    val = a.parameters.get("value")
                    if val is not None:
                        out[col] = out[col].fillna(val)
                elif a.action == "drop_nulls":
                    mask = ~out[col].isna().to_numpy()
                    out = pd.DataFrame({c: out[c].to_numpy()[mask] for c in out.columns})
            elif a.stage == "outliers":
                if a.action == "clip_iqr":
                    lower = a.parameters.get("lower")
                    upper = a.parameters.get("upper")
                    if lower is not None and upper is not None:
                        out[col] = out[col].clip(lower=lower, upper=upper)
                elif a.action == "drop_outliers":
                    lower = a.parameters.get("lower")
                    upper = a.parameters.get("upper")
                    if lower is not None and upper is not None:
                        mask = ((out[col] >= lower) & (out[col] <= upper) | out[col].isna()).to_numpy()
                        out = pd.DataFrame({c: out[c].to_numpy()[mask] for c in out.columns})
        return out

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
        cfg_eff = normalize_config(cfg)
        for name, dtype in schema.items():
            if cfg_eff.strip_strings and (dtype == pl.Utf8 or dtype == pl.String):
                actions.append(ActionRecord("strings", name, "strip_strings", {}))
            if dtype.is_numeric():
                rule = cfg_eff.rules.get(name)
                impute_strat = rule.impute if rule else (cfg_eff.default_impute if cfg_eff.default_impute != "preserve" else None)
                if impute_strat in ("median", "auto"):
                    actions.append(ActionRecord("missing", name, "impute_median", {}))
                elif impute_strat == "mean":
                    actions.append(ActionRecord("missing", name, "impute_mean", {}))

                outlier_strat = rule.outliers if rule else cfg_eff.default_outliers
                if outlier_strat == "clip":
                    actions.append(ActionRecord("outliers", name, "clip_iqr", {}))
        if cfg_eff.drop_duplicates:
            actions.append(ActionRecord("duplicates", "*", "drop_duplicates", {}))
        return CleanPlan(actions=tuple(actions))

    def apply_plan(self, df: pl.DataFrame | pl.LazyFrame, plan: CleanPlan, cfg: CleanConfig) -> pl.DataFrame | pl.LazyFrame:
        cur = df
        # 1. Renames
        renames = {a.column: a.parameters["to"] for a in plan.actions if a.stage == "column_names" and a.action == "normalize_name"}
        if renames:
            cur = cur.rename(renames)

        # 2. Column expressions
        exprs = []
        for a in plan.actions:
            col_name = renames.get(a.column, a.column)
            if a.stage == "strings" and a.action == "strip_strings":
                exprs.append(pl.col(col_name).str.strip_chars().alias(col_name))
            elif a.stage == "missing":
                if a.action == "impute_median":
                    val = a.parameters.get("value")
                    expr = pl.col(col_name).fill_null(val if val is not None else pl.col(col_name).median()).alias(col_name)
                    exprs.append(expr)
                elif a.action == "impute_mean":
                    val = a.parameters.get("value")
                    expr = pl.col(col_name).fill_null(val if val is not None else pl.col(col_name).mean()).alias(col_name)
                    exprs.append(expr)
            elif a.stage == "outliers" and a.action == "clip_iqr":
                lower = a.parameters.get("lower")
                upper = a.parameters.get("upper")
                if lower is not None and upper is not None:
                    exprs.append(pl.col(col_name).clip(lower, upper).alias(col_name))
        if exprs:
            cur = cur.with_columns(exprs)

        # 3. Duplicates
        if any(a.stage == "duplicates" and a.action == "drop_duplicates" for a in plan.actions):
            if isinstance(cur, pl.LazyFrame):
                cur = cur.unique(maintain_order=False)
            else:
                cur = cur.unique()
        return cur

def resolve_backend(df: Any) -> PandasBackend | PolarsBackend:
    if isinstance(df, pd.DataFrame):
        return PandasBackend()
    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return PolarsBackend()
    raise TypeError(f"Unsupported frame type: {type(df).__name__}. freshdata supports pandas.DataFrame, polars.DataFrame, and polars.LazyFrame.")
