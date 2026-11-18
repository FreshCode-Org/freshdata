# src/freshdata/_core/backends.py
from __future__ import annotations
from typing import TYPE_CHECKING, Any
import pandas as pd
from freshdata._core.models import (
    ActionRecord,
    CleanConfig,
    CleanPlan,
    ColumnProfile,
    ProfileReport,
)
from freshdata._core.stages import (
    _is_protected,
    _strip_str,
    _normalize_name,
    normalize_config,
    stage_column_names,
    stage_dtypes,
    stage_duplicates,
    stage_missing,
    stage_outliers,
    stage_strings,
)

if TYPE_CHECKING:
    import polars as pl

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
        cur = df
        for stage in (stage_column_names, stage_strings, stage_dtypes, stage_duplicates, stage_missing, stage_outliers):
            cur, act = stage(cur, cfg_eff)
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
            if a.stage == "duplicates" and a.action == "drop_duplicates":
                mask = ~out.duplicated().to_numpy()
                out = out[mask].reset_index(drop=True)
                continue

            col = renames.get(a.column, a.column)
            if col not in out.columns:
                continue

            if a.stage == "strings" and a.action == "strip_strings":
                if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
                    out[col] = out[col].astype(object).apply(_strip_str)
            elif a.stage == "dtypes" and a.action == "parse_currency":
                cleaned = out[col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
                out[col] = pd.to_numeric(cleaned, errors="coerce")
            elif a.stage == "missing":
                if a.action.startswith("impute_"):
                    val = a.parameters.get("value")
                    if val is not None:
                        out[col] = out[col].fillna(val)
                elif a.action == "drop_nulls":
                    mask = ~out[col].isna().to_numpy()
                    out = out[mask].reset_index(drop=True)
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
                        out = out[mask].reset_index(drop=True)
        return out

class PolarsBackend:
    def profile(self, df: pl.DataFrame | pl.LazyFrame) -> ProfileReport:
        import polars as pl

        frame = None if isinstance(df, pl.LazyFrame) else df
        schema = df.collect_schema() if frame is None else frame.schema
        row_count = 0 if frame is None else len(frame)
        cols = {}
        for name, dtype in schema.items():
            role = "numeric" if dtype.is_numeric() else "categorical"
            nulls = 0 if frame is None else frame[name].null_count()
            cols[name] = ColumnProfile(
                name=name,
                inferred_role=role,
                null_count=nulls,
                null_ratio=nulls / max(row_count, 1),
                unique_count=0 if frame is None else frame[name].drop_nulls().n_unique(),
                sample_values=(),
            )
        return ProfileReport(row_count=row_count, column_count=len(schema), columns=cols)

    def suggest_plan(self, df: pl.DataFrame | pl.LazyFrame, cfg: CleanConfig) -> CleanPlan:
        import polars as pl

        schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
        actions = []
        cfg_eff = normalize_config(cfg)
        clip_cols: dict[str, pl.Expr] = {}
        for orig, dtype in schema.items():
            name = _normalize_name(orig) if cfg_eff.normalize_names else orig
            if name != orig:
                actions.append(ActionRecord("column_names", orig, "normalize_name", {"to": name}))
            if _is_protected(name, cfg_eff):
                continue
            rule = cfg_eff.rules.get(name)

            if cfg_eff.strip_strings and (dtype == pl.Utf8 or dtype == pl.String):
                actions.append(ActionRecord("strings", name, "strip_strings", {}))
            if dtype.is_numeric():
                # Mirror the pandas stage order: IQR bounds see the imputed column.
                base = pl.col(orig)
                impute_strat = rule.impute if rule else (cfg_eff.default_impute if cfg_eff.default_impute != "preserve" else None)
                if impute_strat in ("median", "auto"):
                    actions.append(ActionRecord("missing", name, "impute_median", {}))
                    base = base.fill_null(base.median())
                elif impute_strat == "mean":
                    actions.append(ActionRecord("missing", name, "impute_mean", {}))
                    base = base.fill_null(base.mean())

                outlier_strat = rule.outliers if rule else cfg_eff.default_outliers
                if outlier_strat == "clip":
                    clip_cols[name] = base
        if clip_cols:
            # One 1-row aggregate query; apply_plan itself never collects.
            stats = df.lazy().select(
                e
                for i, base in enumerate(clip_cols.values())
                for e in (
                    base.count().alias(f"n{i}"),
                    base.quantile(0.25, interpolation="linear").alias(f"q25_{i}"),
                    base.quantile(0.75, interpolation="linear").alias(f"q75_{i}"),
                )
            ).collect().row(0, named=True)
            for i, name in enumerate(clip_cols):
                if stats[f"n{i}"] < 4:
                    continue
                q25, q75 = stats[f"q25_{i}"], stats[f"q75_{i}"]
                iqr = q75 - q25
                lower, upper = q25 - cfg_eff.iqr_factor * iqr, q75 + cfg_eff.iqr_factor * iqr
                actions.append(ActionRecord("outliers", name, "clip_iqr", {"lower": float(lower), "upper": float(upper)}))
        if cfg_eff.drop_duplicates:
            actions.append(ActionRecord("duplicates", "*", "drop_duplicates", {}))
        return CleanPlan(actions=tuple(actions))

    def apply_plan(self, df: pl.DataFrame | pl.LazyFrame, plan: CleanPlan, cfg: CleanConfig) -> pl.DataFrame | pl.LazyFrame:
        import polars as pl

        cur = df
        # 1. Renames
        renames = {a.column: a.parameters["to"] for a in plan.actions if a.stage == "column_names" and a.action == "normalize_name"}
        if renames:
            cur = cur.rename(renames)

        # 2. Column expressions, chained per column so actions compose in plan order
        exprs: dict[str, pl.Expr] = {}
        for a in plan.actions:
            col_name = renames.get(a.column, a.column)
            expr = exprs.get(col_name, pl.col(col_name))
            if a.stage == "strings" and a.action == "strip_strings":
                exprs[col_name] = expr.str.strip_chars()
            elif a.stage == "missing":
                if a.action == "impute_median":
                    val = a.parameters.get("value")
                    exprs[col_name] = expr.fill_null(val if val is not None else expr.median())
                elif a.action == "impute_mean":
                    val = a.parameters.get("value")
                    exprs[col_name] = expr.fill_null(val if val is not None else expr.mean())
            elif a.stage == "outliers" and a.action == "clip_iqr":
                lower = a.parameters.get("lower")
                upper = a.parameters.get("upper")
                if lower is not None and upper is not None:
                    exprs[col_name] = expr.clip(lower, upper)
        if exprs:
            cur = cur.with_columns(e.alias(c) for c, e in exprs.items())

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
    if type(df).__module__.split(".")[0] == "polars":
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "Polars frames need the polars extra: pip install 'freshdata-cleaner[polars]'"
            ) from exc
        if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            return PolarsBackend()
    raise TypeError(f"Unsupported frame type: {type(df).__name__}. freshdata supports pandas.DataFrame, polars.DataFrame, and polars.LazyFrame.")
