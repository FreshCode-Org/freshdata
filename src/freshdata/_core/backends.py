# src/freshdata/_core/backends.py
from __future__ import annotations
import warnings
from numbers import Real
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
    _impute_strategy,
    _is_protected,
    _outlier_strategy,
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

def _reject(a: ActionRecord, cfg: CleanConfig, backend: str, reason: str = "unknown action") -> None:
    """Refuse an action the backend cannot run: raise when strict, else warn and skip."""
    msg = f"{backend} backend cannot run {a.stage}/{a.action} on column {a.column!r}: {reason}"
    if cfg.strict_backend:
        raise ValueError(f"{msg}. Fix the plan, or set CleanConfig(strict_backend=False) to skip it with a warning.")
    warnings.warn(f"{msg}. Skipped.", UserWarning, stacklevel=4)

def _bounds(a: ActionRecord) -> tuple[float, float] | None:
    lower, upper = a.parameters.get("lower"), a.parameters.get("upper")
    if isinstance(lower, Real) and isinstance(upper, Real):
        return float(lower), float(upper)
    return None

def _first_mode(s: pd.Series) -> Any:
    modes = s.mode()
    return None if modes.empty else modes.iloc[0]

_PANDAS_FILLS = {
    ("missing", "impute_median"): lambda s: s.median(),
    ("missing", "impute_mean"): lambda s: s.mean(),
    ("missing", "impute_mode"): _first_mode,
}

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
            key = (a.stage, a.action)
            if key == ("column_names", "normalize_name"):
                continue
            if key == ("duplicates", "drop_duplicates"):
                mask = ~out.duplicated().to_numpy()
                out = out[mask].reset_index(drop=True)
                continue

            col = renames.get(a.column, a.column)
            if col not in out.columns:
                _reject(a, cfg, "pandas", f"column {col!r} is not in the frame")
                continue
            s = out[col]

            if key == ("strings", "strip_strings"):
                if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
                    out[col] = s.astype(object).apply(_strip_str)
            elif key == ("dtypes", "parse_currency"):
                cleaned = s.astype(str).str.replace(r"[^\d.-]", "", regex=True)
                out[col] = pd.to_numeric(cleaned, errors="coerce")
            elif key in _PANDAS_FILLS:
                val = a.parameters.get("value")
                if val is None:  # plans from the Polars planner carry no value
                    val = _PANDAS_FILLS[key](s)
                if val is not None and not pd.isna(val):
                    out[col] = s.fillna(val)
            elif key == ("missing", "drop_nulls"):
                out = out[s.notna().to_numpy()].reset_index(drop=True)
            elif key in (("outliers", "clip_iqr"), ("outliers", "drop_outliers")):
                bounds = _bounds(a)
                if bounds is None:
                    _reject(a, cfg, "pandas", "parameters need numeric 'lower' and 'upper'")
                elif a.action == "clip_iqr":
                    out[col] = s.clip(lower=bounds[0], upper=bounds[1])
                else:
                    mask = (s.between(*bounds) | s.isna()).to_numpy()
                    out = out[mask].reset_index(drop=True)
            else:
                _reject(a, cfg, "pandas")
        return out

_OUTLIER_ACTIONS = {"clip": "clip_iqr", "drop": "drop_outliers"}

def _pl_mode(expr: pl.Expr) -> pl.Expr:
    # Smallest of the most frequent non-null values: the same tie-break as pandas Series.mode().
    return expr.drop_nulls().mode().sort().first()

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
        bounded: dict[str, tuple[pl.Expr, str]] = {}
        for orig, dtype in schema.items():
            name = _normalize_name(orig) if cfg_eff.normalize_names else orig
            if name != orig:
                actions.append(ActionRecord("column_names", orig, "normalize_name", {"to": name}))
            if _is_protected(name, cfg_eff):
                continue

            if cfg_eff.strip_strings and (dtype == pl.Utf8 or dtype == pl.String):
                actions.append(ActionRecord("strings", name, "strip_strings", {}))
            # Mirror the pandas stage order: IQR bounds see the imputed column.
            base = pl.col(orig)
            impute_strat = _impute_strategy(name, cfg_eff)
            if impute_strat == "drop":
                actions.append(ActionRecord("missing", name, "drop_nulls", {}))
            elif impute_strat == "mode":
                actions.append(ActionRecord("missing", name, "impute_mode", {}))
                base = base.fill_null(_pl_mode(base))
            elif dtype.is_numeric() and impute_strat in ("median", "auto"):
                actions.append(ActionRecord("missing", name, "impute_median", {}))
                base = base.fill_null(base.median())
            elif dtype.is_numeric() and impute_strat == "mean":
                actions.append(ActionRecord("missing", name, "impute_mean", {}))
                base = base.fill_null(base.mean())

            outlier_strat = _outlier_strategy(name, cfg_eff)
            if dtype.is_numeric() and outlier_strat in _OUTLIER_ACTIONS:
                bounded[name] = (base, _OUTLIER_ACTIONS[outlier_strat])
        if bounded:
            # One 1-row aggregate query; apply_plan itself never collects.
            stats = df.lazy().select(
                e
                for i, (base, _) in enumerate(bounded.values())
                for e in (
                    base.count().alias(f"n{i}"),
                    base.quantile(0.25, interpolation="linear").alias(f"q25_{i}"),
                    base.quantile(0.75, interpolation="linear").alias(f"q75_{i}"),
                )
            ).collect().row(0, named=True)
            for i, (name, (_, action)) in enumerate(bounded.items()):
                if stats[f"n{i}"] < 4:
                    continue
                q25, q75 = stats[f"q25_{i}"], stats[f"q75_{i}"]
                iqr = q75 - q25
                lower, upper = q25 - cfg_eff.iqr_factor * iqr, q75 + cfg_eff.iqr_factor * iqr
                actions.append(ActionRecord("outliers", name, action, {"lower": float(lower), "upper": float(upper)}))
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

        # 2. Column expressions, chained per column so actions compose in plan order.
        #    Row filters run after the fused with_columns, on the transformed columns.
        exprs: dict[str, pl.Expr] = {}
        filters: list[pl.Expr] = []
        dedup = False
        for a in plan.actions:
            key = (a.stage, a.action)
            if key == ("column_names", "normalize_name"):
                continue
            if key == ("duplicates", "drop_duplicates"):
                dedup = True
                continue
            col_name = renames.get(a.column, a.column)
            expr = exprs.get(col_name, pl.col(col_name))
            val = a.parameters.get("value")
            if key == ("strings", "strip_strings"):
                exprs[col_name] = expr.str.strip_chars()
            elif key == ("dtypes", "parse_currency"):
                exprs[col_name] = (
                    expr.cast(pl.String).str.replace_all(r"[^\d.-]", "").cast(pl.Float64, strict=False)
                )
            elif key == ("missing", "impute_median"):
                exprs[col_name] = expr.fill_null(val if val is not None else expr.median())
            elif key == ("missing", "impute_mean"):
                exprs[col_name] = expr.fill_null(val if val is not None else expr.mean())
            elif key == ("missing", "impute_mode"):
                exprs[col_name] = expr.fill_null(val if val is not None else _pl_mode(expr))
            elif key == ("missing", "drop_nulls"):
                filters.append(pl.col(col_name).is_not_null())
            elif key in (("outliers", "clip_iqr"), ("outliers", "drop_outliers")):
                bounds = _bounds(a)
                if bounds is None:
                    _reject(a, cfg, "polars", "parameters need numeric 'lower' and 'upper'")
                elif a.action == "clip_iqr":
                    exprs[col_name] = expr.clip(*bounds)
                else:
                    filters.append(pl.col(col_name).is_between(*bounds) | pl.col(col_name).is_null())
            else:
                _reject(a, cfg, "polars")
        if exprs:
            cur = cur.with_columns(e.alias(c) for c, e in exprs.items())
        if filters:
            cur = cur.filter(*filters)

        # 3. Duplicates
        if dedup:
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
