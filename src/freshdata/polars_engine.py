from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, overload

import polars as pl

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)

RoleType = Literal["id", "target", "numeric", "categorical", "datetime", "text"]
ImputeType = Literal["preserve", "drop", "mean", "median", "mode"]
OutlierType = Literal["preserve", "clip", "flag"]


@dataclass(frozen=True)
class ColumnRule:
    role: RoleType | None = None
    impute: ImputeType | None = None
    outliers: OutlierType | None = None
    protected: bool = False


@dataclass(frozen=True)
class CleanConfig:
    strategy: Literal["conservative", "balanced", "aggressive"] = "balanced"
    rules: Mapping[str, ColumnRule] = field(default_factory=dict)
    id_columns: tuple[str, ...] = ()
    target_column: str | None = None
    normalize_names: bool = True
    strip_strings: bool = True
    drop_duplicates: bool = False
    default_impute: Literal["auto", "preserve", "median", "mode"] = "auto"
    default_outliers: OutlierType = "preserve"
    iqr_factor: float = 1.5
    strict_backend: bool = True


@dataclass(frozen=True)
class ActionRecord:
    column: str
    stage: Literal["strings", "missing", "outliers", "drop_column"]
    operation: Literal[
        "strip_whitespace",
        "fill_null",
        "clip_iqr",
        "flag_iqr",
        "drop_column",
        "preserve",
    ]
    params: Mapping[str, Any] = field(default_factory=dict)
    affected_rows: int = 0
    rationale: str = ""


@dataclass(frozen=True)
class CleanPlan:
    rename_map: Mapping[str, str]
    actions: tuple[ActionRecord, ...]
    drop_duplicates: bool = False
    total_rows: int = 0

    def diff(
        self, other: CleanPlan
    ) -> dict[tuple[str, str], tuple[ActionRecord | None, ActionRecord | None]]:
        self_map = {(a.column, a.stage): a for a in self.actions}
        other_map = {(a.column, a.stage): a for a in other.actions}
        changes: dict[tuple[str, str], tuple[ActionRecord | None, ActionRecord | None]] = {}
        for key in sorted(set(self_map) | set(other_map)):
            if self_map.get(key) != other_map.get(key):
                changes[key] = (self_map.get(key), other_map.get(key))
        return changes


@dataclass(frozen=True)
class CleanReport:
    plan: CleanPlan
    renamed_columns: Mapping[str, str]
    dropped_columns: tuple[str, ...]
    total_actions: int


def _to_snake_case(name: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", name).strip()
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cleaned)
    return cleaned.lower()


def _infer_role(
    col: str, dtype: pl.DataType, config: CleanConfig, rule: ColumnRule | None
) -> RoleType:
    if rule and rule.role is not None:
        return rule.role
    if col in config.id_columns or col.endswith("_id") or col == "id":
        return "id"
    if col == config.target_column:
        return "target"
    if dtype.is_numeric():
        return "numeric"
    if dtype.is_temporal():
        return "datetime"
    return "categorical"


def suggest_plan(
    df: pl.DataFrame | pl.LazyFrame,
    config: CleanConfig | None = None,
) -> CleanPlan:
    cfg = config or CleanConfig()
    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(
            f"Expected pl.DataFrame or pl.LazyFrame when strict_backend=True, got {type(df).__name__}."
        )

    lf = df.lazy()
    schema = lf.collect_schema()

    rename_map: dict[str, str] = {}
    for orig_col in schema.names():
        norm_col = _to_snake_case(orig_col) if cfg.normalize_names else orig_col
        rename_map[orig_col] = norm_col

    # Build a single fused aggregation pass so LazyFrames never materialize rows
    stat_exprs: list[pl.Expr] = [pl.len().alias("__total_rows")]
    for orig_col, dtype in schema.items():
        norm_col = rename_map[orig_col]
        col_expr = pl.col(orig_col)
        stat_exprs.append(col_expr.null_count().alias(f"{norm_col}__nulls"))

        if dtype == pl.String and cfg.strip_strings:
            stat_exprs.append(
                col_expr.str.strip_chars()
                .ne(col_expr)
                .fill_null(False)
                .sum()
                .alias(f"{norm_col}__ws_dirty")
            )
        elif dtype.is_numeric():
            stat_exprs.extend(
                [
                    col_expr.median().alias(f"{norm_col}__median"),
                    col_expr.mean().alias(f"{norm_col}__mean"),
                    col_expr.quantile(0.25).alias(f"{norm_col}__q25"),
                    col_expr.quantile(0.75).alias(f"{norm_col}__q75"),
                ]
            )
        if dtype == pl.String:
            stat_exprs.append(
                col_expr.str.strip_chars()
                .drop_nulls()
                .mode()
                .first()
                .alias(f"{norm_col}__mode")
            )

    stats = lf.select(stat_exprs).collect(engine="streaming").row(0, named=True)
    total_rows: int = int(stats["__total_rows"] or 0)

    actions: list[ActionRecord] = []

    for orig_col, dtype in schema.items():
        col = rename_map[orig_col]
        rule = cfg.rules.get(col) or cfg.rules.get(orig_col)
        role = _infer_role(col, dtype, cfg, rule)
        is_protected = bool(
            (rule and rule.protected) or role in ("id", "target")
        )

        # 1. String whitespace repair
        if dtype == pl.String and cfg.strip_strings:
            ws_dirty = int(stats.get(f"{col}__ws_dirty") or 0)
            if ws_dirty > 0:
                actions.append(
                    ActionRecord(
                        column=col,
                        stage="strings",
                        operation="strip_whitespace",
                        affected_rows=ws_dirty,
                        rationale="Strip leading and trailing whitespace",
                    )
                )

        # 2. Missing value planning
        null_count = int(stats.get(f"{col}__nulls") or 0)
        if null_count > 0:
            if is_protected or cfg.strategy == "conservative":
                actions.append(
                    ActionRecord(
                        column=col,
                        stage="missing",
                        operation="preserve",
                        affected_rows=null_count,
                        rationale=f"Protected role={role!r} or conservative strategy",
                    )
                )
            else:
                null_ratio = null_count / max(total_rows, 1)
                if cfg.strategy == "aggressive" and null_ratio >= 0.6:
                    actions.append(
                        ActionRecord(
                            column=col,
                            stage="drop_column",
                            operation="drop_column",
                            affected_rows=null_count,
                            rationale=f"Null ratio {null_ratio:.1%} exceeds aggressive drop threshold",
                        )
                    )
                    continue

                impute_choice = (
                    rule.impute
                    if (rule and rule.impute is not None)
                    else cfg.default_impute
                )
                if impute_choice == "auto":
                    impute_choice = "median" if dtype.is_numeric() else "mode"

                fill_val = stats.get(f"{col}__{impute_choice}")
                if impute_choice != "preserve" and fill_val is not None:
                    actions.append(
                        ActionRecord(
                            column=col,
                            stage="missing",
                            operation="fill_null",
                            params={"value": fill_val, "method": impute_choice},
                            affected_rows=null_count,
                            rationale=f"Impute {null_count} nulls via {impute_choice}={fill_val!r}",
                        )
                    )
                else:
                    actions.append(
                        ActionRecord(
                            column=col,
                            stage="missing",
                            operation="preserve",
                            affected_rows=null_count,
                            rationale="Preserved nulls by policy",
                        )
                    )

        # 3. Outlier planning on numeric columns
        if dtype.is_numeric() and not is_protected:
            outlier_choice = (
                rule.outliers
                if (rule and rule.outliers is not None)
                else cfg.default_outliers
            )
            q25 = stats.get(f"{col}__q25")
            q75 = stats.get(f"{col}__q75")
            if outlier_choice in ("clip", "flag") and q25 is not None and q75 is not None:
                iqr = float(q75) - float(q25)
                lower = float(q25) - cfg.iqr_factor * iqr
                upper = float(q75) + cfg.iqr_factor * iqr
                op = "clip_iqr" if outlier_choice == "clip" else "flag_iqr"
                actions.append(
                    ActionRecord(
                        column=col,
                        stage="outliers",
                        operation=op,
                        params={"lower": lower, "upper": upper, "iqr_factor": cfg.iqr_factor},
                        rationale=f"IQR bounds [{lower:.4g}, {upper:.4g}] (factor={cfg.iqr_factor})",
                    )
                )

    return CleanPlan(
        rename_map=rename_map,
        actions=tuple(actions),
        drop_duplicates=cfg.drop_duplicates,
        total_rows=total_rows,
    )


@overload
def apply_plan(
    df: FrameT, plan: CleanPlan, *, return_report: Literal[False] = False
) -> FrameT: ...


@overload
def apply_plan(
    df: FrameT, plan: CleanPlan, *, return_report: Literal[True]
) -> tuple[FrameT, CleanReport]: ...


def apply_plan(
    df: FrameT, plan: CleanPlan, *, return_report: bool = False
) -> FrameT | tuple[FrameT, CleanReport]:
    is_eager = isinstance(df, pl.DataFrame)
    lf: pl.LazyFrame = df.lazy()

    non_trivial_renames = {k: v for k, v in plan.rename_map.items() if k != v}
    if non_trivial_renames:
        lf = lf.rename(non_trivial_renames)

    dropped_cols: list[str] = [
        a.column for a in plan.actions if a.operation == "drop_column"
    ]
    if dropped_cols:
        lf = lf.drop(dropped_cols)

    # Group column expressions by column so chained operations fuse into one .with_columns()
    col_exprs: dict[str, pl.Expr] = {}
    extra_exprs: list[pl.Expr] = []

    for action in plan.actions:
        col = action.column
        if col in dropped_cols or action.operation == "preserve":
            continue

        expr = col_exprs.get(col, pl.col(col))

        if action.operation == "strip_whitespace":
            expr = expr.str.strip_chars()
            col_exprs[col] = expr
        elif action.operation == "fill_null":
            expr = expr.fill_null(pl.lit(action.params["value"]))
            col_exprs[col] = expr
        elif action.operation == "clip_iqr":
            expr = expr.clip(
                lower_bound=action.params["lower"],
                upper_bound=action.params["upper"],
            )
            col_exprs[col] = expr
        elif action.operation == "flag_iqr":
            lo = action.params["lower"]
            hi = action.params["upper"]
            extra_exprs.append(
                (pl.col(col).lt(lo) | pl.col(col).gt(hi))
                .fill_null(False)
                .alias(f"{col}_is_outlier")
            )

    all_exprs = [expr.alias(col) for col, expr in col_exprs.items()] + extra_exprs
    if all_exprs:
        lf = lf.with_columns(all_exprs)

    if plan.drop_duplicates:
        # maintain_order=False allows Polars to stream deduplication without materializing
        lf = lf.unique(maintain_order=False)

    output: FrameT = lf.collect(engine="streaming") if is_eager else lf  # type: ignore[assignment]

    if not return_report:
        return output

    report = CleanReport(
        plan=plan,
        renamed_columns=non_trivial_renames,
        dropped_columns=tuple(dropped_cols),
        total_actions=len(plan.actions),
    )
    return output, report
