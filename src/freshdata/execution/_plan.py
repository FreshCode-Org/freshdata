"""Decide, without touching data, which stages a backend can run natively.

The native backends reproduce freshdata's deterministic "representation repair"
subset plus simple imputation/outlier handling. Anything that needs the
accuracy-first decision engine or heuristic dtype inference is delegated to the
pandas pipeline. :class:`PlanGenerator` makes that split from the
:class:`~freshdata.CleanConfig` alone, so it is pure and cheap.

The same :class:`PlanGenerator` also emits a *shared logical plan*
(:meth:`PlanGenerator.logical_plan`) — an ordered list of backend-agnostic
:class:`LogicalStep` nodes that every engine (pandas, Polars, DuckDB, Spark)
consumes. Each node carries the step name, its input/output columns, its
parameters, the audit-event schema it is expected to emit, and a fallback
policy. The pandas pipeline remains the reference implementation: a node's
``audit_schema`` mirrors exactly what ``cleaner.run_pipeline`` records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..steps.columns import normalized_column_labels

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import CleanConfig

#: Ordered native stages, matching ``cleaner.run_pipeline`` execution order.
#:
#: The native subset is the *deterministic* part of the pipeline (representation
#: repair + structural reduction + full-row/subset dedup) plus the opt-in
#: ``impute``/``outliers`` overrides, which the Polars/DuckDB/Spark backends
#: reproduce. The accuracy-first decision engine and heuristic dtype inference
#: are evaluated by the pandas backend so results stay identical to ``fd.clean``.
NATIVE_STAGE_ORDER = (
    "column_names",
    "clean_strings",
    "drop_empty_columns",
    "drop_empty_rows",
    "drop_duplicates",
    "impute",
    "outliers",
    "reset_index",
)

#: Duplicate keep-policies the native dedup can express exactly.
_NATIVE_DUPLICATE_KEEP = ("first", "last")
#: Outlier-detection methods the native backends compute deterministically.
#: ``"auto"``/``"isolation_forest"`` are data-dependent / model-based and run on
#: the pandas reference instead.
_NATIVE_OUTLIER_METHODS = ("iqr", "zscore")

#: The audit-event schema each step is expected to emit, keyed by step name.
#: This is the *common audit contract*: every backend must record actions whose
#: ``step`` is one of these and whose payload matches the documented keys. The
#: keys mirror :class:`~freshdata.report.Action` fields populated by the pandas
#: reference for that step.
AUDIT_EVENT_SCHEMA: dict[str, dict[str, str]] = {
    "column_names": {"step": "column_names", "column": "none", "count": "n_renamed",
                     "risk": "low"},
    "strip_whitespace": {"step": "strip_whitespace", "column": "str", "count": "n_trimmed",
                         "risk": "low"},
    "normalize_sentinels": {"step": "normalize_sentinels", "column": "str",
                            "count": "n_replaced", "risk": "low"},
    "drop_empty_columns": {"step": "drop_empty_columns", "column": "none",
                           "count": "n_dropped", "risk": "low"},
    "drop_empty_rows": {"step": "drop_empty_rows", "column": "none", "count": "n_dropped",
                        "risk": "low"},
    "drop_duplicates": {"step": "drop_duplicates", "column": "none", "count": "n_dropped",
                        "risk": "low|medium"},
    "impute": {"step": "impute", "column": "str", "count": "n_filled", "risk": "low"},
    "outliers": {"step": "outliers", "column": "str", "count": "n_outliers", "risk": "low"},
}


@dataclass(frozen=True)
class LogicalStep:
    """One backend-agnostic node of the shared logical cleaning plan.

    Every engine consumes the same ordered list of these. The pandas backend is
    the reference: ``audit_schema`` documents exactly what ``run_pipeline``
    records for this step, and native backends must match it.
    """

    #: Step name (matches the ``step`` of the actions it emits).
    name: str
    #: Columns this step reads.
    input_columns: tuple[str, ...]
    #: Columns this step produces (renames/new flag columns reflected here).
    output_columns: tuple[str, ...]
    #: Step-specific parameters resolved from the config.
    parameters: dict[str, Any]
    #: Audit-event schema this step is expected to emit (see AUDIT_EVENT_SCHEMA).
    audit_schema: dict[str, str]
    #: ``"native"`` (every backend runs it) or ``"pandas"`` (delegated).
    fallback_policy: str


@dataclass
class NativePlan:
    """The result of planning a clean for a native backend."""

    rename_map: dict[str, str]
    stages: list[str]
    fallback_reason: str | None = None
    extra_sentinels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_fallback(self) -> bool:
        return self.fallback_reason is not None


class PlanGenerator:
    """Build a :class:`NativePlan` for a config (no data access)."""

    def __init__(self, config: CleanConfig) -> None:
        self.config = config

    def fallback_reason(self) -> str | None:
        """Return why this config needs the pandas fallback, or ``None``."""
        c = self.config
        if c.engine_mode is not None:
            return (
                f"strategy={c.strategy!r} runs the accuracy-first decision engine, "
                "which is evaluated by the pandas backend"
            )
        if c.fix_dtypes:
            return "fix_dtypes uses sampled heuristics evaluated by the pandas backend"
        if c.drop_constant_columns:
            return "drop_constant_columns is evaluated by the pandas backend"
        if c.optimize_memory:
            return "optimize_memory downcasting is evaluated by the pandas backend"
        if c.outliers is not None and c.outlier_method not in _NATIVE_OUTLIER_METHODS:
            return (
                f"outlier_method={c.outlier_method!r} is data-dependent / model-based "
                "and is evaluated by the pandas backend"
            )
        if c.duplicate_subset is not None:
            return (
                "drop_duplicates with a subset has order-sensitive keep semantics "
                "evaluated by the pandas backend"
            )
        if c.duplicate_keep not in _NATIVE_DUPLICATE_KEEP:
            return (
                f"duplicate_keep={c.duplicate_keep!r} is evaluated by the pandas backend"
            )
        return None

    def _enabled_stages(self) -> list[str]:
        c = self.config
        enabled = {
            "column_names": c.column_names,
            "clean_strings": c.strip_whitespace or c.normalize_sentinels,
            "drop_empty_columns": c.drop_empty_columns,
            "drop_empty_rows": c.drop_empty_rows,
            "drop_duplicates": c.drop_duplicates,
            "impute": c.impute is not None,
            "outliers": c.outliers is not None,
            "reset_index": c.reset_index,
        }
        return [s for s in NATIVE_STAGE_ORDER if enabled.get(s, False)]

    def plan(self, columns: list[object]) -> NativePlan:
        """Plan a clean for a frame with the given *columns*."""
        renamed = normalized_column_labels(columns) if self.config.column_names else list(columns)
        rename_map = {
            str(old): str(new)
            for old, new in zip(columns, renamed)
            if isinstance(old, str) and old != new
        }
        return NativePlan(
            rename_map=rename_map,
            stages=self._enabled_stages(),
            fallback_reason=self.fallback_reason(),
            extra_sentinels=tuple(self.config.extra_sentinels),
        )

    def logical_plan(self, columns: list[object]) -> list[LogicalStep]:
        """Build the shared, ordered logical plan every engine consumes.

        Each node is backend-agnostic; engines translate it into their dialect.
        ``fallback_policy`` is ``"native"`` for every step a native backend can
        run and ``"pandas"`` for the whole plan when :meth:`fallback_reason`
        fires (the decision engine, dtype inference, etc.).
        """
        c = self.config
        plan = self.plan(columns)
        post_rename = tuple(str(plan.rename_map.get(str(col), col)) for col in columns)
        global_fallback = plan.fallback_reason is not None

        def policy(step: str, native: bool = True) -> str:
            return "native" if native and not global_fallback else "pandas"

        steps: list[LogicalStep] = []
        for stage in plan.stages:
            if stage == "column_names":
                steps.append(LogicalStep(
                    name="column_names",
                    input_columns=tuple(str(col) for col in columns),
                    output_columns=post_rename,
                    parameters={"rename_map": dict(plan.rename_map)},
                    audit_schema=AUDIT_EVENT_SCHEMA["column_names"],
                    fallback_policy=policy("column_names"),
                ))
            elif stage == "clean_strings":
                if c.strip_whitespace:
                    steps.append(self._string_step("strip_whitespace", post_rename, c))
                if c.normalize_sentinels:
                    steps.append(self._string_step("normalize_sentinels", post_rename, c))
            elif stage in ("drop_empty_columns", "drop_empty_rows"):
                steps.append(LogicalStep(
                    name=stage,
                    input_columns=post_rename,
                    output_columns=post_rename,
                    parameters={},
                    audit_schema=AUDIT_EVENT_SCHEMA[stage],
                    fallback_policy=policy(stage),
                ))
            elif stage == "drop_duplicates":
                steps.append(LogicalStep(
                    name="drop_duplicates",
                    input_columns=post_rename,
                    output_columns=post_rename,
                    parameters={"keep": c.duplicate_keep,
                                "subset": list(c.duplicate_subset) if c.duplicate_subset else None,
                                "threshold": c.duplicate_threshold},
                    audit_schema=AUDIT_EVENT_SCHEMA["drop_duplicates"],
                    fallback_policy=policy("drop_duplicates"),
                ))
            elif stage == "impute":
                steps.append(LogicalStep(
                    name="impute",
                    input_columns=post_rename,
                    output_columns=post_rename,
                    parameters={"strategy": c.impute},
                    audit_schema=AUDIT_EVENT_SCHEMA["impute"],
                    fallback_policy=policy("impute"),
                ))
            elif stage == "outliers":
                steps.append(LogicalStep(
                    name="outliers",
                    input_columns=post_rename,
                    output_columns=post_rename,
                    parameters={"action": c.outliers, "method": c.outlier_method,
                                "factor": c.outlier_factor},
                    audit_schema=AUDIT_EVENT_SCHEMA["outliers"],
                    fallback_policy=policy("outliers", native=c.outlier_method
                                           in _NATIVE_OUTLIER_METHODS),
                ))
        return steps

    def _string_step(self, name: str, cols: tuple[str, ...], c: CleanConfig) -> LogicalStep:
        return LogicalStep(
            name=name,
            input_columns=cols,
            output_columns=cols,
            parameters={"extra_sentinels": list(c.extra_sentinels)}
            if name == "normalize_sentinels" else {},
            audit_schema=AUDIT_EVENT_SCHEMA[name],
            fallback_policy="native" if self.fallback_reason() is None else "pandas",
        )
