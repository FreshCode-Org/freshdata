"""Structured record of everything :func:`freshdata.clean` did.

Trust is the core feature of an auto-cleaner: every transformation is recorded
as an :class:`Action` — with a rationale, a risk level, and a confidence score
when it came from the decision engine — so users can audit exactly what
changed, how much, and why. Columns that were deliberately *not* touched get
an action too, so remaining NaNs are always explained.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ._util import format_bytes
from .findings import findings_from_dict
from .render.mixins import HtmlReprMixin

#: Valid risk levels, in increasing order of severity.
RISK_LEVELS = ("low", "medium", "high")


def _json_scalar(value: Any) -> Any:
    """One cell value in a JSON-representable form (repr as last resort)."""
    if isinstance(value, float) and value != value:  # noqa: PLR0124 — NaN check
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):  # numpy scalars
        with contextlib.suppress(Exception):
            return _json_scalar(value.item())
    return repr(value)


@dataclass(frozen=True)
class Action:
    """One transformation (or deliberate non-transformation) of the data.

    Attributes
    ----------
    step:
        Machine-readable step name, e.g. ``"fix_dtypes"`` or ``"missing"``.
    column:
        Column the action applied to, or ``None`` for table-level actions.
    description:
        Human-readable summary of what happened.
    count:
        Number of cells or rows affected (0 for informational notes).
    rationale:
        Why the decision engine chose this action ("" for non-engine steps).
    risk:
        "low", "medium", or "high" — how likely the action is to need review.
    confidence:
        Engine confidence in the decision, in [0, 1] (1.0 for non-engine steps,
        which are deterministic representation repairs).
    """

    step: str
    column: str | None
    description: str
    count: int = 0
    rationale: str = ""
    risk: str = "low"
    confidence: float = 1.0
    model_id: str = ""
    #: How the decision was reached: ``"automatic"`` (engine applied it),
    #: ``"suggested"`` (proposed, not applied), ``"skipped"`` (deliberately not
    #: applied), or ``"approved"`` (a human/memory-approved decision).
    status: str = "automatic"
    #: ``True``/``False`` if the change is (ir)reversible, ``None`` if unknown.
    reversible: bool | None = None
    #: ``True`` when a cleaning-memory decision influenced this action.
    memory_influenced: bool = False
    #: ``True`` when the action is flagged for human review.
    human_review: bool = False
    #: Free-form, JSON-friendly audit detail (e.g. the semantic layer's
    #: raw/proposed value, evidence, and memory-replay key). Empty for steps
    #: that don't need it, so ``to_dict()`` omits it for those.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        target = f"{self.column!r}: " if self.column is not None else ""
        return f"[{self.step}] {target}{self.description}"


@dataclass
class CleanReport(HtmlReprMixin):
    """Everything one :func:`freshdata.clean` run did, in order.

    Iterable and sized: ``len(report)`` is the number of actions, and
    ``for action in report`` walks them in execution order. ``bool(report)``
    is ``True`` iff anything was changed.

    Beyond the action log, the report carries a cleaning summary (missing
    cells before/after, duplicates removed, outliers handled, columns
    dropped/imputed/preserved), engine warnings for risky columns, and
    recommendations for manual review.
    """

    _render_kind = "clean_report"

    actions: list[Action] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    memory_before: int = 0
    memory_after: int = 0
    duration_seconds: float = 0.0
    missing_before: int = 0
    missing_after: int = 0
    duplicates_removed: int = 0
    outliers_handled: int = 0
    columns_dropped: list[str] = field(default_factory=list)
    columns_imputed: list[str] = field(default_factory=list)
    columns_preserved: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    #: Per-cell record of values that ``fix_dtypes`` coerced to missing because
    #: they did not parse as the column's inferred type: ``{column: {row_label:
    #: original_value}}``. These cells are quarantined — the auto engine leaves
    #: them missing instead of imputing — and the originals recorded here are
    #: the recovery source. Capped per column (the action count stays exact).
    coerced_cells: dict[str, dict[Any, Any]] = field(default_factory=dict)
    #: Full row keys of every coercion casualty per column (keys only, never
    #: capped): ``{column: (row_label, ...)}``. ``coerced_cells`` holds the
    #: reviewable payload capped per column; the engine's imputation guard
    #: uses this so no casualty is ever imputed, even beyond the cap.
    coerced_rows: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    #: Domain pack applied via ``clean(df, domain=...)``, or ``None``.
    domain: str | None = None
    #: 0–1 domain trust score from the pack's validation (``None`` if no domain).
    domain_trust_score: float | None = None
    #: Per-rule domain validation findings (JSON-friendly dicts).
    domain_findings: list[dict[str, Any]] = field(default_factory=list)
    #: Domain repair-log entries (JSON-friendly dicts).
    domain_repairs: list[dict[str, Any]] = field(default_factory=list)
    #: Streaming/micro-batch metadata (batch id, per-batch + rolling + cumulative
    #: trust, drift flag, warmup flag), or ``None`` for a normal in-memory clean.
    streaming: dict[str, Any] | None = None
    #: Execution backend that produced this report (``"pandas"``, ``"polars"``,
    #: ``"duckdb"``, ``"spark"``, ``"freshcore"``), or ``None`` for the default
    #: in-memory path.
    backend: str | None = None
    #: Backend the caller *asked* for (``engine=`` / ``EngineConfig.engine``,
    #: including ``"auto"``), before any resolution or fallback. Compare with
    #: :attr:`backend` (what actually ran) to see execution divergence at a glance.
    requested_backend: str | None = None
    #: Process peak RSS in bytes at report-finalize time (``ru_maxrss``, the
    #: process-lifetime high-water mark — not a per-call delta). ``None`` where
    #: the ``resource`` module is unavailable (e.g. Windows).
    peak_memory: int | None = None
    #: Number of rows pulled into memory for the returned result, when the
    #: output was materialized and cheaply countable. ``None`` for native
    #: un-materialized handles (nothing was pulled) and for Spark results
    #: (counting would trigger a job).
    rows_materialized: int | None = None
    #: ``False`` when the cleaned result was returned as a native, un-materialized
    #: handle (a DuckDB relation or a Polars ``LazyFrame`` via
    #: ``output_format="duckdb"``/``"polars-lazy"``). In that case the "after"
    #: counts are *not* computed, since doing so would force a scan/collect and
    #: defeat the out-of-core path; ``summary()`` says so plainly.
    materialized: bool = True
    #: When a backend delegated a step to the pandas reference implementation,
    #: one ``{"backend", "fallback_step", "fallback_reason"}`` dict per delegation.
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    #: Recorded semantic divergences between a native backend and the pandas
    #: reference (e.g. quantile interpolation): JSON-friendly dicts with at least
    #: ``{"backend", "step", "column", "detail"}``.
    backend_differences: list[dict[str, Any]] = field(default_factory=list)

    #: Learned-profile replay outcome (profile id, drift severity, column
    #: overlap, reasons) when ``profile=`` was supplied; None otherwise.
    profile_replay: dict[str, Any] | None = None
    #: Backend-provided stage timings. Native engines may populate this with
    #: ``{"backend", "stage", "seconds"}`` records so benchmark reports can
    #: show operation-level cost without parsing human-readable action text.
    stage_timings: list[dict[str, Any]] = field(default_factory=list)
    #: Per-column source provenance summary (page/region/parser_confidence/
    #: source_file/extracted_at + ``modified``/``low_confidence_repair`` flags)
    #: when ``clean`` was called with ``source_provenance=``, else ``None``.
    #: JSON-friendly; see :mod:`freshdata.provenance`.
    source_provenance: dict[str, Any] | None = None
    #: Contract schema-diff result (``DriftReport.to_dict()``) when ``clean`` was
    #: called with ``contract=``, else ``None``. JSON-friendly so ``CleanReport``
    #: stays in the light core and never imports the enterprise layer. Explains
    #: incoming schema drift (added/removed/renamed/dtype/nullable/semantic) that
    #: was present *before* any repair ran; see :func:`freshdata.diff_schema`.
    contract_violations: dict[str, Any] | None = None
    #: Deterministic SHA-256 digest of the approved/rejected decisions when this
    #: report came from :func:`freshdata.apply_plan`, else ``None``. Stable for
    #: a given reviewed plan — suitable for audit trails and change control.
    decisions_hash: str | None = None
    #: Compact undo information (``apply_plan(..., keep_undo=True)`` only):
    #: ``{"entries": [{action_id, column, index, value}], "column_dtypes": {...}}``.
    #: Never serialized by :meth:`to_dict` — it can hold raw cell values.
    undo_log: dict[str, Any] | None = None

    def record_fallback(self, backend: str, step: str, reason: str) -> None:
        """Record that *backend* delegated *step* to the pandas reference."""
        self.fallback_events.append(
            {"backend": backend, "fallback_step": step, "fallback_reason": reason}
        )

    def record_backend_difference(
        self, backend: str, step: str, detail: str, *, column: str | None = None
    ) -> None:
        """Record a semantics difference between a native backend and pandas."""
        self.backend_differences.append(
            {"backend": backend, "step": step, "column": column, "detail": detail}
        )

    def record_stage_timing(self, backend: str, stage: str, seconds: float) -> None:
        """Record a backend-provided stage runtime in seconds."""
        self.stage_timings.append(
            {"backend": backend, "stage": stage, "seconds": float(seconds)}
        )

    def add(
        self,
        step: str,
        description: str,
        *,
        column: str | None = None,
        count: int = 0,
        rationale: str = "",
        risk: str = "low",
        confidence: float = 1.0,
        model_id: str = "",
        status: str = "automatic",
        reversible: bool | None = None,
        memory_influenced: bool = False,
        human_review: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one action (internal; called by the pipeline)."""
        self.actions.append(
            Action(
                step=step,
                column=column,
                description=description,
                count=int(count),
                rationale=rationale,
                risk=risk,
                confidence=float(confidence),
                model_id=model_id,
                status=status,
                reversible=reversible,
                memory_influenced=memory_influenced,
                human_review=human_review,
                metadata=dict(metadata) if metadata else {},
            )
        )

    def add_warning(self, message: str) -> None:
        """Record a warning about a risky column or decision (internal)."""
        if message not in self.warnings:
            self.warnings.append(message)

    def add_recommendation(self, message: str) -> None:
        """Record a suggestion for manual review (internal)."""
        if message not in self.recommendations:
            self.recommendations.append(message)

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __bool__(self) -> bool:
        return any(a.count for a in self.actions)

    @property
    def cells_changed(self) -> int:
        """Total affected cells/rows summed across all actions."""
        return sum(a.count for a in self.actions)

    @staticmethod
    def _action_dict(a: Action) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "step": a.step,
            "column": a.column,
            "description": a.description,
            "count": a.count,
            "rationale": a.rationale,
            "risk": a.risk,
            "confidence": a.confidence,
            "model_id": a.model_id,
            "status": a.status,
            "reversible": a.reversible,
            "memory_influenced": a.memory_influenced,
            "human_review": a.human_review,
        }
        if a.metadata:
            entry["metadata"] = dict(a.metadata)
        return entry

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation of the report.

        This format is ideal for writing to logs, persisting audit snapshots,
        or returning a stable object from service endpoints.

        Examples
        --------
        >>> report = CleanReport(rows_before=10, rows_after=8, cols_before=4, cols_after=3)
        >>> payload = report.to_dict()
        >>> 'actions' in payload
        True
        >>> payload['rows_before'], payload['rows_after']
        (10, 8)
        """
        payload: dict[str, Any] = {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "cols_before": self.cols_before,
            "cols_after": self.cols_after,
            "memory_before": self.memory_before,
            "memory_after": self.memory_after,
            "duration_seconds": self.duration_seconds,
            "missing_before": self.missing_before,
            "missing_after": self.missing_after,
            "duplicates_removed": self.duplicates_removed,
            "outliers_handled": self.outliers_handled,
            "columns_dropped": list(self.columns_dropped),
            "columns_imputed": list(self.columns_imputed),
            "columns_preserved": list(self.columns_preserved),
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
            "actions": [self._action_dict(a) for a in self.actions],
        }
        if self.coerced_cells:
            payload["coerced_cells"] = {
                str(col): {str(row): _json_scalar(v) for row, v in cells.items()}
                for col, cells in self.coerced_cells.items()
            }
        if self.domain is not None:
            payload["domain"] = self.domain
            payload["domain_trust_score"] = self.domain_trust_score
            payload["domain_findings"] = list(self.domain_findings)
            payload["domain_repairs"] = list(self.domain_repairs)
        if self.streaming is not None:
            payload["streaming"] = dict(self.streaming)
        if self.backend is not None:
            payload["backend"] = self.backend
        if self.requested_backend is not None:
            payload["requested_backend"] = self.requested_backend
        if self.peak_memory is not None:
            payload["peak_memory"] = self.peak_memory
        if self.rows_materialized is not None:
            payload["rows_materialized"] = self.rows_materialized
        if not self.materialized:
            payload["materialized"] = False
        if self.fallback_events:
            payload["fallback_events"] = list(self.fallback_events)
        if self.backend_differences:
            payload["backend_differences"] = list(self.backend_differences)
        if self.stage_timings:
            payload["stage_timings"] = list(self.stage_timings)
        if self.source_provenance is not None:
            payload["source_provenance"] = self.source_provenance
        if self.contract_violations is not None:
            payload["contract_violations"] = self.contract_violations
        if self.decisions_hash is not None:
            payload["decisions_hash"] = self.decisions_hash
        if self.profile_replay is not None:
            payload["profile_replay"] = dict(self.profile_replay)
        return payload

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the report's stable audit payload as JSON.

        Keyword arguments are forwarded to :func:`json.dumps`, so callers can
        choose options such as ``indent=2`` or ``sort_keys=True``.
        """
        return json.dumps(self.to_dict(), **kwargs)

    def write_json(self, path: str | Path, **kwargs: Any) -> None:
        """Write the report's JSON audit payload to *path* as UTF-8 text."""
        Path(path).write_text(self.to_json(**kwargs) + "\n", encoding="utf-8")

    def revert(
        self, df: pd.DataFrame, action_ids: list[str] | None = None
    ) -> pd.DataFrame:
        """Undo reversible plan actions on *df*, returning a new frame.

        Requires this report to come from
        ``freshdata.apply_plan(..., keep_undo=True)``; with ``action_ids=None``
        every recorded reversible action is undone. Cells whose rows no longer
        exist in *df* are skipped silently (they cannot be restored).
        """
        if not self.undo_log or not self.undo_log.get("entries"):
            raise ValueError(
                "this report holds no undo information — apply the plan with "
                "keep_undo=True to enable revert()"
            )
        wanted = None if action_ids is None else set(action_ids)
        entries = [
            e
            for e in self.undo_log["entries"]
            if wanted is None or e["action_id"] in wanted
        ]
        if wanted is not None:
            known = {e["action_id"] for e in self.undo_log["entries"]}
            missing = sorted(wanted - known)
            if missing:
                raise KeyError(
                    f"no undo information for action id(s) {missing}; "
                    f"reversible actions: {sorted(known)}"
                )
        out = df.copy(deep=False)
        touched: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            touched.setdefault(entry["column"], []).append(entry)
        for column, column_entries in touched.items():
            if column not in out.columns:
                continue
            series = out[column]
            if series.dtype != object:
                series = series.astype(object)
            for entry in column_entries:
                labels = [i for i in entry["index"] if i in series.index]
                if labels:
                    series.loc[labels] = entry["value"]
            original_dtype = (self.undo_log.get("column_dtypes") or {}).get(column)
            if original_dtype is not None:
                # Mixed values after a partial revert legitimately stay object.
                with contextlib.suppress(ValueError, TypeError):
                    series = series.astype(original_dtype)
            out[column] = series
        return out

    def to_findings(self, *, lineage_run_id: str | None = None) -> list:
        """Project this report into normalized :class:`~freshdata.QualityFinding` objects.

        Surfaces violated domain rules (enriched with any repair that was applied)
        and medium/high-risk engine actions, so the result can be exported to dbt
        tests, a Great Expectations suite, or an exception table. ``CleanReport``
        keeps its own shape; this is a pure read-only projection.

        Examples
        --------
        >>> CleanReport().to_findings()
        []
        """
        payload = {
            "domain_findings": self.domain_findings,
            "domain_repairs": self.domain_repairs,
            "actions": [
                {
                    "step": a.step,
                    "column": a.column,
                    "description": a.description,
                    "count": a.count,
                    "risk": a.risk,
                }
                for a in self.actions
            ],
        }
        return findings_from_dict(payload, lineage_run_id=lineage_run_id)

    def to_frame(self) -> pd.DataFrame:
        """Return one action per row as a `pandas.DataFrame`.

        This representation works best when you want to inspect the report in
        notebooks, ad hoc dashboards, or quick filtering workflows.

        Examples
        --------
        >>> from freshdata import CleanReport, Action
        >>> report = CleanReport(actions=[Action(step='coerce', column='age')])
        >>> frame = report.to_frame()
        >>> frame.loc[0, 'step']
        'coerce'
        """
        return pd.DataFrame(
            [
                (
                    a.step,
                    a.column,
                    a.description,
                    a.count,
                    a.rationale,
                    a.risk,
                    a.confidence,
                    a.model_id,
                    a.status,
                    a.reversible,
                    a.memory_influenced,
                    a.human_review,
                )
                for a in self.actions
            ],
            columns=[
                "step",
                "column",
                "description",
                "count",
                "rationale",
                "risk",
                "confidence",
                "model_id",
                "status",
                "reversible",
                "memory_influenced",
                "human_review",
            ],
        )

    def summary(self) -> str:
        """Render a concise text summary for terminal or notebook output.

        This method is the quickest way to create a human-readable snapshot
        of what happened during a clean run.

        Examples
        --------
        >>> report = CleanReport(rows_before=4, rows_after=4, cols_before=2, cols_after=2)
        >>> text = report.summary()
        >>> text.startswith('freshdata clean report')
        True
        >>> 'rows:' in text
        True
        """
        if not self.materialized:
            lines = [
                "freshdata clean report",
                f"  rows:    {self.rows_before:,} -> (native handle — not materialized)",
                f"  columns: {self.cols_before:,} -> (native handle — not materialized)",
                "  result:  returned un-materialized; call .fetchdf()/.collect() to pull rows",
                f"  time:    {self.duration_seconds:.3f}s",
            ]
        else:
            d_rows = self.rows_after - self.rows_before
            d_cols = self.cols_after - self.cols_before
            lines = [
                "freshdata clean report",
                f"  rows:    {self.rows_before:,} -> {self.rows_after:,} ({d_rows:+,})",
                f"  columns: {self.cols_before:,} -> {self.cols_after:,} ({d_cols:+,})",
                f"  missing: {self.missing_before:,} -> {self.missing_after:,} cell(s)",
                f"  memory:  {format_bytes(self.memory_before)} -> "
                f"{format_bytes(self.memory_after)}",
                f"  time:    {self.duration_seconds:.3f}s",
            ]
        facts = []
        if self.duplicates_removed:
            facts.append(f"{self.duplicates_removed} duplicate row(s) removed")
        if self.outliers_handled:
            facts.append(f"{self.outliers_handled} outlier(s) handled")
        if self.columns_dropped:
            facts.append(f"dropped: {', '.join(self.columns_dropped)}")
        if self.columns_imputed:
            facts.append(f"imputed: {', '.join(self.columns_imputed)}")
        if self.columns_preserved:
            facts.append(f"preserved: {', '.join(self.columns_preserved)}")
        if facts:
            lines.append("  engine:  " + "; ".join(facts))
        if self.domain is not None:
            n_err = sum(
                1
                for f in self.domain_findings
                if f.get("status") == "violated" and f.get("severity") == "error"
            )
            n_warn = sum(
                1
                for f in self.domain_findings
                if f.get("status") == "violated" and f.get("severity") == "warning"
            )
            score = self.domain_trust_score if self.domain_trust_score is not None else 1.0
            applied = sum(1 for r in self.domain_repairs if r.get("status") == "applied")
            lines.append(
                f"  domain:  {self.domain} — trust {score:.2f}, "
                f"{n_err} error(s), {n_warn} warning(s), {applied} repair(s) applied"
            )
        if self.actions:
            lines.append(f"  actions ({len(self.actions)}):")
            lines.extend(f"    - {a}" for a in self.actions)
        else:
            lines.append("  actions: none — data was already clean")
        if self.warnings:
            lines.append(f"  warnings ({len(self.warnings)}):")
            lines.extend(f"    ! {w}" for w in self.warnings)
        if self.recommendations:
            lines.append(f"  review ({len(self.recommendations)}):")
            lines.extend(f"    ? {r}" for r in self.recommendations)
        cv = self.contract_violations
        if cv is not None:
            verdict = "PASS" if cv.get("passed") else "FAIL"
            n_err = cv.get("n_errors", 0)
            n_warn = cv.get("n_warnings", 0)
            lines.append(
                f"  contract '{cv.get('baseline_name')}' v{cv.get('baseline_version')}: "
                f"{verdict} ({n_err} error(s), {n_warn} warning(s))"
            )
            for f in cv.get("findings", []):
                if f.get("status") == "passed":
                    continue
                marker = "✗" if f.get("status") == "failed" else "!"
                col = f" `{f['column']}`" if f.get("column") else ""
                lines.append(f"    {marker} [{f.get('check_id')}]{col}: {f.get('message')}")
        return "\n".join(lines)

    def brief(self) -> str:
        """Compact summary for ``verbose=True`` console output."""
        line = (
            f"freshdata: rows {self.rows_before:,}->{self.rows_after:,}, "
            f"cols {self.cols_before}->{self.cols_after}, "
            f"missing {self.missing_before:,}->{self.missing_after:,}"
        )
        extras = []
        if self.duplicates_removed:
            extras.append(f"{self.duplicates_removed} dup(s) removed")
        if self.outliers_handled:
            extras.append(f"{self.outliers_handled} outlier(s) handled")
        if self.columns_dropped:
            extras.append(f"dropped {len(self.columns_dropped)} column(s)")
        if extras:
            line += " (" + ", ".join(extras) + ")"
        if self.domain is not None:
            score = self.domain_trust_score if self.domain_trust_score is not None else 1.0
            line += f"\n  domain {self.domain}: trust {score:.2f}"
        for w in self.warnings:
            line += f"\n  warning: {w}"
        return line

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"<CleanReport: {len(self.actions)} actions, "
            f"rows {self.rows_before:,}->{self.rows_after:,}, "
            f"cols {self.cols_before}->{self.cols_after}>"
        )
