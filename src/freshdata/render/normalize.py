"""Report objects → :class:`~freshdata.render.view.PeelView`.

One pure function per ``_render_kind``. Normalizers read only the report
object — never a DataFrame — and are deterministic: the same report always
yields the same view, including attention ordering and finding ids.

Finding-id prefixes are per source so terminal, notebook, and programmatic
access all name the same thing:

``W`` report warnings · ``R`` recommendations · ``S`` suggested/review actions
· ``D`` domain findings · ``C`` contract findings · ``F`` fallback events
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from . import _semantic
from ._vocabulary import plain_step
from .view import AttentionItem, Metric, PeelView, Section, rank_attention

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..report import CleanReport

#: Normalizers by ``_render_kind``; extended by later layers (parse, copilot).
_NORMALIZERS: dict[str, Callable[[Any], PeelView]] = {}


def register_normalizer(kind: str, fn: Callable[[Any], PeelView]) -> None:
    """Register *fn* as the normalizer for ``_render_kind == kind``.

    Third-party report objects use this the same way built-ins do (spec §12.3).
    """
    _NORMALIZERS[kind] = fn


def normalize(obj: Any) -> PeelView:
    """Return the PeelView for *obj*, dispatching on its ``_render_kind``."""
    kind = getattr(obj, "_render_kind", "")
    fn = _NORMALIZERS.get(kind)
    if fn is None:
        raise KeyError(f"no Peel normalizer registered for kind {kind!r}")
    return fn(obj)


# -- clean_report ------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds >= 0.05 else f"{seconds * 1000:.0f}ms"


def _headline(rep: CleanReport) -> str:
    if not rep.materialized:
        return (
            f"{rep.rows_before:,} rows in · result kept in the engine "
            f"(counts not computed to avoid a full scan) · {_fmt_duration(rep.duration_seconds)}"
        )
    parts = [f"{rep.rows_after:,} of {rep.rows_before:,} rows kept"]
    if rep.cols_after != rep.cols_before:
        parts.append(f"{rep.cols_before:,} → {rep.cols_after:,} columns")
    else:
        parts.append(f"{rep.cols_after:,} columns")
    parts.append(f"{rep.cells_changed:,} cells changed")
    parts.append(_fmt_duration(rep.duration_seconds))
    return " · ".join(parts)


def _statuses(rep: CleanReport, attention: tuple[AttentionItem, ...]) -> tuple[str, ...]:
    statuses = ["CHANGED" if bool(rep) else "CLEAN"]
    if any(a.severity in ("error", "warning", "review") for a in attention):
        statuses.append("REVIEW")
    if not rep.materialized:
        statuses.append("PARTIAL")
    return tuple(statuses)


def _metrics(rep: CleanReport) -> tuple[Metric, ...]:
    if not rep.materialized:
        return (Metric("rows in", f"{rep.rows_before:,}"),)
    metrics = [
        Metric(
            "missing",
            f"{rep.missing_after:,}",
            before=f"{rep.missing_before:,}",
            after=f"{rep.missing_after:,}",
        )
    ]
    if rep.duplicates_removed:
        metrics.append(Metric("duplicates", f"-{rep.duplicates_removed:,}"))
    if rep.outliers_handled:
        metrics.append(Metric("extreme values adjusted", f"{rep.outliers_handled:,}"))
    if rep.columns_preserved:
        metrics.append(Metric("protected", f"{len(rep.columns_preserved)}"))
    if rep.columns_dropped:
        metrics.append(Metric("columns dropped", f"{len(rep.columns_dropped)}"))
    return tuple(metrics)


def _clean_attention(rep: CleanReport) -> tuple[AttentionItem, ...]:
    items: list[AttentionItem] = []

    for i, message in enumerate(rep.warnings, 1):
        items.append(
            AttentionItem(
                id=f"W{i}",
                severity="warning",
                subject="",
                text=message,
                domain="reliability",
                detail={"source": "warnings"},
            )
        )
    for i, message in enumerate(rep.recommendations, 1):
        items.append(
            AttentionItem(
                id=f"R{i}",
                severity="review",
                subject="",
                text=message,
                domain="reliability",
                detail={"source": "recommendations"},
            )
        )

    suggested = [
        a
        for a in rep.actions
        if a.status == "suggested"
        or (a.human_review and a.count)
        # ambiguous semantic proposals made no change but still want a human's eye
        or (_semantic.is_semantic(a) and a.status == "skipped")
    ]
    for i, action in enumerate(suggested, 1):
        if _semantic.is_semantic(action):
            text = _semantic.attention_text(action)
        else:
            text = f"{action.description} — held for your review"
        items.append(
            AttentionItem(
                id=f"S{i}",
                severity="review",
                subject=action.column or "",
                text=text,
                domain="corruption" if action.risk == "high" else "reliability",
                count=action.count,
                detail={"source": "actions", "action": rep._action_dict(action)},
            )
        )

    for i, finding in enumerate(rep.domain_findings, 1):
        if finding.get("status") != "violated":
            continue
        severity = "error" if finding.get("severity") == "error" else "warning"
        items.append(
            AttentionItem(
                id=f"D{i}",
                severity=severity,
                subject=str(finding.get("column") or finding.get("rule") or ""),
                text=str(finding.get("message") or finding.get("rule") or "domain rule violated"),
                domain="policy",
                count=int(finding.get("count") or 0),
                detail={"source": "domain_findings", "finding": finding},
            )
        )

    cv = rep.contract_violations
    if cv is not None and not cv.get("passed", True):
        for i, finding in enumerate(cv.get("findings", []), 1):
            if finding.get("status") == "passed":
                continue
            severity = "error" if finding.get("status") == "failed" else "warning"
            items.append(
                AttentionItem(
                    id=f"C{i}",
                    severity=severity,
                    subject=str(finding.get("column") or ""),
                    text=(
                        f"contract '{cv.get('baseline_name')}': "
                        f"{finding.get('message') or finding.get('check_id')}"
                    ),
                    domain="policy",
                    detail={"source": "contract_violations", "finding": finding},
                )
            )

    for i, event in enumerate(rep.fallback_events, 1):
        items.append(
            AttentionItem(
                id=f"F{i}",
                severity="info",
                subject=str(event.get("fallback_step") or ""),
                text=(
                    "FreshData continued without the optional engine step "
                    f"({event.get('backend')}: {event.get('fallback_reason')})"
                ),
                domain="reliability",
                detail={"source": "fallback_events", "event": event},
            )
        )

    return rank_attention(items)


def _next_step(rep: CleanReport, attention: tuple[AttentionItem, ...]) -> str | None:
    if not attention or attention[0].severity == "info":
        return None
    top = attention[0]
    source = top.detail.get("source")
    if source == "contract_violations":
        return "report.contract_violations  # review contract failures"
    if source == "domain_findings":
        return "report.domain_findings  # review domain rule violations"
    if source == "actions":
        return "fd.suggest_plan(df)  # review and approve the suggested changes"
    return "fd.explain_clean(df)  # see the evidence behind each decision"


def _column_rows(rep: CleanReport) -> list[dict[str, Any]]:
    """Per-column aggregation of the action log, most-changed first."""
    by_column: dict[str, dict[str, Any]] = {}
    risk_rank = {"low": 0, "medium": 1, "high": 2}
    for action in rep.actions:
        key = action.column if action.column is not None else "(table)"
        row = by_column.setdefault(
            key, {"column": key, "changed": 0, "what": [], "risk": "low"}
        )
        row["changed"] += action.count
        row["what"].append(plain_step(action))
        if risk_rank.get(action.risk, 0) > risk_rank.get(row["risk"], 0):
            row["risk"] = action.risk
    for name in rep.columns_preserved:
        row = by_column.setdefault(
            name, {"column": name, "changed": 0, "what": [], "risk": "low"}
        )
        row["protected"] = True
    rows = list(by_column.values())
    rows.sort(key=lambda r: (-r["changed"], r["column"]))
    for row in rows:
        row["what"] = "; ".join(dict.fromkeys(row["what"])) or (
            "protected — left untouched" if row.get("protected") else ""
        )
    return rows


def _audit_rows(rep: CleanReport) -> list[dict[str, Any]]:
    keys = (
        "backend",
        "materialized",
        "duration_seconds",
        "memory_before",
        "memory_after",
        "fallback_events",
        "backend_differences",
        "stage_timings",
        "domain",
        "domain_trust_score",
        "domain_repairs",
        "streaming",
        "profile_replay",
        "source_provenance",
        "contract_violations",
        "decisions_hash",
    )
    rows = []
    for key in keys:
        value = getattr(rep, key)
        if value not in (None, [], {}):
            rows.append({"field": key, "value": value})
    return rows


def normalize_clean_report(rep: CleanReport) -> PeelView:
    """Normalize a :class:`~freshdata.report.CleanReport` (spec §6)."""
    attention = _clean_attention(rep)
    banner = None
    if not rep.materialized:
        banner = (
            "PARTIAL — result kept in the engine; row counts not computed "
            "to avoid a full scan. Call .fetchdf()/.collect() to pull rows."
        )
    sections_list = [
        Section(
            "columns",
            "Column changes",
            lambda: _column_rows(rep),
            count=len({a.column for a in rep.actions if a.column is not None}),
        ),
    ]
    semantic_counts = _semantic.count_by_decision(rep)
    n_semantic = sum(semantic_counts.values())
    if n_semantic:
        note = _semantic.coverage_note(rep)
        title = (
            f"Semantic proposals ({semantic_counts['applied']} applied · "
            f"{semantic_counts['review']} review · {semantic_counts['ambiguous']} no change)"
        )
        if note:
            title = f"{title} — {note}"
        sections_list.append(
            Section("semantic", title, lambda: _semantic.semantic_rows(rep), count=n_semantic)
        )
    sections_list.extend(
        [
            Section(
                "actions",
                "All actions",
                lambda: [rep._action_dict(a) for a in rep.actions],
                count=len(rep.actions),
            ),
            Section("audit", "Audit", lambda: _audit_rows(rep), count=len(_audit_rows(rep))),
        ]
    )
    sections = tuple(sections_list)
    return PeelView(
        kind="clean_report",
        status=_statuses(rep, attention),
        headline=_headline(rep),
        metrics=_metrics(rep),
        attention=attention,
        next_step=_next_step(rep, attention),
        sections=sections,
        audit_ref=rep,
        banner=banner,
    )


register_normalizer("clean_report", normalize_clean_report)


# -- parse -------------------------------------------------------------------


def _frame_status(name: str, n_rows: int, warnings: list[str]) -> tuple[str, str]:
    """(machine, human) status for one parsed frame."""
    hits = [w for w in warnings if _warning_mentions(w, name)]
    if n_rows == 0:
        return "unsupported", "unsupported items found"
    if hits:
        return "warnings", f"{len(hits)} warning(s)"
    return "ready", "ready"


def _warning_mentions(warning: str, frame: str) -> bool:
    low = warning.lower()
    return low.startswith(f"{frame.lower()}:") or f"{frame.lower()} " in low


def _stage_ladder() -> str:
    # parse() only reaches the first rung; later stages run under fd.clean(...).
    return "parsed ✓ · validated — · cleaned — · reference —"


def normalize_parse_result(res: Any) -> PeelView:
    """Normalize a :class:`~freshdata.parsers.base.ParseResult` (spec §9).

    Reads only frame *row counts* (``len``) — never frame contents.
    """
    warnings = list(res.warnings)
    frame_rows = {name: len(df) for name, df in res.frames.items()}
    total_rows = sum(frame_rows.values())
    empty_frames = [n for n, r in frame_rows.items() if r == 0]
    partial = bool(warnings or empty_frames)

    status = ("PARTIAL",) if partial else ("CLEAN",)
    headline = (
        f"{len(res.frames)} frame(s) · {total_rows:,} rows total"
        + (f" · {len(warnings)} warning(s)" if warnings else "")
    )

    metrics = [Metric("stage", _stage_ladder())]
    if res.suggested_domain:
        metrics.append(Metric("suggested domain", f"{res.suggested_domain} (advisory)"))

    items: list[AttentionItem] = []
    wi = 0
    for name in res.frames:
        n_rows = frame_rows[name]
        machine, _human = _frame_status(name, n_rows, warnings)
        if machine == "unsupported":
            wi += 1
            items.append(
                AttentionItem(
                    id=f"P{wi}",
                    severity="warning",
                    subject=name,
                    text="0 rows — unsupported items were skipped and preserved in warnings",
                    domain="reliability",
                    detail={"source": "frames", "frame": name},
                )
            )
    for warning in warnings:
        if any(_warning_mentions(warning, n) and frame_rows[n] == 0 for n in res.frames):
            continue  # already represented by the frame's unsupported item
        wi += 1
        items.append(
            AttentionItem(
                id=f"P{wi}",
                severity="warning",
                subject="",
                text=warning,
                domain="reliability",
                detail={"source": "warnings"},
            )
        )
    attention = rank_attention(items)

    banner = None
    if partial:
        banner = (
            "PARTIAL — this data has been read, not checked. "
            "Structural parsing succeeded; validation and cleaning have not run."
        )

    next_step = None
    if res.suggested_domain and res.frames:
        first = next(iter(res.frames))
        next_step = f'fd.clean(result.frames["{first}"], domain="{res.suggested_domain}")'

    def frame_rows_section() -> list[dict[str, Any]]:
        rows = []
        for name in res.frames:
            machine, human = _frame_status(name, frame_rows[name], warnings)
            rows.append({"frame": name, "rows": frame_rows[name], "status": human})
        return rows

    sections = (
        Section("frames", "Frames", frame_rows_section, count=len(res.frames)),
        Section(
            "warnings",
            "All warnings",
            lambda: [{"warning": w} for w in warnings],
            count=len(warnings),
        ),
        Section(
            "metadata",
            "Format metadata",
            lambda: [{"key": k, "value": v} for k, v in dict(res.metadata).items()],
            count=len(res.metadata),
        ),
    )

    return PeelView(
        kind="parse",
        status=status,
        headline=headline,
        metrics=tuple(metrics),
        attention=attention,
        next_step=next_step,
        sections=sections,
        audit_ref=res,
        banner=banner,
    )


register_normalizer("parse", normalize_parse_result)


# -- copilot -----------------------------------------------------------------

_PROBLEM_SEVERITY = {"high": "error", "medium": "warning", "low": "info"}
_PROBLEM_DOMAIN = {"high": "corruption", "medium": "reliability", "low": "cosmetic"}


def normalize_copilot_report(rep: Any) -> PeelView:
    """Normalize a :class:`~freshdata.experimental.ai_copilot.CopilotReport`.

    The attention queue is ranked by the shared comparator, so privacy and
    policy findings always sit above data-quality ones — a high trust score can
    never bury them (spec §8). IDs (``A1``…) are assigned after ranking.
    """
    items: list[AttentionItem] = []

    if rep.pii_warning:
        columns = rep.audit.get("pii_columns") or rep.audit.get("pii_entities_found")
        items.append(
            AttentionItem(
                id="_pii",
                severity="error",
                subject="",
                text=str(rep.pii_warning),
                domain="privacy",
                detail={"source": "pii_warning", "pii_columns": columns},
            )
        )
    for i, violation in enumerate(rep.policy_violations):
        text = (
            violation.to_dict()
            if hasattr(violation, "to_dict")
            else {"detail": str(violation)}
        )
        message = text.get("message") or text.get("detail") or str(violation)
        items.append(
            AttentionItem(
                id=f"_pol{i}",
                severity="error",
                subject=str(text.get("column") or ""),
                text=f"policy: {message}",
                domain="policy",
                detail={"source": "policy_violations", "violation": text},
            )
        )
    for i, problem in enumerate(rep.problems):
        items.append(
            AttentionItem(
                id=f"_prob{i}",
                severity=_PROBLEM_SEVERITY.get(problem.severity, "warning"),
                subject=str(problem.column or ""),
                text=str(problem.detail),
                domain=_PROBLEM_DOMAIN.get(problem.severity, "reliability"),
                count=int(problem.count or 0),
                detail={"source": "problems", "problem": problem.to_dict()},
            )
        )

    ranked = rank_attention(items)
    attention = tuple(
        AttentionItem(
            id=f"A{i}",
            severity=item.severity,
            subject=item.subject,
            text=item.text,
            domain=item.domain,
            count=item.count,
            detail=item.detail,
        )
        for i, item in enumerate(ranked, 1)
    )

    provider_error = rep.audit.get("provider_error")
    status: list[str] = ["REVIEW"] if attention else ["CLEAN"]
    if provider_error:
        status.append("PARTIAL")

    trust = rep.trust
    headline = (
        f'goal "{rep.goal}" · trust {trust.overall:.0f}/100 ({trust.grade}) · '
        f"{len(attention)} finding(s)"
    )

    pii_state = (
        "PII found — masked before the model saw any data"
        if rep.pii_warning
        else "no PII detected — nothing was masked"
    )
    metrics = (
        Metric("privacy", pii_state),
        Metric("trust", f"{trust.overall:.0f}/100 ({trust.grade}) — data quality only"),
        Metric(
            "dimensions",
            f"completeness {trust.completeness:.0f} · validity {trust.validity:.0f} · "
            f"uniqueness {trust.uniqueness:.0f} · consistency {trust.consistency:.0f}",
        ),
    )

    banner = "Experimental API — review all generated code before running."
    if provider_error:
        banner += (
            "  Model narrative unavailable (provider error) — the deterministic "
            "findings above are complete."
        )

    next_step = None
    if rep.cleaning_plan.steps:
        next_step = str(rep.cleaning_plan.steps[0].tool)

    def plan_rows() -> list[dict[str, Any]]:
        return [
            {"order": s.order, "action": s.action, "why": s.rationale, "tool": s.tool}
            for s in rep.cleaning_plan.steps
        ]

    def trust_columns() -> list[dict[str, Any]]:
        rows = []
        for col in getattr(trust, "columns", ()):  # ColumnTrust tuple
            if hasattr(col, "to_dict"):
                rows.append(dict(col.to_dict()))
        return rows

    sections = (
        Section("plan", "Cleaning plan", plan_rows, count=len(rep.cleaning_plan.steps)),
        Section(
            "code",
            "Recommended code (machine-generated — review before running)",
            lambda: [{"code": rep.recommended_code}],
            count=1 if rep.recommended_code else 0,
        ),
        Section("trust_columns", "Per-column trust", trust_columns,
                count=len(getattr(trust, "columns", ()))),
        Section(
            "model_context",
            "Model context (masked — the only payload a provider sees)",
            lambda: [{"key": k, "value": v} for k, v in dict(rep.model_context).items()],
            count=len(rep.model_context),
        ),
        Section(
            "audit",
            "Audit",
            lambda: [{"field": k, "value": v} for k, v in dict(rep.audit).items()],
            count=len(rep.audit),
        ),
    )

    return PeelView(
        kind="copilot",
        status=tuple(status),
        headline=headline,
        metrics=metrics,
        attention=attention,
        next_step=next_step,
        sections=sections,
        audit_ref=rep,
        banner=banner,
    )


register_normalizer("copilot", normalize_copilot_report)
