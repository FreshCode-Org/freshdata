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
