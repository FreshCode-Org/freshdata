"""Per-object HTML renderers, dispatched by ``_render_kind``.

Each function takes an already-computed report object (or DataFrame) and returns
a self-contained HTML fragment built from :mod:`freshdata.render.html`. No object
state is mutated; nothing here is imported by ``import freshdata``.
"""

from __future__ import annotations

import contextlib
from typing import Any

from . import html as H


def render(obj: Any, kind: str) -> str:
    fn = _DISPATCH.get(kind)
    if fn is None:
        raise ValueError(f"no renderer registered for kind {kind!r}")
    return fn(obj)


# -- clean report: action timeline + audit ledger ---------------------------

def _action_status(a: Any) -> str:
    return getattr(a, "status", "automatic") or "automatic"


def render_clean_report(report: Any) -> str:
    materialized = getattr(report, "materialized", True)
    after_rows = "—" if not materialized else f"{report.rows_after:,}"
    cards = H.scorecards([
        ("rows", f"{report.rows_before:,} → {after_rows}"),
        ("columns", f"{report.cols_before} → {report.cols_after if materialized else '—'}"),
        ("missing cells", f"{report.missing_before:,} → "
                          f"{report.missing_after:,}" if materialized else "—"),
        ("duplicates removed", report.duplicates_removed),
        ("outliers handled", report.outliers_handled),
        ("actions", len(report.actions)),
    ])

    # Collapsible action timeline — one row per step.
    items = []
    for a in report.actions:
        badges = H.risk_badge(a.risk) + " " + H.status_badge(_action_status(a))
        rev = getattr(a, "reversible", None)
        if rev is True:
            badges += " " + H.badge("reversible", "#1a7f37")
        elif rev is False:
            badges += " " + H.badge("irreversible", "#cf222e")
        if getattr(a, "memory_influenced", False):
            badges += " " + H.badge("memory", "#8250df")
        if getattr(a, "human_review", False):
            badges += " " + H.badge("review", "#9a6700")
        col = f" <span class='fd-mono'>{H.esc(a.column)}</span>" if a.column else ""
        summary = (f"<b>{H.esc(a.step)}</b>{col} — {H.esc(a.description)} "
                   f"<span class='fd-meta'>({a.count:,} affected)</span> {badges}")
        body = ""
        if a.rationale:
            body += f"<div class='fd-meta'>why: {H.esc(a.rationale)}</div>"
        body += ("<div class='fd-meta'>confidence</div>"
                 + H.bar(a.confidence) + f"<div class='fd-meta'>{a.confidence:.0%}</div>")
        items.append(H.collapsible(summary, body))
    timeline = H.section("Action timeline", "".join(items) or "<i>no changes</i>")

    # Interactive audit ledger from to_frame().
    ledger = ""
    try:
        frame = report.to_frame()
        if len(frame):
            rows = [
                [str(r["column"] or ""), str(r["step"]),
                 H.risk_badge(str(r["risk"])), f"{float(r['confidence']):.0%}",
                 str(int(r["count"])), str(r["description"])]
                for _, r in frame.iterrows()
            ]
            tbl = H.filterable_table(
                "fd-ledger",
                ["column", "action", "risk", "confidence", "count", "description"],
                rows,
                filters={"column": 0, "action": 1, "risk": 2},
                raw_columns=[2],
            )
            dl = (H.json_download("clean_ledger.json", report.to_dict(), "⬇ JSON")
                  + H.data_uri_download("clean_ledger.csv", frame.to_csv(index=False),
                                        "text/csv", "⬇ CSV"))
            ledger = H.section("Audit ledger", dl + tbl)
    except Exception:  # pragma: no cover - ledger is best-effort
        ledger = ""

    warns = ""
    if report.warnings:
        warns += H.section("Warnings", "<ul>" + "".join(
            f"<li>{H.esc(w)}</li>" for w in report.warnings) + "</ul>")
    if report.recommendations:
        warns += H.section("Needs review", "<ul>" + "".join(
            f"<li>{H.esc(r)}</li>" for r in report.recommendations) + "</ul>")

    sub = f"backend={report.backend or 'pandas'} · {report.duration_seconds:.3f}s"
    if not materialized:
        sub += " · result returned as a native handle (not materialized)"
    return H.document("freshdata clean report", cards, timeline, ledger, warns, subtitle=sub)


# -- profile: quality cockpit -----------------------------------------------

def _quality_score(profile: Any) -> int:
    completeness = 1.0 - (profile.missing_pct / 100.0)
    issue_penalty = min(0.3, 0.02 * profile.n_issues)
    dup = profile.duplicate_rows or 0
    dup_penalty = min(0.2, dup / profile.n_rows) if profile.n_rows else 0.0
    return max(0, round(100 * (completeness - issue_penalty - dup_penalty)))


def render_profile(profile: Any) -> str:
    score = _quality_score(profile)
    dup = "n/a" if profile.duplicate_rows is None else f"{profile.duplicate_rows:,}"
    cards = H.scorecards([
        ("quality score", f"{score}/100"),
        ("rows", f"{profile.n_rows:,}"),
        ("columns", f"{profile.n_cols:,}"),
        ("missing", f"{profile.missing_pct:.1f}%"),
        ("duplicate rows", dup),
        ("issues", profile.n_issues),
    ])

    # Type-inference summary: columns whose suggested dtype differs.
    retype = [c for c in profile.columns
              if c.suggested_dtype and c.suggested_dtype != c.dtype]
    type_summary = H.section(
        "Type inference",
        ("<ul>" + "".join(
            f"<li class='fd-mono'>{H.esc(c.name)}: {H.esc(c.dtype)} → "
            f"{H.esc(c.suggested_dtype)}</li>" for c in retype) + "</ul>")
        if retype else "<div class='fd-meta'>no dtype changes suggested</div>")

    # Issue-ranked column list (most issues / most missing first) with null bars.
    ranked = sorted(profile.columns,
                    key=lambda c: (len(c.issues), c.missing_pct), reverse=True)
    rows = []
    cardinality_warn, outlier_warn = [], []
    for c in ranked:
        if c.unique is not None and profile.n_rows:
            if c.unique == profile.n_rows and profile.n_rows > 1:
                cardinality_warn.append(f"{c.name} (all-unique)")
            elif c.unique == 1:
                cardinality_warn.append(f"{c.name} (constant)")
        if any("outlier" in i.lower() for i in c.issues):
            outlier_warn.append(c.name)
        rows.append([
            str(c.name), str(c.dtype),
            H.bar(c.missing_pct / 100.0, color="#cf222e") + f" {c.missing_pct:.0f}%",
            "n/a" if c.unique is None else f"{c.unique:,}",
            "; ".join(c.issues) or "—",
        ])
    col_table = H.section("Columns (issue-ranked)", H.filterable_table(
        "fd-profile-cols",
        ["column", "dtype", "missing", "unique", "issues"],
        rows, filters={"column": 0, "issues": 4}, raw_columns=[2]))

    warn_html = ""
    if cardinality_warn:
        warn_html += H.section("Cardinality warnings",
                               "<ul>" + "".join(f"<li>{H.esc(w)}</li>"
                                                for w in cardinality_warn) + "</ul>")
    if outlier_warn:
        warn_html += H.section("Outlier warnings",
                               "<div>" + H.esc(", ".join(outlier_warn)) + "</div>")

    # Correlation on demand — not computed by default (kept fast).
    corr = H.section("Correlations", "<div class='fd-meta'>Numeric correlations are "
                     "computed on demand to keep this view fast. Call "
                     "<span class='fd-mono'>profile.to_frame()</span> or compute "
                     "<span class='fd-mono'>df.corr()</span> when you need them.</div>")

    dl = H.json_download("profile.json", profile.to_dict(), "⬇ JSON")
    return H.document("freshdata profile", cards, type_summary, col_table,
                      warn_html, corr, dl,
                      subtitle="inline quality cockpit")


# -- suggest_plan: decision cards -------------------------------------------

def render_clean_plan(plan: Any) -> str:
    cards_html = []
    for col, cp in sorted(plan.column_plans.items()):
        choice = cp.missing
        needs_review = choice is not None and (
            not choice.eligible or bool(choice.rejection_reason))
        title = f"<b class='fd-mono'>{H.esc(col)}</b>"
        if needs_review:
            title += " " + H.badge("needs review", "#9a6700")
        body = ""
        if choice is not None:
            body += (f"<div>missing → <b>{H.esc(choice.model_id)}</b></div>"
                     + H.bar(choice.confidence)
                     + f"<div class='fd-meta'>confidence {choice.confidence:.0%}</div>")
            if choice.rationale:
                body += f"<div class='fd-meta'>why: {H.esc(choice.rationale)}</div>"
            alts = [a.model_id for a in cp.missing_alternatives]
            if alts:
                body += f"<div class='fd-meta'>alternatives: {H.esc(', '.join(alts))}</div>"
        if cp.outlier_action:
            body += (f"<div>outliers → <b>{H.esc(cp.outlier_action)}</b> "
                     f"<span class='fd-meta'>({cp.n_outliers} flagged)</span></div>")
        if not body:
            body = "<div class='fd-meta'>no engine action</div>"
        cards_html.append(
            f"<div class='fd-card' style='min-width:240px'>{title}{body}</div>")
    grid = f"<div class='fd-cards'>{''.join(cards_html)}</div>" if cards_html else \
        "<div class='fd-meta'>no engine actions (conservative or empty frame)</div>"
    dl = H.json_download("clean_plan.json", plan.to_dict(), "⬇ JSON")
    return H.document(
        f"freshdata clean plan — {plan.config.strategy}", grid, dl,
        subtitle="per-column decision cards")


# -- explain_clean: before/after diff explorer ------------------------------

def render_explain(rep: Any) -> str:
    cards = H.scorecards([
        ("shape", f"{rep.rows_before}×{rep.cols_before} → {rep.rows_after}×{rep.cols_after}"),
        ("missing", f"{rep.report.missing_before} → {rep.report.missing_after}"),
        ("steps", len(rep.report)),
        ("changed cells", sum(rep.cell_changes.values())),
    ])
    # Before/after column comparison with changed-cell counts; risky separated.
    safe_rows: list[list[str]] = []
    risky_rows: list[list[str]] = []
    for col, changes in sorted(rep.cell_changes.items(), key=lambda kv: kv[1], reverse=True):
        before = rep.before_stats.get(col, {})
        after = rep.after_stats.get(col, {})
        risky = bool(after.get("dtype") and before.get("dtype")
                     and after["dtype"] != before["dtype"]) or changes > 0.5 * max(
                         1, rep.rows_before)
        row = [str(col), str(before.get("dtype", "")), str(after.get("dtype", "")),
               str(changes)]
        (risky_rows if risky else safe_rows).append(row)
    headers = ["column", "before dtype", "after dtype", "changed cells"]
    safe = H.section("Safe changes", H.table(headers, safe_rows) if safe_rows
                     else "<div class='fd-meta'>none</div>")
    risky = H.section("Risky changes (review)", H.table(headers, risky_rows) if risky_rows
                      else "<div class='fd-meta'>none</div>")
    narr = ""
    if rep.narratives:
        narr = H.section("Decisions", "<ul>" + "".join(
            f"<li>{H.esc(n)}</li>" for n in rep.narratives) + "</ul>")
    dl = H.json_download("explain.json", rep.to_dict(), "⬇ JSON")
    return H.document(f"freshdata explain — {rep.strategy}", cards, narr, safe, risky, dl,
                      subtitle="before/after diff explorer")


# -- DataFrame-backed surfaces ----------------------------------------------

def _df_table(df: Any, table_id: str, filters: dict[str, int] | None = None) -> str:
    headers = [str(c) for c in df.columns]
    rows = [[("" if v is None else str(v)) for v in rec]
            for rec in df.itertuples(index=False, name=None)]
    return H.filterable_table(table_id, headers, rows, filters=filters or {})


def render_compare_plans(df: Any) -> str:
    filters = {}
    cols = list(df.columns)
    if "column" in cols:
        filters["column"] = cols.index("column")
    if "strategy" in cols:
        filters["strategy"] = cols.index("strategy")
    body = _df_table(df, "fd-cmp-plans", filters)
    return H.document("freshdata strategy comparison", body,
                      subtitle="strategy diff grid")


def render_compare_clean(df: Any) -> str:
    # Scorecards per strategy when the expected metric columns are present.
    cards = ""
    cols = list(df.columns)
    if "strategy" in cols:
        bits = []
        for rec in df.to_dict("records"):
            label = rec.get("strategy", "?")
            ma = rec.get("missing_after")
            dur = rec.get("duration_seconds")
            val = f"miss {ma}" if ma is not None else ""
            if dur is not None:
                val += f" · {float(dur):.2f}s"
            bits.append((str(label), val or "—"))
        cards = H.scorecards(bits)
    body = _df_table(df, "fd-cmp-clean", {"strategy": cols.index("strategy")}
                     if "strategy" in cols else {})
    return H.document("freshdata clean comparison", cards, body,
                      subtitle="outcome dashboard")


def render_infer_roles(df: Any) -> str:
    cards_html = []
    for rec in df.to_dict("records"):
        name = rec.get("column", "?")
        role = rec.get("role", "?")
        conf = rec.get("confidence")
        title = f"<b class='fd-mono'>{H.esc(name)}</b> {H.badge(str(role), '#0969da')}"
        body = ""
        if conf is not None:
            with contextlib.suppress(TypeError, ValueError):
                body += (H.bar(float(conf))
                         + f"<div class='fd-meta'>confidence {float(conf):.0%}</div>")
        evidence = {k: v for k, v in rec.items()
                    if k not in ("column", "role", "confidence") and v not in (None, "")}
        if evidence:
            body += "<div class='fd-meta'>" + H.esc("; ".join(
                f"{k}={v}" for k, v in evidence.items())) + "</div>"
        cards_html.append(
            f"<div class='fd-card' style='min-width:220px'>{title}{body}</div>")
    grid = f"<div class='fd-cards'>{''.join(cards_html)}</div>"
    return H.document("freshdata inferred roles", grid,
                      subtitle="role / type confidence cards")


def render_insight_report(report: Any) -> str:
    payload = report.to_dict()
    summary = payload["summary"]
    dataset = payload["dataset"]
    cards = H.scorecards([
        ("rows", f"{dataset.get('rows', 0):,}"),
        ("columns", f"{dataset.get('columns', 0):,}"),
        ("issues", summary.get("issue_count", 0)),
        ("actions", summary.get("action_count", 0)),
        ("highest severity", summary.get("highest_severity", "none")),
    ])

    issue_rows = []
    for issue in payload["issues"]:
        evidence = issue.get("evidence", {})
        issue_rows.append([
            str(issue.get("column") or ""),
            H.risk_badge(str(issue.get("severity", "low"))),
            str(issue.get("inferred_role") or "unknown"),
            str(issue.get("finding") or ""),
            f"<span class='fd-mono'>{H.esc(issue.get('fix_code') or '')}</span>",
            str(evidence.get("missing_pct", "")),
        ])
    issues = H.section(
        "Action intelligence",
        H.filterable_table(
            "fd-insight-issues",
            ["column", "severity", "role", "finding", "fix", "missing %"],
            issue_rows,
            filters={"column": 0, "finding": 3},
            raw_columns=[1, 4],
        )
        if issue_rows
        else "<div class='fd-meta'>no profile issues detected</div>",
    )

    action_rows = []
    for action in payload["actions"]:
        action_rows.append([
            str(action.get("column") or ""),
            str(action.get("step") or ""),
            H.risk_badge(str(action.get("risk", "low"))),
            f"{float(action.get('confidence', 0.0)):.0%}",
            str(action.get("count", 0)),
            str(action.get("description") or ""),
        ])
    actions = H.section(
        "CleanReport actions",
        H.table(
            ["column", "step", "risk", "confidence", "count", "description"],
            action_rows,
            raw_columns=[2],
        )
        if action_rows
        else "<div class='fd-meta'>pass clean_report= to attach applied actions</div>",
    )

    next_step = H.section(
        "Next step",
        f"<div class='fd-mono'>{H.esc(summary.get('recommended_next_step', ''))}</div>",
    )
    strategy = ""
    comparison = payload.get("strategy_comparison")
    if comparison:
        rows = [
            [
                str(row.get("strategy") or ""),
                str(row.get("column") or ""),
                str(row.get("missing_model") or ""),
                str(row.get("outlier_action") or ""),
                str(row.get("n_outliers") or 0),
            ]
            for row in comparison.get("records", [])
        ]
        body = (
            f"<div class='fd-meta fd-mono'>{H.esc(comparison.get('command'))}</div>"
            + H.table(["strategy", "column", "missing", "outliers", "n"], rows)
        )
        strategy = H.section("Strategy comparison", body)

    trust_section = ""
    trust = payload.get("trust")
    if trust:
        after = trust.get("after") or {}
        gate = trust.get("gate") or {}
        cards = H.scorecards([
            ("trust score", after.get("overall", "—")),
            ("grade", after.get("grade", "—")),
            ("gate", gate.get("passed")),
            ("threshold", gate.get("threshold", "—")),
        ])
        ci = payload.get("surfaces", {}).get("cli", {}).get("ci_summary", "")
        trust_section = H.section(
            "Trust gate",
            cards + (f"<pre class='fd-mono'>{H.esc(ci)}</pre>" if ci else ""),
        )

    dl = H.json_download("freshdata_insight.json", payload, "⬇ JSON")
    return H.document(
        "FreshData insight report",
        cards,
        issues,
        actions,
        strategy,
        trust_section,
        next_step,
        dl,
        subtitle="decision/action workspace",
    )


_DISPATCH = {
    "clean_report": render_clean_report,
    "profile": render_profile,
    "clean_plan": render_clean_plan,
    "explain": render_explain,
    "compare_plans": render_compare_plans,
    "compare_clean": render_compare_clean,
    "infer_roles": render_infer_roles,
    "insight_report": render_insight_report,
}
