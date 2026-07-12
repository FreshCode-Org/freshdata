"""Peel notebook renderer — PeelView → self-contained HTML (spec §10).

Opt-in for now: ``fd.set_display("peel")`` (or ``FRESHDATA_DISPLAY=peel``)
switches ``_repr_html_``/``to_html`` to this renderer for objects that have a
Peel normalizer; everything else keeps the legacy layout. Disclosure uses
native ``<details>`` so static exports and no-JS viewers keep full access.

Everything user-provided is escaped via :func:`freshdata.render.html.esc`;
values never land in HTML attributes.
"""

from __future__ import annotations

import json

from . import html as H
from .options import RenderOptions, get_display
from .view import AttentionItem, PeelView, Section

#: severity → chip color (labels always accompany color; see spec §14.1).
_SEVERITY_COLORS = {
    "error": "#cf222e",
    "warning": "#9a6700",
    "review": "#0969da",
    "info": "#57606a",
}

_STATUS_COLORS = {
    "CLEAN": "#1a7f37",
    "CHANGED": "#0969da",
    "REVIEW": "#9a6700",
    "BLOCKED": "#cf222e",
    "PARTIAL": "#9a6700",
    "SKIPPED": "#57606a",
    "FAILED": "#cf222e",
}

#: Cap on rows rendered per inspect section; the full data stays reachable
#: through to_dict()/to_json() and verbose text modes (spec §16).
_MAX_SECTION_ROWS = 50


def _status_chips(view: PeelView) -> str:
    chips = "".join(
        H.badge(status, _STATUS_COLORS.get(status, "#57606a")) for status in view.status
    )
    return f'<div class="fd-meta">{chips}</div>'


def _metric_strip(view: PeelView) -> str:
    if not view.metrics:
        return ""
    items = []
    for metric in view.metrics:
        if metric.before is not None:
            items.append((metric.label, f"{metric.before} → {metric.after}"))
        else:
            items.append((metric.label, metric.value))
    return H.scorecards(items)


def _attention_html(items: tuple[AttentionItem, ...]) -> str:
    if not items:
        return '<p class="fd-meta">nothing needs review</p>'
    rows = []
    for item in items:
        chip = H.badge(item.severity.capitalize(), _SEVERITY_COLORS.get(item.severity, "#57606a"))
        subject = f"<b>{H.esc(item.subject)}</b> " if item.subject else ""
        rows.append(
            f'<li>{chip} {subject}{H.esc(item.text)} '
            f'<span class="fd-meta">[{H.esc(item.id)}]</span></li>'
        )
    return (
        f"<h3>Needs attention ({len(items)})</h3>"
        f'<ul class="fd-attention" style="list-style:none;padding-left:0;margin:.3rem 0">'
        + "".join(rows)
        + "</ul>"
    )


def _next_step_html(view: PeelView) -> str:
    if not view.next_step:
        return ""
    return (
        '<p class="fd-meta">next → '
        f"<code>{H.esc(view.next_step)}</code></p>"
    )


def _banner_html(view: PeelView) -> str:
    if not view.banner:
        return ""
    return (
        '<p style="border-left:4px solid #9a6700;padding:.3rem .6rem;margin:.4rem 0">'
        f"<b>{H.esc(view.banner)}</b></p>"
    )


def _section_html(section: Section) -> str:
    rows = section.rows()
    shown = rows[:_MAX_SECTION_ROWS]
    columns: list[str] = []
    for row in shown:
        for key in row:
            if key not in columns:
                columns.append(key)
    body = H.table(columns, [[_cell(row.get(key)) for key in columns] for row in shown])
    if len(rows) > len(shown):
        body += (
            f'<p class="fd-meta">showing {len(shown)} of {len(rows)} — '
            "use <code>report.to_json()</code> or "
            '<code>report.show(mode="verbose")</code> for all</p>'
        )
    summary = f"{H.esc(section.title)} ({section.count})"
    return H.collapsible(summary, body)


def _cell(value: object) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


def _export_html(view: PeelView) -> str:
    report = view.audit_ref
    if report is None or not hasattr(report, "to_dict"):
        return ""
    try:
        return H.json_download(f"freshdata_{view.kind}.json", report.to_dict())
    except Exception:  # pragma: no cover - export must never break display
        return ""


def render_notebook(view: PeelView, options: RenderOptions | None = None) -> str:
    """Render *view* as a self-contained Peel HTML fragment."""
    options = options or get_display()
    parts = [
        _status_chips(view),
        _banner_html(view),
        _metric_strip(view),
        _attention_html(view.attention),
        _next_step_html(view),
    ]
    parts.extend(_section_html(section) for section in view.sections)
    parts.append(_export_html(view))
    from .plain import _TITLES

    title = _TITLES.get(view.kind, f"freshdata {view.kind}")
    return H.document(title, *parts, subtitle=view.headline)
