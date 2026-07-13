"""Plain-text Peel renderer — no ANSI, works everywhere (spec §11).

This is the reference renderer: the rich terminal renderer and the notebook
renderer must present the same information; only styling may differ. Because
every state is a text label, output stays grep-able in logs and CI
(``grep REVIEW``, ``grep PARTIAL``).
"""

from __future__ import annotations

import json

from .options import RenderOptions, get_display
from .view import PeelView, Section

#: view.kind → command-style title.
_TITLES = {
    "clean_report": "freshdata clean",
    "profile": "freshdata profile",
    "parse": "freshdata parse",
    "copilot": "freshdata ai-copilot",
}

_SEVERITY_LABELS = {
    "error": "Error",
    "warning": "Warning",
    "review": "Review",
    "info": "Info",
}


def _title(view: PeelView) -> str:
    return _TITLES.get(view.kind, f"freshdata {view.kind}")


def _sep(options: RenderOptions) -> str:
    return " - " if options.ascii_icons else " · "


def _status_text(view: PeelView, options: RenderOptions) -> str:
    return (" - " if options.ascii_icons else " · ").join(view.status)


def _rule(text_left: str, text_right: str, options: RenderOptions) -> str:
    dash = "-" if options.ascii_icons else "─"
    body = f"{dash * 2} {text_left} "
    right = f" {text_right} {dash * 2}" if text_right else dash * 2
    fill = max(1, options.width - len(body) - len(right))
    return body + dash * fill + right


def _attention_lines(view: PeelView, options: RenderOptions) -> list[str]:
    if not view.attention:
        return [" nothing needs review"]
    lines = [f" Needs attention ({len(view.attention)})"]
    subject_width = min(12, max((len(a.subject) for a in view.attention), default=0))
    for item in view.attention:
        subject = item.subject[:subject_width].ljust(subject_width)
        label = _SEVERITY_LABELS.get(item.severity, item.severity).ljust(8)
        lines.append(f"  {label} {subject} {item.text}  [{item.id}]".rstrip())
    return lines


def _section_lines(section: Section) -> list[str]:
    lines = [f" {section.title} ({section.count})"]
    for row in section.rows():
        cells = ", ".join(f"{k}={_cell(v)}" for k, v in row.items() if v not in ("", None, []))
        lines.append(f"  - {cells}")
    return lines


def _cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


def render_plain(view: PeelView, options: RenderOptions | None = None) -> str:
    """Render *view* as plain text in the mode carried by *options*."""
    options = options or get_display()
    mode = options.resolved_mode()

    if mode == "silent":
        return ""
    if mode == "json":
        report = view.audit_ref
        if report is not None and hasattr(report, "to_dict"):
            return json.dumps(report.to_dict(), indent=2, default=str)
        return json.dumps({"kind": view.kind, "status": list(view.status)}, indent=2)

    sep = _sep(options)

    if mode == "compact":
        pieces = [
            _title(view),
            _status_text(view, options),
            view.headline,
        ]
        if view.attention:
            ids = " ".join(a.id for a in view.attention[:3])
            pieces.append(f"{len(view.attention)} attention ({ids})")
        first = "  ".join(pieces)
        return first + "\n  -> report.show() for details"

    # standard / verbose / debug / plain share the panel layout
    lines = [_rule(_title(view), _status_text(view, options), options)]
    lines.append(f" {view.headline}")
    if view.metrics:
        chips = []
        for metric in view.metrics:
            if metric.before is not None:
                arrow = "->" if options.ascii_icons else "→"
                chips.append(f"{metric.label} {metric.before}{arrow}{metric.after}")
            else:
                chips.append(f"{metric.label} {metric.value}")
        lines.append(" " + sep.join(chips).strip())
    if view.banner:
        lines.append("")
        lines.append(f" !! {view.banner}")
    lines.append("")
    lines.extend(_attention_lines(view, options))
    lines.append("")
    if view.next_step:
        lines.append(f" next   {view.next_step}")
    lines.append(f" more   report.show(mode=\"verbose\"){sep}report.to_json()")

    if mode in ("verbose", "debug"):
        for section in view.sections:
            if section.key == "audit" and mode != "debug":
                continue
            lines.append("")
            lines.extend(_section_lines(section))

    dash = "-" if options.ascii_icons else "─"
    lines.append(dash * options.width)
    out = "\n".join(lines)
    if options.ascii_icons:
        out = _asciify(out)
    return out


def _asciify(text: str) -> str:
    """Replace Peel's framing glyphs with ASCII (plain mode / ASCII icon set)."""
    for glyph, ascii_form in (("·", "-"), ("→", "->"), ("—", "--"), ("─", "-")):
        text = text.replace(glyph, ascii_form)
    return text
