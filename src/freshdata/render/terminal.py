"""Styled terminal renderer over :class:`PeelView`, using ``rich`` when present.

``rich`` is an optional extra (``pip install freshdata-cleaner[rich]``). Without it —
or for the text-first modes (``compact``/``json``/``plain``/``silent``) — this
module delegates to the plain renderer, which is the reference for content:
the styled output may only differ in styling, never in information.
"""

from __future__ import annotations

from typing import Any

from .options import RenderOptions, get_display
from .plain import render_plain
from .view import PeelView

_SEVERITY_STYLES = {
    "error": "bold red",
    "warning": "yellow",
    "review": "cyan",
    "info": "dim",
}

_TITLES_STYLE = "bold"


def render_terminal_text(view: PeelView, options: RenderOptions | None = None) -> str:
    """Render *view* for a terminal, returning the final text (ANSI included
    only when color is enabled). Falls back to plain text when ``rich`` is not
    installed."""
    options = options or get_display()
    mode = options.resolved_mode()
    if mode in ("compact", "json", "plain", "silent"):
        return render_plain(view, options)
    try:
        return _render_rich(view, options, mode)
    except ImportError:
        return render_plain(view, options)


def _render_rich(view: PeelView, options: RenderOptions, mode: str) -> str:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    color = options.color != "never"
    console = Console(
        record=True,
        width=options.width,
        force_terminal=color,
        no_color=not color,
        highlight=False,
    )

    body: list[Any] = [Text(view.headline)]
    if view.metrics:
        chips = []
        for metric in view.metrics:
            if metric.before is not None:
                chips.append(f"{metric.label} {metric.before}→{metric.after}")
            else:
                chips.append(f"{metric.label} {metric.value}")
        body.append(Text(" · ".join(chips), style="dim"))
    if view.banner:
        body.append(Text(f"!! {view.banner}", style="bold yellow"))

    body.append(Text())
    if view.attention:
        table = Table(
            title=f"Needs attention ({len(view.attention)})",
            title_justify="left",
            show_header=False,
            box=None,
            pad_edge=False,
        )
        table.add_column("severity", style="bold")
        table.add_column("subject")
        table.add_column("text", overflow="fold")
        table.add_column("id", style="dim")
        for item in view.attention:
            table.add_row(
                Text(item.severity.capitalize(), style=_SEVERITY_STYLES.get(item.severity, "")),
                item.subject,
                item.text,
                f"[{item.id}]",
            )
        body.append(table)
    else:
        body.append(Text("nothing needs review", style="dim"))

    tail: list[str] = []
    if view.next_step:
        tail.append(f"next   {view.next_step}")
    tail.append('more   report.show(mode="verbose") · report.to_json()')
    body.append(Text())
    body.append(Text("\n".join(tail), style="dim"))

    if mode in ("verbose", "debug"):
        for section in view.sections:
            if section.key == "audit" and mode != "debug":
                continue
            table = Table(
                title=f"{section.title} ({section.count})", title_justify="left", box=None
            )
            rows = section.rows()
            columns: list[str] = []
            for row in rows:
                for key in row:
                    if key not in columns:
                        columns.append(key)
            for key in columns:
                table.add_column(key)
            for row in rows:
                table.add_row(*(_cell(row.get(key)) for key in columns))
            body.append(Text())
            body.append(table)

    console.print(
        Panel(
            Group(*body),
            title=Text(_panel_title(view), style=_TITLES_STYLE),
            subtitle=view.status_label,
            title_align="left",
            subtitle_align="right",
        )
    )
    return console.export_text(styles=color)


def _panel_title(view: PeelView) -> str:
    from .plain import _TITLES

    return _TITLES.get(view.kind, f"freshdata {view.kind}")


def _cell(value: object) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
