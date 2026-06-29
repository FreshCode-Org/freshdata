"""Self-contained HTML building blocks for freshdata reports.

Pure stdlib. Every helper returns an HTML fragment string; objects compose them
inside :func:`document` so the result drops straight into a notebook
``_repr_html_`` or a standalone ``.html`` file. Styling is scoped under
``.fd-report`` so it never leaks into the surrounding page.
"""

from __future__ import annotations

import html as _html
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_RISK_COLOR = {"low": "#1a7f37", "medium": "#9a6700", "high": "#cf222e"}
_STATUS_COLOR = {
    "automatic": "#0969da",
    "suggested": "#8250df",
    "approved": "#1a7f37",
    "skipped": "#57606a",
}

#: Scoped stylesheet, injected once per fragment (duplicate <style> is harmless).
_CSS = """
<style>
.fd-report{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  color:#1f2328;line-height:1.45;max-width:980px}
.fd-report h2{font-size:1.15rem;margin:.2rem 0 .6rem}
.fd-report h3{font-size:.95rem;margin:1rem 0 .4rem;color:#57606a;
  text-transform:uppercase;letter-spacing:.03em}
.fd-cards{display:flex;flex-wrap:wrap;gap:.6rem;margin:.4rem 0}
.fd-card{border:1px solid #d0d7de;border-radius:8px;padding:.55rem .7rem;
  min-width:120px;background:#fff}
.fd-card .v{font-size:1.35rem;font-weight:600}
.fd-card .l{font-size:.72rem;color:#57606a;text-transform:uppercase;letter-spacing:.03em}
.fd-badge{display:inline-block;padding:.05rem .45rem;border-radius:999px;
  font-size:.72rem;font-weight:600;color:#fff}
.fd-bar{background:#eaeef2;border-radius:4px;height:8px;overflow:hidden;min-width:60px}
.fd-bar>span{display:block;height:100%;background:#0969da}
.fd-table{border-collapse:collapse;width:100%;font-size:.84rem;margin:.3rem 0}
.fd-table th,.fd-table td{border:1px solid #d0d7de;padding:.3rem .5rem;text-align:left;
  vertical-align:top}
.fd-table th{background:#f6f8fa;position:sticky;top:0}
.fd-table tr:nth-child(even){background:#f6f8fa}
.fd-report details{border:1px solid #d0d7de;border-radius:8px;padding:.35rem .6rem;
  margin:.35rem 0;background:#fff}
.fd-report summary{cursor:pointer;font-weight:600}
.fd-meta{color:#57606a;font-size:.8rem}
.fd-controls{margin:.4rem 0;font-size:.82rem}
.fd-controls input,.fd-controls select{padding:.2rem .35rem;margin-right:.4rem;
  border:1px solid #d0d7de;border-radius:6px}
.fd-del-pos{color:#1a7f37}.fd-del-neg{color:#cf222e}
.fd-mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.8rem}
</style>
"""


def esc(value: Any) -> str:
    """HTML-escape *value* (``None`` → empty string)."""
    return _html.escape("" if value is None else str(value))


def document(title: str, *sections: str, subtitle: str | None = None) -> str:
    """Wrap *sections* in the scoped ``.fd-report`` container with a heading."""
    sub = f'<div class="fd-meta">{esc(subtitle)}</div>' if subtitle else ""
    body = "\n".join(s for s in sections if s)
    return f'{_CSS}<div class="fd-report"><h2>{esc(title)}</h2>{sub}{body}</div>'


def scorecards(items: Sequence[tuple[str, Any]]) -> str:
    """A row of label/value cards, e.g. ``[("rows", 1000), ("score", "0.92")]``."""
    cards = "".join(
        f'<div class="fd-card"><div class="v">{esc(v)}</div>'
        f'<div class="l">{esc(label)}</div></div>'
        for label, v in items
    )
    return f'<div class="fd-cards">{cards}</div>'


def badge(text: str, color: str) -> str:
    return f'<span class="fd-badge" style="background:{color}">{esc(text)}</span>'


def risk_badge(risk: str) -> str:
    return badge(risk, _RISK_COLOR.get(str(risk).lower(), "#57606a"))


def status_badge(status: str) -> str:
    return badge(status, _STATUS_COLOR.get(str(status).lower(), "#57606a"))


def bar(fraction: float, *, color: str = "#0969da") -> str:
    """A 0..1 progress bar."""
    pct = max(0.0, min(1.0, float(fraction))) * 100
    return f'<div class="fd-bar"><span style="width:{pct:.0f}%;background:{color}"></span></div>'


def delta(value: float, *, fmt: str = "{:+,}", good_when_negative: bool = False) -> str:
    """A signed delta, green/red by direction."""
    cls = "fd-del-pos"
    if (value < 0) != good_when_negative:
        cls = "fd-del-neg" if value != 0 else ""
    return f'<span class="{cls}">{fmt.format(value)}</span>'


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    raw_columns: Sequence[int] = (),
) -> str:
    """An HTML table. Cells are escaped unless their index is in *raw_columns*
    (used for badges/bars that are already safe HTML)."""
    raw = set(raw_columns)
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{cell if i in raw else esc(cell)}</td>" for i, cell in enumerate(row)
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        f'<table class="fd-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )


def collapsible(summary_html: str, body_html: str, *, open_: bool = False) -> str:
    o = " open" if open_ else ""
    return f"<details{o}><summary>{summary_html}</summary>{body_html}</details>"


def section(title: str, body_html: str) -> str:
    return f"<h3>{esc(title)}</h3>{body_html}"


def kv_list(mapping: Mapping[str, Any]) -> str:
    items = "".join(
        f"<li><b>{esc(k)}:</b> {esc(v)}</li>" for k, v in mapping.items()
    )
    return f'<ul style="margin:.2rem 0">{items}</ul>'


def filterable_table(
    table_id: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    filters: Mapping[str, int] | None = None,
    raw_columns: Sequence[int] = (),
) -> str:
    """A table with client-side text/select filters — no JS libraries needed.

    *filters* maps a label to the 0-based column index it filters (a free-text
    box). The whole thing is self-contained vanilla JS scoped by *table_id*.
    """
    tbl = table(headers, rows, raw_columns=raw_columns).replace(
        '<table class="fd-table">', f'<table class="fd-table" id="{esc(table_id)}">', 1
    )
    controls = ""
    js = ""
    if filters:
        boxes = "".join(
            f'<input data-col="{idx}" placeholder="filter {esc(label)}…" '
            f'oninput="fdFilter_{esc(table_id)}()">'
            for label, idx in filters.items()
        )
        controls = f'<div class="fd-controls">{boxes}</div>'
        js = (
            f"<script>function fdFilter_{table_id}(){{"
            f"var t=document.getElementById('{table_id}');"
            "var inp=t.parentNode.querySelectorAll('.fd-controls input');"
            "var rows=t.tBodies[0].rows;"
            "for(var i=0;i<rows.length;i++){var show=true;"
            "inp.forEach(function(b){var c=+b.dataset.col,q=b.value.toLowerCase();"
            "if(q&&rows[i].cells[c].innerText.toLowerCase().indexOf(q)<0)show=false;});"
            "rows[i].style.display=show?'':'none';}}</script>"
        )
    return f"{controls}{tbl}{js}"


def data_uri_download(filename: str, content: str, mime: str, label: str) -> str:
    """A download link that embeds *content* as a data URI (no server)."""
    import base64

    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    return (
        f'<a download="{esc(filename)}" href="data:{mime};base64,{b64}" '
        f'style="font-size:.8rem;margin-right:.6rem">{esc(label)}</a>'
    )


def json_download(filename: str, payload: Any, label: str = "⬇ JSON") -> str:
    return data_uri_download(
        filename, json.dumps(payload, indent=2, default=str), "application/json", label
    )
