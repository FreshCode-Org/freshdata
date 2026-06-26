"""UN/EDIFACT interchange parser.

Splits an EDIFACT interchange into segments and elements, honoring the optional ``UNA``
service-string advice (custom delimiters) and the release/escape character. Output is a
tidy ``segments`` frame — one row per element — plus interchange/message metadata pulled
from ``UNB``/``UNH``. This is a structural tokenizer, not a message-type (ORDERS, INVOIC,
…) schema validator.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Parser, ParseResult

_DEFAULTS = {"component": ":", "element": "+", "decimal": ".", "release": "?", "segment": "'"}


def _split(text: str, sep: str, release: str) -> list[str]:
    """Split *text* on *sep*, honoring the release char — but **keep** the release
    char in the output tokens so escapes still apply at the next (finer) level."""
    out: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
        elif ch == release:
            buf.append(ch)  # preserve the escape for lower-level splits
            escaped = True
        elif ch == sep:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out


def _unescape(text: str, release: str) -> str:
    """Remove release characters, yielding the literal value (``ACME?+CO`` -> ``ACME+CO``)."""
    out: list[str] = []
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
        elif ch == release:
            escaped = True
        else:
            out.append(ch)
    return "".join(out)


class EDIFACTParser(Parser):
    """Tokenize a UN/EDIFACT interchange into a tidy segment/element frame."""

    format = "edifact"
    suggested_domain = None

    def parse(self, source: Any) -> ParseResult:
        text = self.read_text(source).strip()
        warnings: list[str] = []
        d = dict(_DEFAULTS)

        if text.startswith("UNA") and len(text) >= 9:
            una = text[3:9]
            d.update(component=una[0], element=una[1], decimal=una[2],
                     release=una[3], segment=una[5])
            text = text[9:]

        release = d["release"]
        segments = [s for s in _split(text, d["segment"], release) if s.strip("\r\n \t")]
        rows: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {}

        for i, seg in enumerate(segments):
            elements = _split(seg.strip("\r\n \t"), d["element"], release)
            tag = _unescape(elements[0], release) if elements else ""
            for j, el in enumerate(elements[1:], start=1):
                comps = _split(el, d["component"], release)
                rows.append({"seg_index": i, "tag": tag, "element": j,
                             "value": _unescape(el, release), "component_count": len(comps)})
            if tag == "UNB" and len(elements) > 3:
                metadata["sender"] = _unescape(elements[2], release)
                metadata["recipient"] = _unescape(elements[3], release)
            elif tag == "UNH" and len(elements) > 2:
                metadata["message_type"] = _unescape(
                    _split(elements[2], d["component"], release)[0], release)

        tags = {r["tag"] for r in rows}
        if "UNB" not in tags and "UNH" not in tags:
            warnings.append("no UNB/UNH service segments found; input may not be EDIFACT")

        return ParseResult(
            format=self.format,
            frames={"segments": pd.DataFrame(rows)},
            suggested_domain=self.suggested_domain,
            metadata={"segments": len(segments), **metadata},
            warnings=warnings,
        )
