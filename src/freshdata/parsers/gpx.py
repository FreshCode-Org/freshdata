"""GPX (GPS Exchange Format) parser.

Parses GPX 1.0/1.1 XML into three frames — ``waypoints``, ``route_points``, and
``track_points`` — each with ``lat``/``lon`` and any present ``ele``/``time``/``name``.
Namespaces are matched by local name so both GPX versions work. Malformed coordinates
are recorded as warnings rather than raising.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd

from .base import Parser, ParseResult

_POINT_KIND = {"wpt": "waypoints", "rtept": "route_points", "trkpt": "track_points"}
_CHILD_FIELDS = ("ele", "time", "name", "sym", "type")


def _local(tag: str) -> str:
    """Strip an XML namespace: ``{ns}trkpt`` -> ``trkpt``."""
    return tag.rsplit("}", 1)[-1]


class GPXParser(Parser):
    """Parse GPX waypoints, routes, and tracks into point frames."""

    format = "gpx"
    suggested_domain = "transport"

    def parse(self, source: Any) -> ParseResult:
        warnings: list[str] = []
        rows: dict[str, list[dict[str, Any]]] = {v: [] for v in _POINT_KIND.values()}

        stream = self.open_binary(source)
        try:
            root = ET.parse(stream).getroot()  # noqa: S314 - GPX files are local, trusted
        except ET.ParseError as exc:
            return ParseResult(self.format, {v: pd.DataFrame() for v in _POINT_KIND.values()},
                               self.suggested_domain, {}, [f"invalid GPX XML: {exc}"])
        finally:
            if hasattr(stream, "close"):
                stream.close()

        def _point(elem: ET.Element, kind: str, **extra: Any) -> None:
            row: dict[str, Any] = {}
            try:
                row["lat"] = float(elem.get("lat"))  # type: ignore[arg-type]
                row["lon"] = float(elem.get("lon"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                warnings.append(f"{_local(elem.tag)} with missing/invalid lat/lon skipped")
                return
            for child in elem:
                name = _local(child.tag)
                if name in _CHILD_FIELDS and child.text:
                    row[name] = child.text.strip()
            row.update(extra)
            rows[kind].append(row)

        n_trk = n_rte = 0
        for elem in root.iter():
            tag = _local(elem.tag)
            if tag == "wpt":
                _point(elem, "waypoints")
            elif tag == "rte":
                for pt in elem.iter():
                    if _local(pt.tag) == "rtept":
                        _point(pt, "route_points", route_index=n_rte)
                n_rte += 1
            elif tag == "trk":
                for pt in elem.iter():
                    if _local(pt.tag) == "trkpt":
                        _point(pt, "track_points", track_index=n_trk)
                n_trk += 1
        frames = {kind: pd.DataFrame(data) for kind, data in rows.items()}
        if all(df.empty for df in frames.values()):
            warnings.append("no waypoints, routes, or tracks found")

        return ParseResult(
            format=self.format,
            frames=frames,
            suggested_domain=self.suggested_domain,
            metadata={"tracks": n_trk, "routes": n_rte,
                      "waypoints": len(frames["waypoints"])},
            warnings=warnings,
        )
