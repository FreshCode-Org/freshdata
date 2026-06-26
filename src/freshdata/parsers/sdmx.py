"""SDMX-ML (Statistical Data and Metadata eXchange) parser.

Parses the two common SDMX data layouts into a single ``observations`` frame:

- **Generic** (``<Series>`` with a ``<SeriesKey>`` of dimension ``<Value id=.. value=..>``
  and ``<Obs>`` carrying ``<ObsDimension>`` / ``<ObsValue>``).
- **Structure-specific** (each ``<Obs>`` is an element whose attributes are the
  dimensions and the measure).

Per the brief, SDMX handling is **audit-only**: an unrecognized layout produces a
warning and an empty frame rather than raising, so a bad feed never crashes a pipeline.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd

from .base import Parser, ParseResult


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class SDMXParser(Parser):
    """Parse SDMX-ML generic/structure-specific data into an observations frame."""

    format = "sdmx"
    suggested_domain = None

    def parse(self, source: Any) -> ParseResult:
        warnings: list[str] = []
        stream = self.open_binary(source)
        try:
            root = ET.parse(stream).getroot()  # noqa: S314 - local trusted SDMX files
        except ET.ParseError as exc:
            return ParseResult(self.format, {"observations": pd.DataFrame()},
                               None, {}, [f"invalid SDMX XML: {exc} (audit only)"])
        finally:
            if hasattr(stream, "close"):
                stream.close()

        rows: list[dict[str, Any]] = []
        series = [e for e in root.iter() if _local(e.tag) == "Series"]

        if series:
            for ser in series:
                key: dict[str, Any] = {}
                for sk in ser:
                    if _local(sk.tag) == "SeriesKey":
                        for v in sk:
                            vid = v.get("id")
                            if _local(v.tag) == "Value" and vid:
                                key[vid] = v.get("value")
                for obs in ser:
                    if _local(obs.tag) != "Obs":
                        continue
                    row = dict(key)
                    for child in obs:
                        ln = _local(child.tag)
                        if ln == "ObsDimension":
                            row["OBS_DIMENSION"] = child.get("value")
                        elif ln == "ObsValue":
                            row["OBS_VALUE"] = child.get("value")
                        elif ln == "Attributes":
                            for a in child:
                                aid = a.get("id")
                                if _local(a.tag) == "Value" and aid:
                                    row[aid] = a.get("value")
                    rows.append(row)
        else:
            # Structure-specific: each <Obs> carries its dimensions as attributes.
            obs_elems = [e for e in root.iter() if _local(e.tag) == "Obs"]
            rows = [dict(o.attrib) for o in obs_elems]
            if not obs_elems:
                warnings.append(
                    "no SDMX <Series>/<Obs> found; layout not recognized (audit only)"
                )

        return ParseResult(
            format=self.format,
            frames={"observations": pd.DataFrame(rows)},
            suggested_domain=self.suggested_domain,
            metadata={"series": len(series), "observations": len(rows)},
            warnings=warnings,
        )
