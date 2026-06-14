"""Lightweight transformation lineage with OpenLineage-compatible export.

Every pipeline step is recorded as a :class:`LineageEvent` capturing *who* ran
it, *when*, the *input* and *output* schemas, and the exact *rule applied*.
:class:`LineageTracker` accumulates the events for a run and serialises them
into OpenLineage ``RunEvent`` JSON (a START + a COMPLETE event with schema and
column-lineage facets) — no dependency on the ``openlineage-python`` client, so
it drops into any enterprise catalog that speaks the open spec.
"""

from __future__ import annotations

import getpass
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..adapters.polars import is_polars_frame
from .config import LineageConfig

_OL_PRODUCER = "https://github.com/FreshCode-Org/freshdata"
_OL_SCHEMA_URL = "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent"
_SCHEMA_FACET_URL = (
    "https://openlineage.io/spec/facets/1-1-1/"
    "SchemaDatasetFacet.json#/$defs/SchemaDatasetFacet"
)
_COLUMN_LINEAGE_FACET_URL = (
    "https://openlineage.io/spec/facets/1-2-0/"
    "ColumnLineageDatasetFacet.json#/$defs/ColumnLineageDatasetFacet"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_actor() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - getuser raises various OSErrors headless
        return "unknown"


def _schema_pairs(obj: object) -> tuple[tuple[str, str], ...]:
    """Extract ``((name, type), ...)`` from a pandas/polars frame or a schema.

    Accepts a precomputed schema (sequence of ``{"name", "type"}`` dicts or
    ``(name, type)`` pairs) so callers can record lineage without holding the
    frame.
    """
    if is_polars_frame(obj):
        return tuple((str(n), str(t)) for n, t in obj.schema.items())  # type: ignore[attr-defined]
    columns = getattr(obj, "columns", None)
    dtypes = getattr(obj, "dtypes", None)
    if columns is not None and dtypes is not None:
        return tuple((str(n), str(t)) for n, t in zip(columns, dtypes))
    if isinstance(obj, (list, tuple)):
        pairs = []
        for item in obj:
            if isinstance(item, dict):
                pairs.append((str(item["name"]), str(item.get("type", ""))))
            else:
                pairs.append((str(item[0]), str(item[1])))
        return tuple(pairs)
    raise TypeError(f"cannot derive a schema from {type(obj).__name__}")


def schema_of(obj: object) -> list[dict[str, str]]:
    """Return a list of ``{"name", "type"}`` field dicts for a frame/schema."""
    return [{"name": n, "type": t} for n, t in _schema_pairs(obj)]


@dataclass(frozen=True)
class LineageEvent:
    """One recorded transformation: who/when/input/output/rule."""

    rule_applied: str
    who: str
    when: str
    input_schema: tuple[tuple[str, str], ...]
    output_schema: tuple[tuple[str, str], ...]
    count: int = 0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_applied": self.rule_applied,
            "who": self.who,
            "when": self.when,
            "count": self.count,
            "description": self.description,
            "input_schema": [{"name": n, "type": t} for n, t in self.input_schema],
            "output_schema": [{"name": n, "type": t} for n, t in self.output_schema],
        }


class LineageTracker:
    """Accumulates :class:`LineageEvent` records for one pipeline run."""

    def __init__(
        self,
        config: LineageConfig | None = None,
        *,
        input_name: str = "input",
        output_name: str = "output",
    ) -> None:
        self.config = config or LineageConfig()
        self.run_id = str(uuid.uuid4())
        self.input_name = input_name
        self.output_name = output_name
        self.events: list[LineageEvent] = []
        self.started_at = _now_iso()

    def record(
        self,
        rule_applied: str,
        input_obj: object,
        output_obj: object,
        *,
        who: str | None = None,
        count: int = 0,
        description: str = "",
    ) -> LineageEvent:
        """Record one transformation and return the resulting event."""
        actor = who or self.config.actor or _default_actor()
        event = LineageEvent(
            rule_applied=rule_applied,
            who=actor,
            when=_now_iso(),
            input_schema=_schema_pairs(input_obj),
            output_schema=_schema_pairs(output_obj),
            count=int(count),
            description=description,
        )
        self.events.append(event)
        return event

    # -- export -------------------------------------------------------------

    @property
    def input_schema(self) -> tuple[tuple[str, str], ...]:
        return self.events[0].input_schema if self.events else ()

    @property
    def output_schema(self) -> tuple[tuple[str, str], ...]:
        return self.events[-1].output_schema if self.events else ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "namespace": self.config.namespace,
            "job": self.config.job_name,
            "input": self.input_name,
            "output": self.output_name,
            "started_at": self.started_at,
            "events": [e.to_dict() for e in self.events],
        }

    def _schema_facet(self, pairs: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        return {
            "_producer": self.config.producer,
            "_schemaURL": _SCHEMA_FACET_URL,
            "fields": [{"name": n, "type": t} for n, t in pairs],
        }

    def _column_lineage_facet(self) -> dict[str, Any]:
        """Identity column lineage: each surviving output column derives from
        the like-named input column (the common case for cleaning)."""
        inputs = {n for n, _ in self.input_schema}
        fields = {
            name: {
                "inputFields": [
                    {
                        "namespace": self.config.dataset_namespace,
                        "name": self.input_name,
                        "field": name,
                    }
                ],
                "transformationType": "DIRECT",
                "transformationDescription": "cleaned / normalized in place",
            }
            for name, _ in self.output_schema
            if name in inputs
        }
        return {
            "_producer": self.config.producer,
            "_schemaURL": _COLUMN_LINEAGE_FACET_URL,
            "fields": fields,
        }

    def _event(self, event_type: str, *, with_facets: bool) -> dict[str, Any]:
        run_facets: dict[str, Any] = {
            "freshdata_transformations": {
                "_producer": self.config.producer,
                "transformations": [
                    {
                        "rule": e.rule_applied,
                        "who": e.who,
                        "when": e.when,
                        "count": e.count,
                        "description": e.description,
                    }
                    for e in self.events
                ],
            }
        }
        output_facets: dict[str, Any] = {}
        input_facets: dict[str, Any] = {}
        if with_facets:
            input_facets["schema"] = self._schema_facet(self.input_schema)
            output_facets["schema"] = self._schema_facet(self.output_schema)
            output_facets["columnLineage"] = self._column_lineage_facet()
        return {
            "eventType": event_type,
            "eventTime": _now_iso(),
            "producer": self.config.producer,
            "schemaURL": _OL_SCHEMA_URL,
            "run": {"runId": self.run_id, "facets": run_facets},
            "job": {"namespace": self.config.namespace, "name": self.config.job_name},
            "inputs": [
                {
                    "namespace": self.config.dataset_namespace,
                    "name": self.input_name,
                    "facets": input_facets,
                }
            ],
            "outputs": [
                {
                    "namespace": self.config.dataset_namespace,
                    "name": self.output_name,
                    "facets": output_facets,
                }
            ],
        }

    def to_openlineage(self) -> list[dict[str, Any]]:
        """A START + COMPLETE pair of OpenLineage ``RunEvent`` objects."""
        return [
            self._event("START", with_facets=False),
            self._event("COMPLETE", with_facets=True),
        ]

    def to_json(self, *, indent: int | None = 2) -> str:
        """OpenLineage events as a JSON string."""
        return json.dumps(self.to_openlineage(), indent=indent, default=str)

    def emit(self, path: str | None = None, *, indent: int | None = 2) -> str:
        """Serialise OpenLineage events; write to *path* if given. Returns JSON."""
        payload = self.to_json(indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(payload)
        return payload

    def __repr__(self) -> str:
        return f"<LineageTracker run={self.run_id[:8]} events={len(self.events)}>"
