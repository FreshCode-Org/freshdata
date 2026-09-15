"""dbt integration for freshdata's trust gate.

Two entry points:

* :class:`FreshDataDbtTransform` — read one model's materialized table from the
  warehouse (via a SQLAlchemy connection string, defaulting to
  ``$FRESHDATA_WAREHOUSE_CONN``), clean + gate it, write a ``<model>_audit.json``,
  and optionally fail on a low score.
* :func:`gate_manifest` — parse a dbt ``target/manifest.json`` and gate every model,
  returning a summary dict (also surfaced by the ``dbt-gate`` CLI and the bundled
  ``freshdata_trust_gate`` macro).

Only SQLAlchemy (the ``dbt`` extra) is needed to read a warehouse; manifest parsing
itself is pure stdlib. Install with ``pip install "freshdata-cleaner[dbt]"``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

from .._core import (
    OnLowScore,
    TrustGateError,
    TrustGateResult,
    evaluate_trust_gate,
    validate_on_low_score,
)
from .tests_exporter import export_dbt_tests

if TYPE_CHECKING:  # annotations only
    import pandas as pd

    from freshdata import CleanConfig

__all__ = ["FreshDataDbtTransform", "export_dbt_tests", "gate_manifest"]

logger = logging.getLogger("freshdata.integrations.dbt")

_SQLALCHEMY_HINT = (
    "Reading a warehouse table requires SQLAlchemy. Install it with: "
    'pip install "freshdata-cleaner[dbt]"'
)


def _validate_audit_table_name(table: str) -> str:
    path = PurePath(table)
    if (
        not table
        or table in {".", ".."}
        or path.is_absolute()
        or path.name != table
        or "/" in table
        or "\\" in table
    ):
        raise ValueError(f"{table!r} is not a safe dbt model name for audit output")
    return table


def _read_table(conn_str: str, schema: str | None, table: str) -> pd.DataFrame:
    """Read ``schema.table`` from ``conn_str`` into a DataFrame."""
    import pandas as pd

    try:
        import sqlalchemy as sa
    except ImportError as exc:  # pragma: no cover - exercised via mocking
        raise ImportError(_SQLALCHEMY_HINT) from exc

    engine = sa.create_engine(conn_str)
    try:
        # Pass a Connection (not the Engine) so this works across pandas versions:
        # older pandas calls ``.execute()`` on the connectable, which SQLAlchemy 2.0
        # removed from Engine but kept on Connection.
        with engine.connect() as connection:
            return pd.read_sql_table(table, connection, schema=schema)
    finally:
        engine.dispose()


@dataclass
class FreshDataDbtTransform:
    """Clean + trust-gate a single dbt model's materialized table.

    ``model_name`` may be ``"schema.table"`` or a bare table name (with ``schema``
    supplied separately). The warehouse is reached via ``conn_str`` or, when omitted,
    the ``FRESHDATA_WAREHOUSE_CONN`` environment variable.
    """

    model_name: str
    conn_str: str | None = None
    schema: str | None = None
    trust_score_threshold: float = 80.0
    on_low_score: OnLowScore = "warn"
    output_dir: str | None = None
    clean_config: CleanConfig | None = None
    system_actor: str = "freshdata"
    fail_on_low_score: bool = False
    #: File stem for the audit (``<audit_name>_audit.json``); defaults to the table
    #: name. :func:`gate_manifest` sets it when two gated models share an alias.
    audit_name: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid gate policies and unsafe audit names at configuration."""
        self.on_low_score = validate_on_low_score(self.on_low_score)
        if self.audit_name is not None:
            self.audit_name = _validate_audit_table_name(self.audit_name)

    def _split_table(self) -> tuple[str | None, str]:
        if self.schema:
            return self.schema, self.model_name
        if "." in self.model_name:
            *prefix, table = self.model_name.split(".")
            return ".".join(prefix) or None, table
        return None, self.model_name

    def _audit_stem(self, table: str) -> str:
        return self.audit_name if self.audit_name is not None else table

    def _write_audit(self, table: str, result: TrustGateResult) -> Path:
        stem = _validate_audit_table_name(self._audit_stem(table))
        out_dir = Path(self.output_dir)  # type: ignore[arg-type]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stem}_audit.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        return path

    def run(self, *, raise_on_fail: bool = True) -> TrustGateResult:
        """Read the model table, gate it, optionally write an audit, return the result.

        A failing gate raises :class:`TrustGateError` (after the audit is written) when
        ``on_low_score="fail"`` or ``fail_on_low_score=True``. Pass
        ``raise_on_fail=False`` to get the failing result back instead.
        """
        conn = self.conn_str or os.environ.get("FRESHDATA_WAREHOUSE_CONN")
        if not conn:
            raise ValueError(
                "No warehouse connection: pass conn_str or set FRESHDATA_WAREHOUSE_CONN."
            )
        schema, table = self._split_table()
        if self.output_dir:
            _validate_audit_table_name(self._audit_stem(table))
        df = _read_table(conn, schema, table)
        _, result = evaluate_trust_gate(
            df,
            clean_config=self.clean_config,
            trust_score_threshold=self.trust_score_threshold,
            on_low_score=self.on_low_score,
            publish_full_report=True,
            system_actor=self.system_actor,
        )
        if self.output_dir:
            self._write_audit(table, result)
        if raise_on_fail and (
            result.should_fail or (self.fail_on_low_score and not result.passed)
        ):
            raise TrustGateError(result.message)
        return result


def _audit_names(models: list[tuple[str, dict[str, Any]]]) -> list[str | None]:
    """Return an audit file stem per model, or ``None`` to keep the table name.

    Audit files are named after the model's alias, which dbt only requires to be
    unique within a schema. When gated models share an alias (compared
    case-insensitively, as audit files may land on a case-insensitive filesystem),
    each of them is named ``"<schema>.<alias>"`` instead, or after its manifest
    ``unique_id`` when it has no schema or that name is still not unique.
    """
    tables = [node.get("alias") or node.get("name") for _, node in models]
    alias_counts = Counter(t.casefold() for t in tables if isinstance(t, str))
    names: list[str | None] = []
    for (node_id, node), table in zip(models, tables):
        if not isinstance(table, str) or alias_counts[table.casefold()] < 2:
            names.append(None)
            continue
        schema = node.get("schema")
        names.append(f"{schema}.{table}" if isinstance(schema, str) and schema else node_id)
    stems = [name if name is not None else table for name, table in zip(names, tables)]
    stem_counts = Counter(s.casefold() for s in stems if isinstance(s, str))
    return [
        node_id if isinstance(stem, str) and stem_counts[stem.casefold()] > 1 else name
        for (node_id, _), name, stem in zip(models, names, stems)
    ]


def gate_manifest(
    manifest_path: str | Path,
    *,
    conn_str: str | None = None,
    trust_score_threshold: float = 80.0,
    on_low_score: OnLowScore = "warn",
    output_dir: str | None = None,
    clean_config: CleanConfig | None = None,
    system_actor: str = "freshdata",
) -> dict[str, Any]:
    """Gate every model in a dbt ``manifest.json`` and return a summary dict.

    The summary has shape ``{"models": [...], "skipped": [...],
    "models_processed": int, "failed_models": int, "all_passed": bool}``. A model
    that raises (e.g. its table is missing) is recorded with an ``"error"`` and
    counted as failed, so one bad model never aborts the whole run.

    Ephemeral models (never materialized by dbt) and disabled models are not read;
    they are listed under ``"skipped"`` and not counted in ``models_processed``.
    ``all_passed`` is ``False`` when no model was gated, so a manifest with nothing
    to gate cannot pass as a clean run.

    Raises:
        ValueError: the file is not valid JSON or has no ``nodes`` mapping (i.e. it
            is not a dbt manifest, e.g. ``run_results.json``).
    """
    on_low_score = validate_on_low_score(on_low_score)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    nodes = manifest.get("nodes") if isinstance(manifest, dict) else None
    if not isinstance(nodes, dict):
        raise ValueError(f"{manifest_path} is not a dbt manifest: no 'nodes' mapping")

    models: list[tuple[str, Any]] = []  # (unique_id, raw manifest node)
    skipped: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        if not isinstance(node, dict) or node.get("resource_type") != "model":
            continue
        config = node.get("config")
        config = config if isinstance(config, dict) else {}
        if config.get("materialized") == "ephemeral":
            skipped.append({"model": node.get("name"), "reason": "ephemeral"})
        elif config.get("enabled") is False:
            skipped.append({"model": node.get("name"), "reason": "disabled"})
        else:
            models.append((node_id, node))

    audit_names = _audit_names(models) if output_dir else [None] * len(models)
    summaries: list[dict[str, Any]] = []
    failed = 0
    for (_, node), audit_name in zip(models, audit_names):
        name = node.get("name")
        schema = node.get("schema")
        table = node.get("alias") or name
        try:
            result = FreshDataDbtTransform(
                model_name=table,
                schema=schema,
                conn_str=conn_str,
                trust_score_threshold=trust_score_threshold,
                on_low_score=on_low_score,
                output_dir=output_dir,
                clean_config=clean_config,
                system_actor=system_actor,
                audit_name=audit_name,
            ).run(raise_on_fail=False)
        except Exception as exc:  # noqa: BLE001 - one bad model must not abort the run
            logger.warning("freshdata: gating model %r failed: %s", name, exc)
            summaries.append({"model": name, "error": str(exc)})
            failed += 1
            continue
        summaries.append(
            {
                "model": name,
                "trust_score": result.trust_score,
                "grade": result.grade,
                "threshold": result.threshold,
                "passed": result.passed,
                "high_risk_count": result.high_risk_count,
            }
        )
        if not result.passed:
            failed += 1

    return {
        "models": summaries,
        "skipped": skipped,
        "models_processed": len(models),
        "failed_models": failed,
        "all_passed": failed == 0 and len(models) > 0,
    }
