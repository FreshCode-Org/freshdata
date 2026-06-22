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
itself is pure stdlib. Install with ``pip install "freshdata[dbt]"``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._core import OnLowScore, TrustGateError, TrustGateResult, evaluate_trust_gate

if TYPE_CHECKING:  # annotations only
    import pandas as pd

    from freshdata import CleanConfig

__all__ = ["FreshDataDbtTransform", "gate_manifest"]

logger = logging.getLogger("freshdata.integrations.dbt")

_SQLALCHEMY_HINT = (
    "Reading a warehouse table requires SQLAlchemy. Install it with: "
    'pip install "freshdata[dbt]"'
)


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

    def _split_table(self) -> tuple[str | None, str]:
        if self.schema:
            return self.schema, self.model_name
        if "." in self.model_name:
            *prefix, table = self.model_name.split(".")
            return ".".join(prefix) or None, table
        return None, self.model_name

    def _write_audit(self, table: str, result: TrustGateResult) -> Path:
        out_dir = Path(self.output_dir)  # type: ignore[arg-type]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{table}_audit.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        return path

    def run(self) -> TrustGateResult:
        """Read the model table, gate it, optionally write an audit, return the result."""
        conn = self.conn_str or os.environ.get("FRESHDATA_WAREHOUSE_CONN")
        if not conn:
            raise ValueError(
                "No warehouse connection: pass conn_str or set FRESHDATA_WAREHOUSE_CONN."
            )
        schema, table = self._split_table()
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
        if self.fail_on_low_score and not result.passed:
            raise TrustGateError(result.message)
        return result


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

    The summary has shape ``{"models": [...], "models_processed": int,
    "failed_models": int, "all_passed": bool}``. A model that raises (e.g. its table
    is missing) is recorded with an ``"error"`` and counted as failed, so one bad
    model never aborts the whole run.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    nodes = manifest.get("nodes", {})
    models = [n for n in nodes.values() if n.get("resource_type") == "model"]

    summaries: list[dict[str, Any]] = []
    failed = 0
    for node in models:
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
            ).run()
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
        "models_processed": len(models),
        "failed_models": failed,
        "all_passed": failed == 0,
    }
