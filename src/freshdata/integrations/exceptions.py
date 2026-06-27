"""Build and persist *exception tables* (a.k.a. quarantine / dead-letter tables).

An exception table is the row-level companion to a dbt/GX suite: one row per data
issue, suitable for routing to a quarantine table, a review queue, or a BI dashboard.
:func:`build_exception_table` turns findings into a tidy :class:`pandas.DataFrame`;
:func:`write_exception_table` persists it as CSV (stdlib/pandas), Parquet (lazy
``pyarrow``), or DuckDB (lazy ``duckdb``).

``observed_value`` is **redacted by default** — the table is the one place raw cell
values would otherwise land — and revealed only when ``include_pii=True``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..findings import REDACT_TOKEN, QualityFinding

if TYPE_CHECKING:  # annotations only
    pass

__all__ = ["EXCEPTION_COLUMNS", "build_exception_table", "write_exception_table"]

#: The exact column order of an exception table.
EXCEPTION_COLUMNS = (
    "exception_id",
    "source_row_id",
    "column",
    "severity",
    "rule_name",
    "observed_value",
    "message",
    "action_taken",
    "created_at",
    "lineage_run_id",
)

_PARQUET_HINT = (
    "Writing Parquet exception tables requires pyarrow. Install it with: "
    'pip install "freshdata[pyarrow]"'
)
_DUCKDB_HINT = (
    "Writing DuckDB exception tables requires duckdb. Install it with: "
    'pip install "freshdata[duckdb]"'
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _exception_id(finding_id: str, source_row_id: Any) -> str:
    return hashlib.sha1(f"{finding_id}|{source_row_id}".encode()).hexdigest()[:12]


def _resolve_findings(report_or_findings: Any) -> list[QualityFinding]:
    if hasattr(report_or_findings, "to_findings"):
        return list(report_or_findings.to_findings())
    return list(report_or_findings)


def build_exception_table(
    df: pd.DataFrame | None,
    findings: Any,
    *,
    include_pii: bool = False,
) -> pd.DataFrame:
    """Build a row-level exception table from *findings*.

    Parameters
    ----------
    df:
        The source frame, used to recover an ``observed_value`` when the finding did
        not carry one. May be ``None`` (e.g. when only a ``report.json`` is available),
        in which case the table is built from the findings alone.
    findings:
        A report (anything with ``to_findings()``) or a list of
        :class:`~freshdata.QualityFinding`.
    include_pii:
        When ``False`` (default) every ``observed_value`` is replaced with the redaction
        token. When ``True``, observed values are written in the clear.

    Returns
    -------
    pandas.DataFrame
        One row per finding, with :data:`EXCEPTION_COLUMNS` in order.
    """
    findings = _resolve_findings(findings)
    created_at = _utc_now()
    rows: list[dict[str, Any]] = []

    for finding in findings:
        source_row_id = (finding.row_index if finding.row_index is not None
                         else finding.row_selector)

        raw = finding.observed_value
        if (raw is None and df is not None and finding.column in getattr(df, "columns", [])
                and finding.row_index is not None):
            try:
                raw = df.loc[finding.row_index, finding.column]
            except (KeyError, IndexError, TypeError):
                raw = None

        if not include_pii:
            observed = REDACT_TOKEN
        elif raw is None:
            observed = None
        else:
            observed = str(raw)

        rows.append({
            "exception_id": _exception_id(finding.finding_id, source_row_id),
            "source_row_id": source_row_id,
            "column": finding.column,
            "severity": finding.severity,
            "rule_name": finding.rule_name,
            "observed_value": observed,
            "message": finding.message,
            "action_taken": finding.action_taken,
            "created_at": created_at,
            "lineage_run_id": finding.lineage_run_id,
        })

    return pd.DataFrame(rows, columns=list(EXCEPTION_COLUMNS))


def _infer_format(path: str) -> str:
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        return "parquet"
    if low.endswith((".duckdb", ".ddb")):
        return "duckdb"
    return "csv"


def write_exception_table(
    table: pd.DataFrame,
    path: str,
    *,
    format: str | None = "csv",  # noqa: A002 - matches the documented public API
    table_name: str = "exceptions",
) -> str:
    """Write an exception table to *path* as CSV, Parquet, or DuckDB.

    ``format`` may be ``"csv"``, ``"parquet"``, ``"duckdb"``, or ``None`` to infer it
    from the file extension. Parquet and DuckDB lazily import their (optional) backends
    and raise a helpful :class:`ImportError` if the backend is missing.

    Returns the *path* written.
    """
    fmt = (format or _infer_format(path)).lower()
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        table.to_csv(path, index=False)
    elif fmt == "parquet":
        try:
            import pyarrow  # noqa: F401, PLC0415 — optional dependency, lazily imported
        except ImportError as exc:  # pragma: no cover - exercised via skip
            raise ImportError(_PARQUET_HINT) from exc
        table.to_parquet(path, index=False)
    elif fmt == "duckdb":
        try:
            import duckdb  # noqa: PLC0415 — optional dependency, lazily imported
        except ImportError as exc:  # pragma: no cover - exercised via skip
            raise ImportError(_DUCKDB_HINT) from exc
        con = duckdb.connect(path)
        try:
            con.register("_freshdata_exceptions", table)
            con.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS '
                "SELECT * FROM _freshdata_exceptions"
            )
        finally:
            con.close()
    else:
        raise ValueError(f"unknown exception-table format: {format!r}")

    return path
