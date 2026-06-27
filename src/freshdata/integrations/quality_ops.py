"""Bundle the quality-ops exporters behind one call and stamp the lineage run.

:func:`export_quality_ops` runs whichever exporters a caller asks for (dbt tests, a
Great Expectations suite, an exception table), threads a single OpenLineage ``run_id``
through every finding, and attaches the produced artifact paths back onto the lineage
run as facets (``dbt_tests_path``, ``gx_suite_path``, ``exception_table_path``). The
result is one :class:`QualityOpsResult` tying the findings, the artifacts, and the
lineage event together.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from ..enterprise.lineage import LineageTracker
from ..findings import QualityFinding
from .dbt import export_dbt_tests
from .exceptions import build_exception_table, write_exception_table
from .great_expectations import export_gx_suite

__all__ = ["QualityOpsResult", "export_quality_ops"]


@dataclass
class QualityOpsResult:
    """Everything one :func:`export_quality_ops` call produced.

    Bundles the normalized findings, the paths of any artifacts written, the in-memory
    exception table, and the OpenLineage event (whose run facets carry the artifact
    paths). ``tracker`` is the :class:`~freshdata.enterprise.LineageTracker` used.
    """

    findings: list[QualityFinding]
    dbt_path: str | None = None
    gx_path: str | None = None
    exception_table_path: str | None = None
    exception_table: pd.DataFrame | None = None
    lineage_event: list[dict[str, Any]] = field(default_factory=list)
    tracker: LineageTracker | None = None

    def to_dict(self, *, include_pii: bool = False) -> dict[str, Any]:
        """JSON-friendly summary; ``observed_value`` redacted unless ``include_pii``."""
        return {
            "findings": [f.to_dict(include_pii=include_pii) for f in self.findings],
            "dbt_tests_path": self.dbt_path,
            "gx_suite_path": self.gx_path,
            "exception_table_path": self.exception_table_path,
            "lineage_event": self.lineage_event,
        }


def _facet(producer: str, path: str) -> dict[str, Any]:
    """An OpenLineage run facet recording one exported artifact path."""
    return {"_producer": producer, "path": str(Path(path).resolve())}


def export_quality_ops(
    report_or_findings: Any,
    *,
    model_name: str = "freshdata_model",
    suite_name: str = "freshdata_suite",
    dbt_path: str | None = None,
    gx_path: str | None = None,
    exception_table_path: str | None = None,
    df: pd.DataFrame | None = None,
    lineage: LineageTracker | None = None,
    include_pii: bool = False,
    severity_map: dict[str, str] | None = None,
    exceptions_format: str | None = None,
) -> QualityOpsResult:
    """Export findings to the requested quality-ops formats under one lineage run.

    Parameters
    ----------
    report_or_findings:
        A freshdata report (anything with ``to_findings()``) or a list of
        :class:`~freshdata.QualityFinding`.
    model_name, suite_name:
        Names for the dbt model and the GX suite, respectively.
    dbt_path, gx_path, exception_table_path:
        Output paths; an exporter runs only when its path is given.
    df:
        Optional source frame, used to enrich exception-table ``observed_value``.
    lineage:
        An existing :class:`~freshdata.enterprise.LineageTracker` to attach facets to.
        When omitted, a fresh one is created (so there is always a ``run_id``).
    include_pii:
        Reveal observed values in the exception table (default redacts them).
    severity_map:
        Optional override forwarded to the dbt exporter.
    exceptions_format:
        ``csv`` / ``parquet`` / ``duckdb``; inferred from the path extension when omitted.

    Returns
    -------
    QualityOpsResult
    """
    tracker = lineage if lineage is not None else LineageTracker()
    run_id = tracker.run_id
    producer = tracker.config.producer

    if hasattr(report_or_findings, "to_findings"):
        findings = list(report_or_findings.to_findings(lineage_run_id=run_id))
    else:
        findings = [
            f if f.lineage_run_id else replace(f, lineage_run_id=run_id)
            for f in report_or_findings
        ]

    result = QualityOpsResult(findings=findings, tracker=tracker)

    if dbt_path is not None:
        export_dbt_tests(findings, model_name, dbt_path, severity_map=severity_map)
        tracker.add_run_facet("dbt_tests_path", _facet(producer, dbt_path))
        result.dbt_path = dbt_path

    if gx_path is not None:
        export_gx_suite(findings, suite_name, gx_path)
        tracker.add_run_facet("gx_suite_path", _facet(producer, gx_path))
        result.gx_path = gx_path

    if exception_table_path is not None:
        table = build_exception_table(df, findings, include_pii=include_pii)
        write_exception_table(table, exception_table_path, format=exceptions_format)
        tracker.add_run_facet("exception_table_path", _facet(producer, exception_table_path))
        result.exception_table = table
        result.exception_table_path = exception_table_path

    result.lineage_event = tracker.to_openlineage()
    return result
