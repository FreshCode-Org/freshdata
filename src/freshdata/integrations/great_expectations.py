"""Export freshdata findings as a Great Expectations expectation suite (JSON).

Produces a plain GX *expectation suite* document — the same JSON shape GX writes under
``expectations/<suite>.json`` — without importing Great Expectations at all. The suite
can be dropped into a GX project, loaded via ``context.add_or_update_expectation_suite``,
or inspected on its own.

Findings map to the column-level expectations GX ships out of the box:

* ``not_null``         -> ``expect_column_values_to_not_be_null``
* ``unique``           -> ``expect_column_values_to_be_unique``
* ``accepted_values``  -> ``expect_column_values_to_be_in_set``
* ``regex``            -> ``expect_column_values_to_match_regex``
* ``between``          -> ``expect_column_values_to_be_between``

Findings without a native column expectation (relationships, table-level, opaque
advanced checks) are not expressed as expectations; their count is recorded in the
suite ``meta`` so nothing is silently dropped.

Great Expectations is **never** a hard (or even optional) dependency of this exporter —
only stdlib ``json`` is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import __version__
from ..findings import QualityFinding, classify_finding

__all__ = ["export_gx_suite"]


def _resolve_findings(report_or_findings: Any) -> list[QualityFinding]:
    if hasattr(report_or_findings, "to_findings"):
        return list(report_or_findings.to_findings())
    return list(report_or_findings)


def _expectation(finding: QualityFinding) -> dict[str, Any] | None:
    """Map one finding to a GX expectation dict, or ``None`` if it has no equivalent."""
    if finding.column is None:
        return None
    kind = classify_finding(finding)
    kwargs: dict[str, Any] = {"column": finding.column}
    if kind == "not_null":
        etype = "expect_column_values_to_not_be_null"
    elif kind == "unique":
        etype = "expect_column_values_to_be_unique"
    elif kind == "accepted_values":
        etype = "expect_column_values_to_be_in_set"
        kwargs["value_set"] = list(finding.extra.get("value_set", []))
    elif kind == "regex":
        etype = "expect_column_values_to_match_regex"
        kwargs["regex"] = finding.extra.get("regex") or finding.extra.get("pattern") or ""
    elif kind == "between":
        etype = "expect_column_values_to_be_between"
        kwargs["min_value"] = finding.extra.get("min_value")
        kwargs["max_value"] = finding.extra.get("max_value")
    else:
        return None
    return {
        "expectation_type": etype,
        "kwargs": kwargs,
        "meta": {"freshdata": {
            "finding_id": finding.finding_id,
            "severity": finding.severity,
            "rule_name": finding.rule_name,
            "message": finding.message,
        }},
    }


def export_gx_suite(report_or_findings: Any, suite_name: str, path: str) -> dict[str, Any]:
    """Export findings as a GX expectation suite; write it to *path* and return it.

    Parameters
    ----------
    report_or_findings:
        A freshdata report (anything with ``to_findings()``) or a list of
        :class:`~freshdata.QualityFinding`.
    suite_name:
        Name recorded as ``expectation_suite_name``.
    path:
        Destination ``.json`` file (parent directories are created as needed).

    Returns
    -------
    dict
        The suite document that was written.
    """
    findings = _resolve_findings(report_or_findings)

    expectations: list[dict[str, Any]] = []
    seen: set[str] = set()
    n_skipped = 0
    for finding in findings:
        exp = _expectation(finding)
        if exp is None:
            n_skipped += 1
            continue
        dedup_key = json.dumps([exp["expectation_type"], exp["kwargs"]], sort_keys=True,
                               default=str)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        expectations.append(exp)

    suite = {
        "expectation_suite_name": suite_name,
        "ge_cloud_id": None,
        "data_asset_type": None,
        "expectations": expectations,
        "meta": {
            "great_expectations_version": None,
            "freshdata": {
                "generated_by": "freshdata",
                "version": __version__,
                "n_findings": len(findings),
                "n_expectations": len(expectations),
                "n_skipped": n_skipped,
            },
        },
    }

    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(suite, indent=2, default=str), encoding="utf-8")
    return suite
