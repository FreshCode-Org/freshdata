"""A normalized quality-finding model that bridges freshdata's reports to the
wider quality-ops stack (dbt tests, Great Expectations suites, exception tables).

Every report in freshdata records issues in its own vocabulary — a ``CleanReport``
has domain findings and repair actions, a ``DriftReport`` has :class:`DriftFinding`
objects, a ``PrivacyReport`` has masking events, an ``EntityResolutionReport`` has
scored match pairs. :class:`QualityFinding` is the single shape all of them project
into via ``report.to_findings()`` so the exporters downstream only ever speak one
language.

This module is deliberately light: stdlib + nothing else at import time, no
``pandas`` and no enterprise layer, so ``CleanReport.to_findings()`` stays as cheap
as the rest of :mod:`freshdata.report`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "CANONICAL_SEVERITIES",
    "REDACT_TOKEN",
    "QualityFinding",
    "classify_finding",
    "findings_from_dict",
    "make_finding_id",
    "normalize_severity",
]

#: Severities, in increasing order, that every source vocabulary is mapped onto.
CANONICAL_SEVERITIES = ("info", "warning", "error")

#: Placeholder substituted for an ``observed_value`` whenever it is not explicitly
#: requested in the clear. ASCII so it survives CSV/Parquet round-trips unchanged.
REDACT_TOKEN = "[redacted]"

#: How each source severity word collapses onto :data:`CANONICAL_SEVERITIES`.
_SEVERITY_ALIASES = {
    "info": "info",
    "low": "info",
    "passed": "info",
    "note": "info",
    "debug": "info",
    "warning": "warning",
    "warn": "warning",
    "warned": "warning",
    "medium": "warning",
    "possible_match": "warning",
    "error": "error",
    "failed": "error",
    "fail": "error",
    "high": "error",
    "critical": "error",
    "match": "error",
}


def normalize_severity(raw: Any) -> str:
    """Collapse any source severity onto ``info`` / ``warning`` / ``error``.

    Unknown values default to ``"warning"`` — visible, but not alarming.

    Examples
    --------
    >>> normalize_severity("high"), normalize_severity("warn"), normalize_severity("passed")
    ('error', 'warning', 'info')
    >>> normalize_severity("something-else")
    'warning'
    """
    return _SEVERITY_ALIASES.get(str(raw).strip().lower(), "warning")


def make_finding_id(step: str, column: Any, rule_name: str, row: Any, message: str) -> str:
    """Return a short, deterministic id for a finding.

    Stable across runs for the same logical issue, so findings can be diffed,
    deduplicated, and cross-referenced (e.g. an exception row back to a suite test).
    """
    raw = f"{step}|{column}|{rule_name}|{row}|{message}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _utc_now() -> str:
    """Current UTC time as an ISO-8601 string ending in ``Z``."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class QualityFinding:
    """One normalized data-quality observation, source-agnostic.

    Produced by ``report.to_findings()`` and consumed by the quality-ops exporters.
    ``observed_value`` is treated as potentially sensitive: it is **redacted by
    default** in :meth:`to_dict` and in exception tables, and revealed only when a
    caller explicitly opts in.

    Attributes
    ----------
    finding_id:
        Short deterministic hash (see :func:`make_finding_id`).
    severity:
        One of :data:`CANONICAL_SEVERITIES`.
    step:
        Where the finding came from — ``"clean"``, a domain layer, ``"drift"``,
        ``"privacy"``, ``"entity_resolution"``.
    column:
        Column the finding is about, or ``None`` for table-level findings.
    rule_name:
        Machine-readable rule/check identifier.
    message:
        Human-readable description.
    row_index:
        Single row identifier when the finding is about one row, else ``None``.
    row_selector:
        A summary (e.g. ``"rows: [3, 7]"``) when many rows are implicated.
    observed_value:
        The offending value (redacted by default on export).
    expected_condition:
        What should have held instead.
    action_taken:
        What freshdata did about it (a repair, a mask, a decision), if anything.
    trust_score_impact:
        Best-effort 0-1 contribution to a trust drop, when derivable, else ``None``.
    lineage_run_id:
        The OpenLineage run id this finding belongs to, if exported under one.
    sensitive:
        ``True`` when ``observed_value`` is known to carry PII.
    extra:
        Exporter hints (``value_set``, ``regex``, ``min_value``/``max_value``,
        ``to_model``/``to_field``) and any other JSON-friendly metadata.
    """

    finding_id: str
    severity: str
    step: str
    column: str | None
    rule_name: str
    message: str
    row_index: Any = None
    row_selector: str | None = None
    observed_value: Any = None
    expected_condition: str | None = None
    action_taken: str | None = None
    trust_score_impact: float | None = None
    lineage_run_id: str | None = None
    sensitive: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        severity: str,
        step: str,
        rule_name: str,
        message: str,
        column: str | None = None,
        row_index: Any = None,
        row_selector: str | None = None,
        observed_value: Any = None,
        expected_condition: str | None = None,
        action_taken: str | None = None,
        trust_score_impact: float | None = None,
        lineage_run_id: str | None = None,
        sensitive: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> QualityFinding:
        """Build a finding, deriving :attr:`finding_id` and normalizing severity."""
        row = row_index if row_index is not None else row_selector
        return cls(
            finding_id=make_finding_id(step, column, rule_name, row, message),
            severity=normalize_severity(severity),
            step=step,
            column=column,
            rule_name=rule_name,
            message=message,
            row_index=row_index,
            row_selector=row_selector,
            observed_value=observed_value,
            expected_condition=expected_condition,
            action_taken=action_taken,
            trust_score_impact=trust_score_impact,
            lineage_run_id=lineage_run_id,
            sensitive=sensitive,
            extra=dict(extra or {}),
        )

    def display_observed(self, *, include_pii: bool = False) -> Any:
        """Return ``observed_value`` for output, redacted unless ``include_pii``."""
        if include_pii:
            return self.observed_value
        return REDACT_TOKEN

    def to_dict(self, *, include_pii: bool = False) -> dict[str, Any]:
        """JSON-friendly dict; ``observed_value`` redacted unless ``include_pii``."""
        return {
            "finding_id": self.finding_id,
            "severity": self.severity,
            "step": self.step,
            "column": self.column,
            "rule_name": self.rule_name,
            "message": self.message,
            "row_index": self.row_index,
            "row_selector": self.row_selector,
            "observed_value": self.display_observed(include_pii=include_pii),
            "expected_condition": self.expected_condition,
            "action_taken": self.action_taken,
            "trust_score_impact": self.trust_score_impact,
            "lineage_run_id": self.lineage_run_id,
            "extra": dict(self.extra),
        }


def classify_finding(finding: QualityFinding) -> str:
    """Map a finding onto a check kind the exporters understand.

    Returns one of ``not_null``, ``unique``, ``accepted_values``, ``relationships``,
    ``regex``, ``between``, or ``custom``. Explicit hints in ``finding.extra`` win;
    otherwise the rule name / expected condition is matched on keywords.
    """
    hints = finding.extra or {}
    if "value_set" in hints:
        return "accepted_values"
    if "regex" in hints or "pattern" in hints:
        return "regex"
    if "min_value" in hints or "max_value" in hints:
        return "between"
    if hints.get("relationship") or ("to_model" in hints and "to_field" in hints):
        return "relationships"
    text = f"{finding.rule_name} {finding.expected_condition or ''}".lower()
    if any(k in text for k in ("not_null", "notnull", "not null", "non-null",
                               "missing", "completeness", "required", "present")):
        return "not_null"
    if any(k in text for k in ("unique", "duplicate", "distinct", "dedup")):
        return "unique"
    return "custom"


def findings_from_dict(
    report_dict: dict[str, Any], *, lineage_run_id: str | None = None
) -> list[QualityFinding]:
    """Reconstruct findings from a :meth:`CleanReport.to_dict` JSON payload.

    Used by the ``freshdata quality-ops`` CLI, which is handed a ``report.json``
    rather than a live report object. Reads ``domain_findings``, ``domain_repairs``,
    and risky ``actions`` — the same sources :meth:`CleanReport.to_findings` uses.
    """
    findings: list[QualityFinding] = []

    repairs_by_rule: dict[Any, list[dict[str, Any]]] = {}
    for repair in report_dict.get("domain_repairs", []):
        repairs_by_rule.setdefault(repair.get("rule_id"), []).append(repair)

    for f in report_dict.get("domain_findings", []):
        if f.get("status") != "violated":
            continue
        fields = tuple(f.get("fields") or ())
        column = fields[0] if fields else None
        applied = [r for r in repairs_by_rule.get(f.get("rule_id"), [])
                   if r.get("status") == "applied"]
        action_taken = None
        if applied:
            strategies = sorted({str(r.get("strategy")) for r in applied})
            action_taken = f"{', '.join(strategies)} ({len(applied)} row(s))"
        rows = list(f.get("violation_rows") or [])
        row_index = rows[0] if len(rows) == 1 else None
        row_selector = None
        if row_index is None and rows:
            row_selector = f"rows: {rows[:20]}"
        findings.append(QualityFinding.create(
            severity=f.get("severity", "warning"),
            step=f.get("layer") or "domain",
            column=column,
            rule_name=f.get("rule_id") or f.get("name") or f.get("check") or "rule",
            message=f.get("message") or "",
            row_index=row_index,
            row_selector=row_selector,
            expected_condition=f.get("check"),
            action_taken=action_taken,
            lineage_run_id=lineage_run_id,
            extra={"n_violations": f.get("n_violations"), "fields": list(fields)},
        ))

    for a in report_dict.get("actions", []):
        if a.get("risk") in ("medium", "high") and a.get("count"):
            findings.append(QualityFinding.create(
                severity=a.get("risk", "warning"),
                step=a.get("step") or "clean",
                column=a.get("column"),
                rule_name=a.get("step") or "action",
                message=a.get("description") or "",
                action_taken=a.get("description"),
                lineage_run_id=lineage_run_id,
            ))

    return findings
