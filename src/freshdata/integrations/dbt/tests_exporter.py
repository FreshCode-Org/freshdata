"""Export freshdata findings as dbt generic tests (``schema.yml``).

This is the *exporter* half of the dbt integration — distinct from the trust-gate in
:mod:`freshdata.integrations.dbt` (``FreshDataDbtTransform`` / ``gate_manifest``). It
turns a report's :class:`~freshdata.QualityFinding` objects into the standard dbt
``version: 2`` schema YAML, mapping each finding to the most specific generic test it
can (``not_null``, ``unique``, ``accepted_values``, ``relationships``) and falling back
to a bundled ``freshdata_expectation`` custom test for advanced findings.

YAML is emitted by a small internal writer (:func:`_emit`), so exporting needs **no
extra dependency** — it works on a bare ``pip install freshdata``.

Example
-------
>>> import freshdata as fd
>>> import pandas as pd
>>> df = pd.DataFrame({"email": [None, "a@b.com"], "qty": [1, 2]})
>>> _, report = fd.clean(df, return_report=True)
>>> yaml_text = fd.export_dbt_tests(report, model_name="orders", path="/tmp/schema.yml")
>>> yaml_text.splitlines()[0]
'version: 2'
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...findings import QualityFinding, classify_finding

__all__ = ["export_dbt_tests"]

#: Canonical severity -> dbt test ``config.severity`` (dbt only knows warn/error).
_DEFAULT_DBT_SEVERITY = {"info": "warn", "warning": "warn", "error": "error"}

_YAML_SAFE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
_YAML_RESERVED = {"true", "false", "null", "yes", "no", "on", "off", "none", "~"}


def _is_number_like(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _scalar(value: Any) -> str:
    """Render one YAML scalar, quoting only when needed to preserve meaning."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    text = str(value)
    if not text:
        return '""'
    if (_YAML_SAFE.match(text) and not _is_number_like(text)
            and text.lower() not in _YAML_RESERVED):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _emit(obj: Any, indent: int = 0) -> list[str]:
    """Serialize nested dict/list/scalar structures to block-style YAML lines."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, dict):
                lines.append(f"{pad}{key}:" if val else f"{pad}{key}: {{}}")
                if val:
                    lines.extend(_emit(val, indent + 1))
            elif isinstance(val, (list, tuple)):
                lines.append(f"{pad}{key}:" if val else f"{pad}{key}: []")
                if val:
                    lines.extend(_emit(list(val), indent + 1))
            else:
                lines.append(f"{pad}{key}: {_scalar(val)}")
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, dict) and item:
                sub = _emit(item, indent + 1)
                head = sub[0][len("  " * (indent + 1)):]
                lines.append(f"{pad}- {head}")
                lines.extend(sub[1:])
            else:
                lines.append(f"{pad}- {_scalar(item)}")
    else:
        lines.append(f"{pad}{_scalar(obj)}")
    return lines


def _resolve_findings(report_or_findings: Any) -> list[QualityFinding]:
    """Accept a report (anything with ``to_findings``) or a findings iterable."""
    if hasattr(report_or_findings, "to_findings"):
        return list(report_or_findings.to_findings())
    return list(report_or_findings)


def _dbt_severity(severity: str, severity_map: dict[str, str]) -> str:
    return severity_map.get(severity, "warn")


def _custom_test(finding: QualityFinding, severity_map: dict[str, str]) -> dict[str, Any]:
    """A ``freshdata_expectation`` generic test carrying the finding's metadata."""
    return {"freshdata_expectation": {
        "rule_name": finding.rule_name,
        "severity_label": finding.severity,
        "message": finding.message,
        "config": {"severity": _dbt_severity(finding.severity, severity_map)},
    }}


def _column_tests(
    findings: list[QualityFinding], severity_map: dict[str, str]
) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in findings:
        kind = classify_finding(finding)
        sev = _dbt_severity(finding.severity, severity_map)
        if kind in ("not_null", "unique"):
            if kind in seen:
                continue
            seen.add(kind)
            tests.append({kind: {"config": {"severity": sev}}})
        elif kind == "accepted_values":
            tests.append({"accepted_values": {
                "values": list(finding.extra.get("value_set", [])),
                "config": {"severity": sev},
            }})
        elif kind == "relationships":
            tests.append({"relationships": {
                "to": finding.extra.get("to_model"),
                "field": finding.extra.get("to_field"),
                "config": {"severity": sev},
            }})
        else:  # regex / between / custom -> no native dbt generic test
            tests.append(_custom_test(finding, severity_map))
    return tests


def export_dbt_tests(
    report_or_findings: Any,
    model_name: str,
    path: str,
    *,
    severity_map: dict[str, str] | None = None,
) -> str:
    """Export findings as a dbt ``schema.yml`` for *model_name*; write it to *path*.

    Parameters
    ----------
    report_or_findings:
        A freshdata report (``CleanReport``, ``DriftReport``, ``PrivacyReport``,
        ``EntityResolutionReport`` — anything exposing ``to_findings()``) or an
        explicit list of :class:`~freshdata.QualityFinding`.
    model_name:
        The dbt model the tests attach to.
    path:
        Destination ``.yml`` file (parent directories are created as needed).
    severity_map:
        Optional override of the canonical-severity -> ``warn``/``error`` mapping.

    Returns
    -------
    str
        The YAML text that was written.
    """
    findings = _resolve_findings(report_or_findings)
    severity_map = {**_DEFAULT_DBT_SEVERITY, **(severity_map or {})}

    by_column: dict[str | None, list[QualityFinding]] = {}
    for finding in findings:
        by_column.setdefault(finding.column, []).append(finding)

    columns_block: list[dict[str, Any]] = []
    for col in (c for c in by_column if c is not None):
        tests = _column_tests(by_column[col], severity_map)
        if tests:
            columns_block.append({"name": col, "tests": tests})

    model_tests = [_custom_test(f, severity_map) for f in by_column.get(None, [])]

    model_entry: dict[str, Any] = {"name": model_name}
    if columns_block:
        model_entry["columns"] = columns_block
    if model_tests:
        model_entry["tests"] = model_tests

    schema = {"version": 2, "models": [model_entry]}
    text = "\n".join(_emit(schema)) + "\n"

    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
