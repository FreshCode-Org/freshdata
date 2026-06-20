"""Domain-specific validator packs for :func:`freshdata.clean`.

A domain pack validates and (separately) repairs a specific kind of tabular data
against versioned, config-driven rules, extending the existing clean audit trail
with domain findings and a trust score. Use it through the normal entry point::

    import freshdata as fd
    df_out = fd.clean(df, domain="finance")
    df_out, report = fd.clean(df, domain="finance", return_report=True)

Third-party packs register via the ``freshdata.domains`` entry-point group; see
``CONTRIBUTING_DOMAINS.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .base import (
    LAYERS,
    MISSING_REQUIRED_FIELD,
    SEVERITIES,
    SEVERITY_TO_RISK,
    ColumnMapping,
    ConfigDrivenValidator,
    DomainError,
    DomainValidator,
    RepairAction,
    RepairLog,
    Rule,
    RuleResult,
    ValidationReport,
)
from .registry import (
    UnknownDomainError,
    available,
    get_validator,
    register,
    validator_class,
)

__all__ = [
    "LAYERS",
    "MISSING_REQUIRED_FIELD",
    "SEVERITIES",
    "SEVERITY_TO_RISK",
    "ColumnMapping",
    "ConfigDrivenValidator",
    "DomainError",
    "DomainOutcome",
    "DomainValidator",
    "RepairAction",
    "RepairLog",
    "Rule",
    "RuleResult",
    "UnknownDomainError",
    "ValidationReport",
    "available",
    "get_validator",
    "register",
    "run_domain",
    "validator_class",
]


@dataclass
class DomainOutcome:
    """Everything a domain run produced, beyond the repaired frame."""

    report: ValidationReport
    repairs: RepairLog

    @property
    def domain(self) -> str:
        return self.report.domain

    @property
    def trust_score(self) -> float:
        return self.report.domain_trust_score


def run_domain(
    df: pd.DataFrame,
    domain: str,
    *,
    column_map: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, DomainOutcome]:
    """Validate then (separately) repair *df* with the named domain pack.

    Returns ``(repaired_df, outcome)``. Validation never mutates *df*; repair
    runs afterward and never touches identifier columns. Raises
    :class:`UnknownDomainError` if *domain* is not registered.
    """
    validator = get_validator(domain, column_map=column_map, **kwargs)
    original_index = df.index
    working = df
    if not original_index.is_unique:
        working = df.copy(deep=False)
        working.index = pd.RangeIndex(len(working))
    report = validator.validate(working)
    repaired, repairs = validator.repair(working, report)
    if working is not df:
        repaired.index = original_index.take(repaired.index.to_numpy(dtype=int))
        for result in report.results:
            result.violation_rows = [
                original_index[row] if isinstance(row, int) else row
                for row in result.violation_rows
            ]
        for action in repairs.actions:
            if isinstance(action.row, int):
                action.row = original_index[action.row]
    return repaired, DomainOutcome(report, repairs)
