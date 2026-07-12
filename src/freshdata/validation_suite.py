"""Declarative validation suites — Great-Expectations-shaped, contract-backed.

A :class:`ValidationSuite` is a named, versioned, serializable set of rules
checked against a DataFrame::

    import freshdata as fd

    suite = fd.ValidationSuite(
        name="customers",
        rules=[
            fd.ColumnRule("customer_id", dtype="string", nullable=False, unique=True),
            fd.ColumnRule("age", min_value=0, max_value=120),
        ],
    )
    result = fd.validate(df, suite=suite)
    result.raise_if_failed()

Suites compile to :class:`~freshdata.DataContract` and run through the same
check engine as contracts — there is one validation engine, two front doors
(:meth:`ValidationSuite.from_contract` migrates existing contracts).

Validation is **read-only**: the input frame is never modified. Execution is
pandas; non-pandas inputs are materialized and that materialization is
recorded on :attr:`ValidationResult.execution` so it is never silent.
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .enterprise.contracts import (
    ColumnContract,
    DataContract,
    DriftFinding,
    DriftReport,
    _add,
    _round,
    enforce_contract,
)

SUITE_SCHEMA_VERSION = "freshdata-suite-v1"

_CROSS_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


class ColumnRule(ColumnContract):
    """A single column's expectations — alias of :class:`ColumnContract`.

    Exists so validation-first users get a validation-first name; any
    :class:`ColumnContract` is accepted wherever a rule is expected.
    """


@dataclass(frozen=True)
class CrossColumnRule:
    """Require ``left <op> right`` to hold row-wise (rows with nulls skipped)."""

    left: str
    op: str
    right: str
    mostly: float = 1.0

    def __post_init__(self) -> None:
        if self.op not in _CROSS_OPS:
            raise ValueError(
                f"op must be one of {sorted(_CROSS_OPS)}, got {self.op!r}"
            )
        if not (0.0 < self.mostly <= 1.0):
            raise ValueError(f"mostly must be in (0, 1], got {self.mostly}")

    def to_dict(self) -> dict[str, Any]:
        return {"left": self.left, "op": self.op, "right": self.right, "mostly": self.mostly}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CrossColumnRule:
        return cls(
            left=d["left"], op=d["op"], right=d["right"], mostly=d.get("mostly", 1.0)
        )


class ValidationError(Exception):
    """Raised by :meth:`ValidationResult.raise_if_failed` on failed validation."""

    def __init__(self, message: str, *, result: ValidationResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class ValidationResult:
    """The outcome of validating a frame against a :class:`ValidationSuite`."""

    suite_name: str
    passed: bool
    report: DriftReport
    #: How validation actually executed: backend plus any materialization or
    #: fallback events — never silent about leaving the requested backend.
    execution: dict[str, Any] = field(default_factory=dict)

    @property
    def findings(self) -> list:
        """Warned/failed findings as :class:`~freshdata.QualityFinding` objects."""
        return self.report.to_findings()

    @property
    def n_errors(self) -> int:
        return self.report.n_errors

    @property
    def n_warnings(self) -> int:
        return self.report.n_warnings

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        lines = [f.message for f in self.report.errors[:5]]
        more = self.report.n_errors - len(lines)
        summary = "; ".join(lines) + (f"; +{more} more" if more > 0 else "")
        raise ValidationError(
            f"validation suite {self.suite_name!r} failed with "
            f"{self.report.n_errors} error(s): {summary}",
            result=self,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "passed": self.passed,
            "execution": self.execution,
            **self.report.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass(frozen=True)
class ValidationSuite:
    """A named, versioned, serializable set of validation rules."""

    name: str
    rules: tuple[ColumnContract, ...] = ()
    cross_column: tuple[CrossColumnRule, ...] = ()
    version: str = "1.0.0"
    min_rows: int | None = None
    max_rows: int | None = None
    compound_unique: tuple[tuple[str, ...], ...] = ()
    #: Require an exact schema match: no undeclared columns, no missing ones.
    strict_columns: bool = False
    allow_extra_columns: bool = True
    trust_score_min: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "cross_column", tuple(self.cross_column))
        object.__setattr__(
            self, "compound_unique", tuple(tuple(g) for g in self.compound_unique)
        )
        for rule in self.rules:
            if not isinstance(rule, ColumnContract):
                raise TypeError(
                    f"rules must be ColumnRule/ColumnContract, got {type(rule).__name__}"
                )

    # -- contract bridge ---------------------------------------------------

    def to_contract(self) -> DataContract:
        """Compile this suite to the :class:`DataContract` that executes it."""
        return DataContract(
            name=self.name,
            version=self.version,
            columns=tuple(self.rules),
            strict_columns=self.strict_columns,
            allow_extra_columns=self.allow_extra_columns,
            trust_score_min=self.trust_score_min,
            min_rows=self.min_rows,
            max_rows=self.max_rows,
            compound_unique=self.compound_unique,
        )

    @classmethod
    def from_contract(cls, contract: DataContract) -> ValidationSuite:
        """Migrate an existing :class:`DataContract` into a suite."""
        return cls(
            name=contract.name,
            version=contract.version,
            rules=tuple(ColumnRule.from_dict(c.to_dict()) for c in contract.columns),
            strict_columns=contract.strict_columns,
            allow_extra_columns=contract.allow_extra_columns,
            trust_score_min=contract.trust_score_min,
            min_rows=contract.min_rows,
            max_rows=contract.max_rows,
            compound_unique=contract.compound_unique,
        )

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SUITE_SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "rules": [r.to_dict() for r in self.rules],
            "cross_column": [r.to_dict() for r in self.cross_column],
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "compound_unique": [list(g) for g in self.compound_unique],
            "strict_columns": self.strict_columns,
            "allow_extra_columns": self.allow_extra_columns,
            "trust_score_min": self.trust_score_min,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ValidationSuite:
        schema = d.get("schema_version", SUITE_SCHEMA_VERSION)
        if schema != SUITE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported suite schema_version {schema!r}; "
                f"this freshdata reads {SUITE_SCHEMA_VERSION!r}"
            )
        return cls(
            name=d["name"],
            version=d.get("version", "1.0.0"),
            rules=tuple(ColumnRule.from_dict(r) for r in d.get("rules", ())),
            cross_column=tuple(
                CrossColumnRule.from_dict(r) for r in d.get("cross_column", ())
            ),
            min_rows=d.get("min_rows"),
            max_rows=d.get("max_rows"),
            compound_unique=tuple(tuple(g) for g in d.get("compound_unique", ())),
            strict_columns=d.get("strict_columns", False),
            allow_extra_columns=d.get("allow_extra_columns", True),
            trust_score_min=d.get("trust_score_min"),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> ValidationSuite:
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ValidationSuite:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _check_cross_column(
    findings: list[DriftFinding], rule: CrossColumnRule, frame: pd.DataFrame
) -> None:
    missing = [c for c in (rule.left, rule.right) if c not in frame.columns]
    label = f"{rule.left} {rule.op} {rule.right}"
    if missing:
        _add(
            findings,
            "suite.cross_column",
            level="error",
            status="failed",
            message=f"cross-column rule {label!r} references missing column(s) {missing}",
            metric="cross_column",
            threshold=label,
        )
        return
    both = frame[[rule.left, rule.right]].dropna()
    n_total = len(both)
    if not n_total:
        return
    try:
        holds = _CROSS_OPS[rule.op](both[rule.left], both[rule.right])
    except TypeError:
        _add(
            findings,
            "suite.cross_column",
            level="error",
            status="failed",
            message=f"cross-column rule {label!r} could not compare column dtypes "
            f"({frame[rule.left].dtype} vs {frame[rule.right].dtype})",
            metric="cross_column",
            threshold=label,
        )
        return
    n_bad = int((~holds).sum())
    if not n_bad:
        return
    ratio = n_bad / n_total
    tolerated = rule.mostly < 1.0 and ratio <= (1.0 - rule.mostly)
    _add(
        findings,
        "suite.cross_column",
        level="warning" if tolerated else "error",
        status="warned" if tolerated else "failed",
        message=f"cross-column rule {label!r} fails for {n_bad}/{n_total} "
        f"row(s) ({ratio:.2%})",
        metric="cross_column",
        threshold=label,
        current_value=n_bad,
        details={
            "violation_ratio": _round(ratio),
            "n_violations": n_bad,
            "mostly": rule.mostly,
        },
    )


def run_suite(df: Any, suite: ValidationSuite) -> ValidationResult:
    """Validate *df* against *suite*. Read-only: *df* is never modified."""
    from .adapters.polars import to_pandas  # noqa: PLC0415  (heavy import kept lazy)

    execution: dict[str, Any] = {"backend": "pandas", "fallback": []}
    if not isinstance(df, pd.DataFrame):
        execution["fallback"].append(
            {
                "step": "validate",
                "reason": f"validation executes on pandas; "
                f"{type(df).__name__} input was materialized",
            }
        )
    frame = to_pandas(df)
    report = enforce_contract(frame, suite.to_contract())
    for rule in suite.cross_column:
        _check_cross_column(report.findings, rule, frame)
    return ValidationResult(
        suite_name=suite.name,
        passed=report.passed,
        report=report,
        execution=execution,
    )
