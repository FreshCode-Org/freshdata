from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exact import JsonValue, TypedValue, encode_typed


class Disposition(str, Enum):
    PRESERVE = "preserve"
    REPAIR = "repair"
    FLAG = "flag"
    REVIEW = "review"


class _Unset:
    pass


UNSET = _Unset()
_TEST_DIGEST_KEY = b"truthbench-decision-record-for-test-v1"


def _disposition(value: Disposition | str) -> Disposition:
    try:
        return Disposition(value)
    except ValueError as exc:
        raise ValueError(f"unknown TruthBench disposition: {value!r}") from exc


@dataclass(frozen=True)
class GoldCell:
    cell_id: str
    fixture_version: str
    domain: str
    row_id: str
    column: str
    disposition: Disposition
    expected_output: TypedValue | None = None
    family: str | None = None
    sensitive: bool = False
    canary_id: str | None = None
    schema_version: int = field(default=1, init=False)

    @classmethod
    def create(
        cls,
        fixture_version: str,
        domain: str,
        row_id: str,
        column: str,
        disposition: Disposition | str,
        *,
        expected_output: Any = UNSET,
        expected_dtype: Any = None,
        family: str | None = None,
        sensitive: bool = False,
        canary_id: str | None = None,
        digest_key: bytes = _TEST_DIGEST_KEY,
    ) -> GoldCell:
        identifiers = (fixture_version, domain, row_id, column)
        if any(not isinstance(item, str) or not item for item in identifiers):
            raise ValueError("cell identity components must be non-empty strings")
        typed_output = None
        if expected_output is not UNSET:
            typed_output = encode_typed(
                expected_output,
                dtype=expected_dtype,
                sensitive=sensitive,
                digest_key=digest_key if sensitive else None,
            )
        return cls(
            cell_id=":".join(identifiers),
            fixture_version=fixture_version,
            domain=domain,
            row_id=row_id,
            column=column,
            disposition=_disposition(disposition),
            expected_output=typed_output,
            family=family,
            sensitive=sensitive,
            canary_id=canary_id,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "cell_id": self.cell_id,
            "fixture_version": self.fixture_version,
            "domain": self.domain,
            "row_id": self.row_id,
            "column": self.column,
            "disposition": self.disposition.value,
            "expected_output": (
                None if self.expected_output is None else self.expected_output.to_dict()
            ),
            "family": self.family,
            "sensitive": self.sensitive,
            "canary_id": self.canary_id,
        }


@dataclass(frozen=True)
class CaseExpectation:
    case_id: str
    fixture_version: str
    domain: str
    kind: str
    name: str
    disposition: Disposition
    expected: TypedValue | None = None
    family: str | None = None
    sensitive: bool = False
    schema_version: int = field(default=1, init=False)

    @classmethod
    def create(
        cls,
        fixture_version: str,
        domain: str,
        kind: str,
        name: str,
        disposition: Disposition | str,
        *,
        expected: Any = UNSET,
        expected_dtype: Any = None,
        family: str | None = None,
        sensitive: bool = False,
        digest_key: bytes = _TEST_DIGEST_KEY,
    ) -> CaseExpectation:
        identifiers = (fixture_version, domain, kind, name)
        if any(not isinstance(item, str) or not item for item in identifiers):
            raise ValueError("case identity components must be non-empty strings")
        typed_expected = None
        if expected is not UNSET:
            typed_expected = encode_typed(
                expected,
                dtype=expected_dtype,
                sensitive=sensitive,
                digest_key=digest_key if sensitive else None,
            )
        return cls(
            case_id=":".join(identifiers),
            fixture_version=fixture_version,
            domain=domain,
            kind=kind,
            name=name,
            disposition=_disposition(disposition),
            expected=typed_expected,
            family=family,
            sensitive=sensitive,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "fixture_version": self.fixture_version,
            "domain": self.domain,
            "kind": self.kind,
            "name": self.name,
            "disposition": self.disposition.value,
            "expected": None if self.expected is None else self.expected.to_dict(),
            "family": self.family,
            "sensitive": self.sensitive,
        }


@dataclass(frozen=True)
class DecisionRecord:
    record_id: str
    run_id: str
    fixture_id: str
    case_id: str | None
    cell_id: str
    domain: str
    row_id: str
    column: str
    surface: str
    repeat: int
    expected_disposition: Disposition
    actual_disposition: Disposition | None
    input: TypedValue
    expected_output: TypedValue | None
    actual_output: TypedValue | None
    confidence: float | None = None
    risk: str | None = None
    status: str | None = None
    rule_id: str | None = None
    rationale: str | None = None
    evidence_kinds: tuple[str, ...] | None = None
    mutated: bool | None = None
    detected: bool | None = None
    quarantined: bool | None = None
    human_review: bool | None = None
    audit_required: bool | None = None
    audit_complete: bool | None = None
    audit_ids: tuple[str, ...] | None = None
    trust_before: float | None = None
    trust_after: float | None = None
    trust_delta: float | None = None
    requested_backend: str | None = None
    actual_backend: str | None = None
    fallback_events: tuple[str, ...] | None = None
    backend_differences: tuple[str, ...] | None = None
    normalized_decision_hash: str | None = None
    repeat_hash: str | None = None
    repeat_consistent: bool | None = None
    schema_version: int = field(default=1, init=False)

    @classmethod
    def for_test(cls, *, cell: GoldCell, input_value: Any) -> DecisionRecord:
        typed_input = encode_typed(
            input_value,
            sensitive=cell.sensitive,
            digest_key=_TEST_DIGEST_KEY if cell.sensitive else None,
        )
        return cls(
            record_id=f"test:{cell.cell_id}",
            run_id="test",
            fixture_id=f"{cell.fixture_version}:{cell.domain}",
            case_id=None,
            cell_id=cell.cell_id,
            domain=cell.domain,
            row_id=cell.row_id,
            column=cell.column,
            surface="test",
            repeat=0,
            expected_disposition=cell.disposition,
            actual_disposition=None,
            input=typed_input,
            expected_output=cell.expected_output,
            actual_output=None,
        )

    @staticmethod
    def _typed(value: TypedValue | None) -> dict[str, JsonValue] | None:
        return None if value is None else value.to_dict()

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "run_id": self.run_id,
            "fixture_id": self.fixture_id,
            "case_id": self.case_id,
            "cell_id": self.cell_id,
            "domain": self.domain,
            "row_id": self.row_id,
            "column": self.column,
            "surface": self.surface,
            "repeat": self.repeat,
            "expected_disposition": self.expected_disposition.value,
            "actual_disposition": (
                None if self.actual_disposition is None else self.actual_disposition.value
            ),
            "input": self.input.to_dict(),
            "input_type": self.input.type_label,
            "expected_output": self._typed(self.expected_output),
            "expected_output_type": (
                None if self.expected_output is None else self.expected_output.type_label
            ),
            "actual_output": self._typed(self.actual_output),
            "actual_output_type": (
                None if self.actual_output is None else self.actual_output.type_label
            ),
            "confidence": self.confidence,
            "risk": self.risk,
            "status": self.status,
            "rule_id": self.rule_id,
            "rationale": self.rationale,
            "evidence_kinds": (
                None if self.evidence_kinds is None else list(self.evidence_kinds)
            ),
            "mutated": self.mutated,
            "detected": self.detected,
            "quarantined": self.quarantined,
            "human_review": self.human_review,
            "audit_required": self.audit_required,
            "audit_complete": self.audit_complete,
            "audit_ids": None if self.audit_ids is None else list(self.audit_ids),
            "trust_before": self.trust_before,
            "trust_after": self.trust_after,
            "trust_delta": self.trust_delta,
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "fallback_events": (
                None if self.fallback_events is None else list(self.fallback_events)
            ),
            "backend_differences": (
                None if self.backend_differences is None else list(self.backend_differences)
            ),
            "normalized_decision_hash": self.normalized_decision_hash,
            "repeat_hash": self.repeat_hash,
            "repeat_consistent": self.repeat_consistent,
        }


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    failures: tuple[str, ...] = ()
    schema_version: int = field(default=1, init=False)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "passed": self.passed,
            "failure_count": len(self.failures),
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class RunResult:
    run_id: str
    profile: str
    fixture_hashes: tuple[tuple[str, str], ...]
    required_backends: tuple[str, ...]
    records: tuple[DecisionRecord, ...]
    gates: tuple[GateResult, ...]
    summary: tuple[tuple[str, JsonValue], ...]
    environment: tuple[tuple[str, JsonValue], ...]
    schema_version: int = field(default=1, init=False)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "profile": self.profile,
            "fixture_hashes": dict(self.fixture_hashes),
            "required_backends": list(self.required_backends),
            "records": [record.to_dict() for record in self.records],
            "gates": [gate.to_dict() for gate in self.gates],
            "summary": dict(self.summary),
            "environment": dict(self.environment),
        }
