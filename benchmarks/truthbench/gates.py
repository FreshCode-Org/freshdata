"""Absolute, fail-closed TruthBench release gates."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType
from typing import Any

from .exact import TypedValue, encode_typed
from .models import DecisionRecord, Disposition, GateResult, RunResult
from .privacy import SinkScanner

_RAW_PII = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.invalid")
_BOILERPLATE = {"", "n/a", "none", "unknown", "automatic", "auto"}
_SURFACE_CLASSES = {"mutator", "validator", "pii", "explicit_transform"}


@dataclass(frozen=True)
class _FixtureEvidence:
    """A copy-protected oracle snapshot, never a live fixture DataFrame."""

    domain: str
    cells: Mapping[str, Any]
    inputs: Mapping[str, TypedValue]
    protected_columns: frozenset[str]
    canaries: tuple[tuple[str, Any], ...]
    required_case_ids: frozenset[str]


def _snapshot(fixture: Any) -> _FixtureEvidence:
    frame = fixture.frame
    cells = {cell.cell_id: cell for cell in fixture.cells}
    inputs = {
        cell.cell_id: encode_typed(
            frame.at[cell.row_id, cell.column], dtype=frame[cell.column].dtype
        )
        for cell in fixture.cells
        if not cell.sensitive
    }
    cases = (*getattr(fixture, "row_cases", ()), *getattr(fixture, "schema_cases", ()))
    return _FixtureEvidence(
        domain=str(fixture.domain),
        cells=MappingProxyType(cells),
        inputs=MappingProxyType(inputs),
        protected_columns=frozenset(str(column) for column in fixture.protected_columns),
        canaries=tuple((str(key), value) for key, value in fixture.pii_canaries.items()),
        required_case_ids=frozenset(case.case_id for case in cases),
    )


@dataclass(frozen=True)
class GateRun:
    """Immutable gate context paired with the existing serialized ``RunResult``.

    The public result schema remains untouched.  At construction, live fixture
    frames are converted into typed, mapping-proxy oracle snapshots so later
    fixture mutation cannot change a release decision.
    """

    run: RunResult
    fixtures: tuple[Any, ...]
    generated_code: tuple[str, ...] = ()
    persisted_sinks: tuple[Any, ...] = ()
    complete: bool = True
    schema_valid: bool = True
    unexpected_exceptions: tuple[Any, ...] = ()
    expected_surfaces: tuple[str, ...] = ()
    expected_backends: tuple[str, ...] = ()
    expected_repeats: tuple[int, ...] = ()
    surface_classes: tuple[tuple[str, str], ...] = ()
    requested_cell_ids: tuple[str, ...] = ()
    case_records: tuple[Any, ...] = ()
    audit_record_ids: tuple[str, ...] = ()
    _evidence: tuple[_FixtureEvidence, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_evidence", tuple(_snapshot(fixture) for fixture in self.fixtures)
        )


@dataclass(frozen=True)
class GateEvaluation:
    gates: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)


def _gate(name: str, failures: Iterable[str]) -> GateResult:
    return GateResult(name, not (items := tuple(failures)), items)


def _cells(context: GateRun) -> dict[str, tuple[_FixtureEvidence, Any]]:
    return {
        cell_id: (evidence, cell)
        for evidence in context._evidence
        for cell_id, cell in evidence.cells.items()
    }


def _surface_class(context: GateRun, surface: str) -> str | None:
    return dict(context.surface_classes).get(surface)


def _is_mutator(context: GateRun, record: DecisionRecord) -> bool:
    return _surface_class(context, record.surface) in {"mutator", "explicit_transform"}


def _changed(record: DecisionRecord) -> bool:
    return record.actual_output is not None and record.actual_output != record.input


def _exact_repair(actual: TypedValue | None, expected: TypedValue | None) -> bool:
    if actual is None or expected is None:
        return False
    if expected.dtype is not None:
        return actual == expected
    # Fixture authors who do not specify a dtype constrain scalar type/value,
    # not an incidental object/string backend representation.
    return (
        actual.kind == expected.kind
        and actual.value == expected.value
        and actual.redacted == expected.redacted
    )


def _substantive(record: DecisionRecord, audit_ids: set[str]) -> bool:
    rationale = (record.rationale or "").strip()
    return (
        rationale.casefold() not in _BOILERPLATE
        and len(rationale) >= 12
        and bool((record.rule_id or "").strip())
        and record.audit_complete is True
        and bool(set(record.audit_ids or ()) & audit_ids)
    )


def _canary_leaks(context: GateRun) -> tuple[str, ...]:
    canaries = {key: value for evidence in context._evidence for key, value in evidence.canaries}
    failures: list[str] = []
    if canaries:
        scanner = SinkScanner.from_canaries(canaries, key=b"truthbench-gate-scan-v1")
        for index, sink in enumerate(context.persisted_sinks):
            for leak in scanner.scan(sink):
                failures.append(f"sink[{index}] leaked canary {leak.canary_id} at {leak.path}")
    for index, sink in enumerate(context.persisted_sinks):
        if _RAW_PII.search(str(sink)):
            failures.append(f"sink[{index}] contains raw synthetic PII")
    return tuple(failures)


def _coverage_failures(context: GateRun, cells: Mapping[str, Any]) -> list[str]:
    if (
        not context.expected_surfaces
        or not context.expected_backends
        or not context.expected_repeats
    ):
        return ["expected surface/backend/repeat coverage was not declared"]
    failures: list[str] = []
    for cell_id, surface, backend, repeat in product(
        cells, context.expected_surfaces, context.expected_backends, context.expected_repeats
    ):
        matches = [
            record
            for record in context.run.records
            if record.cell_id == cell_id
            and record.surface == surface
            and record.requested_backend == backend
            and record.repeat == repeat
        ]
        if len(matches) != 1:
            failures.append(
                f"{cell_id}: expected exactly one {surface}/{backend}/repeat-{repeat} record"
            )
    declared = dict(context.surface_classes)
    for surface in context.expected_surfaces:
        if declared.get(surface) not in _SURFACE_CLASSES:
            failures.append(f"{surface}: surface class was not declared")
    return failures


def evaluate_gates(context: GateRun) -> GateEvaluation:
    """Evaluate all gates independently.  No baseline or stale claim clears one."""

    run, records, cells = context.run, context.run.records, _cells(context)
    audit_ids = set(context.audit_record_ids)
    gates: list[GateResult] = []
    coverage = _coverage_failures(context, cells) if cells else ["fixture policy/cells absent"]
    gates.append(
        _gate(
            "completeness",
            ["run is partial"] * (not context.complete or not records) + coverage,
        )
    )
    gates.append(
        _gate("schema_validation", ["schema validation failed"] * (not context.schema_valid))
    )
    gates.append(
        _gate("fixture_evidence", ["fixture policy/cell evidence is absent"] * (not cells))
    )
    gates.append(
        _gate(
            "unexpected_exception",
            (f"unexpected exception: {e}" for e in context.unexpected_exceptions),
        )
    )
    required = set(context.expected_backends or run.required_backends)
    actual = {record.actual_backend for record in records if record.actual_backend}
    gates.append(
        _gate(
            "required_backend",
            (f"required backend {b} was not executed" for b in sorted(required - actual)),
        )
    )
    gates.append(
        _gate(
            "valid_value_corruption",
            (
                f"{record.record_id}: preserve value was corrupted"
                for record in records
                if _is_mutator(context, record)
                and record.expected_disposition is Disposition.PRESERVE
                and _changed(record)
            ),
        )
    )
    gates.append(
        _gate(
            "protected_column_modification",
            (
                f"{record.record_id}: protected column {record.column} was modified"
                for record in records
                if (item := cells.get(record.cell_id)) is not None
                and record.column in item[0].protected_columns
                and (record.mutated is True or _changed(record))
            ),
        )
    )
    gates.append(_gate("raw_pii_leakage", _canary_leaks(context)))
    gates.append(
        _gate(
            "backend_inconsistency",
            (
                f"{record.record_id}: missing or inconsistent backend disclosure"
                for record in records
                if not record.requested_backend
                or not record.actual_backend
                or record.requested_backend != record.actual_backend
            ),
        )
    )
    gates.append(
        _gate(
            "default_nondeterminism",
            (
                f"{record.record_id}: repeat decision is absent or inconsistent"
                for record in records
                if record.repeat_consistent is not True or not record.repeat_hash
            ),
        )
    )
    broken = []
    for index, code in enumerate(context.generated_code):
        try:
            ast.parse(code)
        except SyntaxError:
            broken.append(f"generated code[{index}] does not parse")
    gates.append(_gate("broken_generated_code", broken))
    gates.append(
        _gate(
            "unexplained_high_confidence",
            (
                f"{record.record_id}: high-confidence decision lacks rationale/provenance/audit"
                for record in records
                if record.confidence is not None
                and record.confidence >= 0.90
                and not _substantive(record, audit_ids)
            ),
        )
    )
    gates.append(
        _gate(
            "trust_inversion",
            (
                f"{record.record_id}: trust increased after data changed"
                for record in records
                if (record.mutated is True or _changed(record))
                and record.trust_before is not None
                and record.trust_after is not None
                and record.trust_after > record.trust_before
            ),
        )
    )
    gates.append(
        _gate(
            "mutation_audit",
            (
                f"{record.record_id}: decision lacks matching audit evidence"
                for record in records
                if (record.mutated or record.detected)
                and (
                    record.audit_complete is not True
                    or not set(record.audit_ids or ()) & audit_ids
                )
            ),
        )
    )
    gates.append(
        _gate(
            "review_routing",
            (
                f"{record.record_id}: review case was not routed to a human"
                for record in records
                if record.expected_disposition is Disposition.REVIEW
                and record.human_review is not True
            ),
        )
    )
    gates.append(
        _gate(
            "exact_repair",
            (
                f"{record.record_id}: repair differs from the exact oracle"
                for record in records
                if _is_mutator(context, record)
                and record.expected_disposition is Disposition.REPAIR
                and not _exact_repair(record.actual_output, record.expected_output)
            ),
        )
    )
    gates.append(
        _gate(
            "flag_mutation",
            (
                f"{record.record_id}: flag/review value was mutated"
                for record in records
                if _is_mutator(context, record)
                and record.expected_disposition in {Disposition.FLAG, Disposition.REVIEW}
                and (record.mutated is True or _changed(record))
            ),
        )
    )
    gates.append(
        _gate(
            "validator_contract",
            (
                f"{record.record_id}: validator mutated data or missed detection"
                for record in records
                if _surface_class(context, record.surface) == "validator"
                and (
                    record.mutated is True
                    or _changed(record)
                    or (
                        record.expected_disposition is not Disposition.PRESERVE
                        and record.detected is not True
                    )
                )
            ),
        )
    )
    gates.append(
        _gate(
            "pii_scope",
            (
                f"{record.record_id}: PII surface missed a canary or changed a false positive"
                for record in records
                if _surface_class(context, record.surface) == "pii"
                and (
                    (record.sensitive and record.detected is not True)
                    or (
                        not record.sensitive
                        and (record.mutated is True or _changed(record) or record.detected is True)
                    )
                )
            ),
        )
    )
    requested = set(context.requested_cell_ids)
    gates.append(
        _gate(
            "requested_behavior",
            (
                (
                    f"{record.record_id}: explicit transform changed an unrequested "
                    "cell or missed requested behavior"
                )
                for record in records
                if _surface_class(context, record.surface) == "explicit_transform"
                and (
                    (
                        record.cell_id not in requested
                        and (record.mutated is True or _changed(record))
                    )
                    or (
                        record.cell_id in requested
                        and record.expected_disposition is Disposition.REPAIR
                        and not _exact_repair(record.actual_output, record.expected_output)
                    )
                )
            ),
        )
    )
    gates.append(
        _gate(
            "input_mutation",
            (
                f"{record.record_id}: input evidence differs from fixture snapshot"
                for record in records
                if (item := cells.get(record.cell_id)) is not None
                and not record.sensitive
                and record.input != item[0].inputs[record.cell_id]
            ),
        )
    )
    required_cases = {
        case for evidence in context._evidence for case in evidence.required_case_ids
    }
    observed_cases = {
        case.case_id for case in context.case_records if getattr(case, "observed", False)
    }
    gates.append(
        _gate(
            "case_coverage",
            (
                f"required case {case} was not observed"
                for case in sorted(required_cases - observed_cases)
            ),
        )
    )
    gates.append(
        _gate(
            "aggregate_consistency",
            ["summary record aggregate does not match records"]
            * (dict(run.summary).get("records") != len(records)),
        )
    )
    return GateEvaluation(tuple(gates))


def failed_gate_names(result: GateEvaluation) -> set[str]:
    return {gate.name for gate in result.gates if not gate.passed}


__all__ = ["GateEvaluation", "GateRun", "evaluate_gates", "failed_gate_names"]
