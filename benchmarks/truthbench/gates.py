"""Absolute, fail-closed TruthBench release gates."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .exact import encode_typed
from .models import DecisionRecord, Disposition, GateResult, RunResult
from .privacy import SinkScanner

_RAW_PII = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.invalid")
_BOILERPLATE = {"", "n/a", "none", "unknown", "automatic", "auto"}


@dataclass(frozen=True)
class GateRun:
    """Immutable, non-serialized evidence required to grade a ``RunResult``.

    Fixture policy, canaries and ``CaseRecord`` evidence are intentionally not
    added to ``RunResult``: the existing public schema remains stable.  Missing
    fixture evidence is a release failure, never an assumed pass.
    """

    run: RunResult
    fixtures: tuple[Any, ...]
    generated_code: tuple[str, ...] = ()
    persisted_sinks: tuple[Any, ...] = ()
    complete: bool = True
    schema_valid: bool = True
    unexpected_exceptions: tuple[Any, ...] = ()


@dataclass(frozen=True)
class GateEvaluation:
    gates: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)


def _gate(name: str, failures: Iterable[str]) -> GateResult:
    return GateResult(name, not (items := tuple(failures)), items)


def _fixture_cells(context: GateRun) -> dict[str, Any]:
    return {
        cell.cell_id: (fixture, cell) for fixture in context.fixtures for cell in fixture.cells
    }


def _records_for(run: RunResult, cell_id: str) -> tuple[DecisionRecord, ...]:
    return tuple(record for record in run.records if record.cell_id == cell_id)


def _typed_equal(left: Any, right: Any) -> bool:
    return left == right


def _has_substantive_explanation(record: DecisionRecord) -> bool:
    rationale = (record.rationale or "").strip()
    return (
        rationale.casefold() not in _BOILERPLATE
        and len(rationale) >= 12
        and bool((record.rule_id or "").strip())
        and record.audit_complete is True
        and bool(record.audit_ids)
    )


def _canary_leaks(context: GateRun) -> tuple[str, ...]:
    canaries: dict[str, Any] = {}
    for fixture in context.fixtures:
        canaries.update(getattr(fixture, "pii_canaries", {}))
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


def evaluate_gates(context: GateRun) -> GateEvaluation:
    """Evaluate every release gate independently; no baseline can clear one."""

    run = context.run
    cells = _fixture_cells(context)
    records = run.records
    gates: list[GateResult] = []
    gates.append(
        _gate(
            "completeness",
            (
                "run is partial"
                for _ in [None]
                if not context.complete
                or not records
                or not all(record.surface for record in records)
            ),
        )
    )
    gates.append(
        _gate(
            "schema_validation", ("schema validation failed",) if not context.schema_valid else ()
        )
    )
    gates.append(
        _gate("fixture_evidence", ("fixture policy/cell evidence is absent",) if not cells else ())
    )
    gates.append(
        _gate(
            "unexpected_exception",
            (f"unexpected exception: {value}" for value in context.unexpected_exceptions),
        )
    )
    required = set(run.required_backends)
    actual = {record.actual_backend for record in records if record.actual_backend}
    gates.append(
        _gate(
            "required_backend",
            (
                f"required backend {backend} was not executed"
                for backend in sorted(required - actual)
            ),
        )
    )
    gates.append(
        _gate(
            "valid_value_corruption",
            (
                f"{record.record_id}: preserve value was mutated"
                for record in records
                if record.expected_disposition is Disposition.PRESERVE
                and record.actual_output is not None
                and record.actual_output != record.input
            ),
        )
    )
    protected = {
        (fixture.domain, column)
        for fixture in context.fixtures
        for column in getattr(fixture, "protected_columns", ())
    }
    gates.append(
        _gate(
            "protected_column_modification",
            (
                f"{record.record_id}: protected column {record.column} was modified"
                for record in records
                if (record.domain, record.column) in protected and record.mutated is True
            ),
        )
    )
    gates.append(_gate("raw_pii_leakage", _canary_leaks(context)))
    gates.append(
        _gate(
            "backend_inconsistency",
            (
                (
                    f"{record.record_id}: requested backend "
                    f"{record.requested_backend!r} differs from actual "
                    f"{record.actual_backend!r}"
                )
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
                f"{record.record_id}: repeat decision is inconsistent"
                for record in records
                if record.repeat_consistent is False
            ),
        )
    )
    broken_code: list[str] = []
    for index, code in enumerate(context.generated_code):
        try:
            ast.parse(code)
        except SyntaxError:
            broken_code.append(f"generated code[{index}] does not parse")
    gates.append(_gate("broken_generated_code", broken_code))
    gates.append(
        _gate(
            "unexplained_high_confidence",
            (
                f"{record.record_id}: high-confidence decision lacks explanation/provenance/audit"
                for record in records
                if record.confidence is not None
                and record.confidence >= 0.90
                and not _has_substantive_explanation(record)
            ),
        )
    )
    gates.append(
        _gate(
            "trust_inversion",
            (
                f"{record.record_id}: trust increased after a mutation"
                for record in records
                if record.mutated is True
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
                f"{record.record_id}: mutation/detection lacks audit"
                for record in records
                if (record.mutated or record.detected)
                and (record.audit_complete is not True or not record.audit_ids)
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
                if record.expected_disposition is Disposition.REPAIR
                and (
                    record.actual_output is None
                    or not _typed_equal(record.actual_output, record.expected_output)
                )
            ),
        )
    )
    gates.append(
        _gate(
            "flag_mutation",
            (
                f"{record.record_id}: flag/review value was mutated"
                for record in records
                if record.expected_disposition in {Disposition.FLAG, Disposition.REVIEW}
                and record.mutated is True
            ),
        )
    )
    # The input gate compares the original typed source, never a repair oracle.
    input_failures: list[str] = []
    for record in records:
        fixture_cell = cells.get(record.cell_id)
        if fixture_cell is None or record.sensitive:
            continue
        fixture, cell = fixture_cell
        source = fixture.frame.at[cell.row_id, cell.column]
        source_typed = encode_typed(source, dtype=fixture.frame[cell.column].dtype)
        if record.input != source_typed:
            input_failures.append(f"{record.record_id}: input evidence differs from fixture")
    gates.append(_gate("input_mutation", input_failures))
    gates.append(
        _gate(
            "aggregate_consistency",
            (
                "summary record aggregate does not match records"
                for _ in [None]
                if dict(run.summary).get("records") != len(records)
            ),
        )
    )
    return GateEvaluation(tuple(gates))


def failed_gate_names(result: GateEvaluation) -> set[str]:
    return {gate.name for gate in result.gates if not gate.passed}


__all__ = ["GateEvaluation", "GateRun", "evaluate_gates", "failed_gate_names"]
