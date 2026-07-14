from __future__ import annotations

from dataclasses import replace

import pytest
from benchmarks.truthbench.exact import encode_typed
from benchmarks.truthbench.gates import GateRun, evaluate_gates, failed_gate_names
from benchmarks.truthbench.models import DecisionRecord, Disposition, GateResult, RunResult


def _passing_run(minimal_fixture) -> GateRun:
    records: list[DecisionRecord] = []
    for cell in minimal_fixture.cells:
        input_value = minimal_fixture.frame.at[cell.row_id, cell.column]
        output = cell.expected_output if cell.disposition is Disposition.REPAIR else None
        records.append(
            DecisionRecord(
                record_id=f"r:{cell.cell_id}",
                run_id="r",
                fixture_id=f"{cell.fixture_version}:{cell.domain}",
                case_id=None,
                cell_id=cell.cell_id,
                domain=cell.domain,
                row_id=cell.row_id,
                column=cell.column,
                surface="cleaning",
                repeat=0,
                expected_disposition=cell.disposition,
                actual_disposition=cell.disposition,
                sensitive=cell.sensitive,
                input=encode_typed(input_value, dtype=minimal_fixture.frame[cell.column].dtype),
                expected_output=cell.expected_output,
                actual_output=output,
                confidence=0.8,
                rationale="deterministic fixture policy applied",
                rule_id="fixture-policy-v1",
                detected=cell.disposition is not Disposition.PRESERVE,
                mutated=cell.disposition is Disposition.REPAIR,
                quarantined=cell.disposition is Disposition.FLAG,
                human_review=cell.disposition is Disposition.REVIEW,
                audit_required=True,
                audit_complete=True,
                audit_ids=(cell.cell_id,),
                trust_before=0.8,
                trust_after=0.8,
                trust_delta=0.0,
                requested_backend="pandas",
                actual_backend="pandas",
                repeat_hash="stable",
                repeat_consistent=True,
            )
        )
    run = RunResult(
        run_id="r",
        profile="release",
        fixture_hashes=((minimal_fixture.domain, minimal_fixture.fixture_hash),),
        required_backends=("pandas",),
        records=tuple(records),
        gates=(),
        summary=(("records", len(records)), ("overall_passed", True)),
        environment=(("python", "test"),),
    )
    return GateRun(run=run, fixtures=(minimal_fixture,), generated_code=("x = 1",), complete=True)


def _replace_records(run: GateRun, records: tuple[DecisionRecord, ...]) -> GateRun:
    return replace(run, run=replace(run.run, records=records))


def _corrupt_preserve(run: GateRun) -> GateRun:
    record = next(
        item
        for item in run.run.records
        if item.expected_disposition is Disposition.PRESERVE and item.column != "name"
    )
    changed = replace(record, actual_output=encode_typed("CORRUPT", dtype="object"), mutated=True)
    return _replace_records(
        run, tuple(changed if item is record else item for item in run.run.records)
    )


def _modify_protected(run: GateRun) -> GateRun:
    record = next(item for item in run.run.records if item.column == "name")
    changed = replace(record, mutated=True, actual_output=record.input)
    return _replace_records(
        run, tuple(changed if item is record else item for item in run.run.records)
    )


def _leak_canary(run: GateRun) -> GateRun:
    return replace(run, persisted_sinks=("error TB-LEAK@example.invalid",))


def _diverge_backend(run: GateRun) -> GateRun:
    record = run.run.records[0]
    changed = replace(record, actual_backend="duckdb")
    return _replace_records(
        run, tuple(changed if item is record else item for item in run.run.records)
    )


def _change_repeat(run: GateRun) -> GateRun:
    record = run.run.records[0]
    changed = replace(record, repeat_hash="changed", repeat_consistent=False)
    return _replace_records(
        run, tuple(changed if item is record else item for item in run.run.records)
    )


def _break_generated_code(run: GateRun) -> GateRun:
    return replace(run, generated_code=("def broken(:",))


def _remove_high_confidence_explanation(run: GateRun) -> GateRun:
    record = run.run.records[0]
    changed = replace(record, confidence=0.95, rationale=None, rule_id=None)
    return _replace_records(
        run, tuple(changed if item is record else item for item in run.run.records)
    )


def _invert_trust(run: GateRun) -> GateRun:
    record = next(
        item for item in run.run.records if item.expected_disposition is Disposition.REPAIR
    )
    changed = replace(record, trust_before=0.2, trust_after=0.9, trust_delta=0.7, mutated=True)
    return _replace_records(
        run, tuple(changed if item is record else item for item in run.run.records)
    )


@pytest.mark.parametrize(
    ("mutator", "gate"),
    [
        (_corrupt_preserve, "valid_value_corruption"),
        (_modify_protected, "protected_column_modification"),
        (_leak_canary, "raw_pii_leakage"),
        (_diverge_backend, "backend_inconsistency"),
        (_change_repeat, "default_nondeterminism"),
        (_break_generated_code, "broken_generated_code"),
        (_remove_high_confidence_explanation, "unexplained_high_confidence"),
        (_invert_trust, "trust_inversion"),
    ],
)
def test_each_mandatory_gate_fails_independently(minimal_fixture, mutator, gate) -> None:
    result = evaluate_gates(mutator(_passing_run(minimal_fixture)))
    assert failed_gate_names(result) == {gate}


def test_gate_evaluation_fails_closed_for_partial_or_invalid_runs(minimal_fixture) -> None:
    run = _passing_run(minimal_fixture)
    result = evaluate_gates(replace(run, complete=False, schema_valid=False))
    assert {"completeness", "schema_validation"} <= failed_gate_names(result)


def test_gate_evaluation_fails_closed_without_fixture_evidence(minimal_fixture) -> None:
    run = _passing_run(minimal_fixture)
    result = evaluate_gates(replace(run, fixtures=()))
    assert "fixture_evidence" in failed_gate_names(result)


def test_gate_results_replace_stale_claims_and_match_summary(minimal_fixture) -> None:
    run = _passing_run(minimal_fixture)
    stale = replace(
        run.run,
        gates=(GateResult("pretend", True),),
        summary=(("records", len(run.run.records)), ("overall_passed", False)),
    )
    result = evaluate_gates(replace(run, run=stale))
    assert not failed_gate_names(result)
    assert all(gate.passed for gate in result.gates)
