from __future__ import annotations

from dataclasses import replace

import pytest
from benchmarks.truthbench.exact import encode_typed
from benchmarks.truthbench.gates import GateRun, evaluate_gates, failed_gate_names
from benchmarks.truthbench.models import (
    CaseExpectation,
    DecisionRecord,
    Disposition,
    GateResult,
    RunResult,
)
from benchmarks.truthbench.normalize import CaseRecord


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
    return GateRun(
        run=run,
        fixtures=(minimal_fixture,),
        generated_code=("x = 1",),
        complete=True,
        expected_surfaces=("cleaning",),
        expected_backends=("pandas",),
        expected_repeats=(0,),
        surface_classes=(("cleaning", "mutator"),),
        audit_record_ids=tuple(record.cell_id for record in records),
    )


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


def test_gate_evaluation_fails_closed_for_missing_cell_surface_backend_repeat_coverage(
    minimal_fixture,
) -> None:
    passing = _passing_run(minimal_fixture)
    one = passing.run.records[:1]
    partial = _replace_records(
        passing,
        one,
    )
    partial = replace(
        partial,
        run=replace(partial.run, summary=(("records", 1), ("overall_passed", True))),
    )
    assert "completeness" in failed_gate_names(evaluate_gates(partial))


def test_gate_evaluation_fails_closed_when_a_declared_repeat_is_unexecuted(
    minimal_fixture,
) -> None:
    passing = replace(_passing_run(minimal_fixture), expected_repeats=(0, 1))
    assert "completeness" in failed_gate_names(evaluate_gates(passing))


def test_validator_receives_detection_credit_without_mutation(minimal_fixture) -> None:
    passing = _passing_run(minimal_fixture)
    records = []
    for record in passing.run.records:
        changed = replace(
            record,
            surface="validation",
            actual_output=None,
            mutated=False,
            detected=record.expected_disposition is not Disposition.PRESERVE,
        )
        records.append(changed)
    validator = _replace_records(passing, tuple(records))
    validator = replace(
        validator,
        expected_surfaces=("validation",),
        surface_classes=(("validation", "validator"),),
    )
    assert not failed_gate_names(evaluate_gates(validator))


def test_pii_surface_ignores_non_pii_repair_oracle_but_rejects_false_positive_mutation(
    minimal_fixture,
) -> None:
    passing = _passing_run(minimal_fixture)
    repair = next(r for r in passing.run.records if r.expected_disposition is Disposition.REPAIR)
    wrong = replace(repair, surface="privacy", actual_output=repair.input, mutated=False)
    records = tuple(
        replace(record, surface="privacy", detected=False)
        if record is not repair
        else replace(wrong, detected=False)
        for record in passing.run.records
    )
    pii = replace(
        _replace_records(passing, records),
        expected_surfaces=("privacy",),
        surface_classes=(("privacy", "pii"),),
    )
    assert not failed_gate_names(evaluate_gates(pii))
    pii_repair = next(record for record in pii.run.records if record.cell_id == repair.cell_id)
    false_positive = replace(pii_repair, mutated=True)
    bad = _replace_records(
        pii,
        tuple(false_positive if record is pii_repair else record for record in pii.run.records),
    )
    assert "pii_scope" in failed_gate_names(evaluate_gates(bad))


def test_explicit_transform_mutates_only_requested_cells(minimal_fixture) -> None:
    passing = _passing_run(minimal_fixture)
    preserve = next(
        r for r in passing.run.records if r.expected_disposition is Disposition.PRESERVE
    )
    changed = replace(preserve, surface="plan", actual_output=preserve.input, mutated=True)
    transform = _replace_records(
        passing,
        tuple(
            changed if record is preserve else replace(record, surface="plan")
            for record in passing.run.records
        ),
    )
    transform = replace(
        transform,
        expected_surfaces=("plan",),
        surface_classes=(("plan", "explicit_transform"),),
        requested_cell_ids=(
            next(
                r.cell_id
                for r in transform.run.records
                if r.expected_disposition is Disposition.REPAIR
            ),
        ),
    )
    assert "requested_behavior" in failed_gate_names(evaluate_gates(transform))


def test_protected_and_trust_gates_use_actual_evidence_not_mutated_claim(minimal_fixture) -> None:
    passing = _passing_run(minimal_fixture)
    protected = next(r for r in passing.run.records if r.column == "name")
    changed = replace(
        protected, actual_output=encode_typed("changed", dtype="object"), mutated=False
    )
    run = _replace_records(
        passing, tuple(changed if r is protected else r for r in passing.run.records)
    )
    assert "protected_column_modification" in failed_gate_names(evaluate_gates(run))
    repair = next(r for r in passing.run.records if r.expected_disposition is Disposition.REPAIR)
    trust_changed = replace(
        repair, mutated=False, trust_before=0.2, trust_after=0.9, trust_delta=0.7
    )
    trust = _replace_records(
        passing, tuple(trust_changed if r is repair else r for r in passing.run.records)
    )
    assert "trust_inversion" in failed_gate_names(evaluate_gates(trust))


def test_missing_disclosure_repeat_and_actual_audit_evidence_fail_closed(minimal_fixture) -> None:
    passing = _passing_run(minimal_fixture)
    record = passing.run.records[0]
    disclosure = _replace_records(
        passing,
        tuple(
            replace(record, requested_backend=None) if r is record else r
            for r in passing.run.records
        ),
    )
    assert "backend_inconsistency" in failed_gate_names(evaluate_gates(disclosure))
    no_repeat = _replace_records(
        passing,
        tuple(
            replace(record, repeat_consistent=None, repeat_hash=None) if r is record else r
            for r in passing.run.records
        ),
    )
    assert "default_nondeterminism" in failed_gate_names(evaluate_gates(no_repeat))
    high = _replace_records(
        passing,
        tuple(replace(record, confidence=0.95) if r is record else r for r in passing.run.records),
    )
    assert "unexplained_high_confidence" in failed_gate_names(
        evaluate_gates(replace(high, audit_record_ids=()))
    )


def test_objectively_changed_mutator_requires_audit_even_when_mutation_is_denied(
    minimal_fixture,
) -> None:
    passing = _passing_run(minimal_fixture)
    repair = next(
        record
        for record in passing.run.records
        if record.expected_disposition is Disposition.REPAIR
    )
    unaudited = replace(
        repair, mutated=False, detected=False, audit_complete=False, audit_ids=None
    )
    run = _replace_records(
        passing,
        tuple(unaudited if record is repair else record for record in passing.run.records),
    )
    assert "mutation_audit" in failed_gate_names(evaluate_gates(run))


def test_exact_repair_ignores_unspecified_dtype_but_preserves_value_type(minimal_fixture) -> None:
    passing = _passing_run(minimal_fixture)
    repair = next(r for r in passing.run.records if r.expected_disposition is Disposition.REPAIR)
    object_typed = replace(repair.actual_output, dtype="object")
    dtype_only = _replace_records(
        passing,
        tuple(
            replace(repair, actual_output=object_typed) if r is repair else r
            for r in passing.run.records
        ),
    )
    assert "exact_repair" not in failed_gate_names(evaluate_gates(dtype_only))


def test_required_cases_fail_closed_when_no_case_record_is_observed(minimal_fixture) -> None:
    fixture = replace(
        minimal_fixture,
        row_cases=(CaseExpectation.create("v1", "minimal", "row", "required", "flag"),),
    )
    passing = _passing_run(minimal_fixture)
    context = replace(passing, fixtures=(fixture,), case_records=())
    assert "case_coverage" in failed_gate_names(evaluate_gates(context))
    observed = replace(
        context,
        case_records=(CaseRecord("v1:minimal:row:required", "row", Disposition.FLAG, True),),
    )
    assert "case_coverage" not in failed_gate_names(evaluate_gates(observed))


def test_gate_run_uses_fixture_snapshot_not_mutable_frame_after_construction(
    minimal_fixture,
) -> None:
    passing = _passing_run(minimal_fixture)
    minimal_fixture.frame.loc["r1", "name"] = "tampered-after-snapshot"
    assert "input_mutation" not in failed_gate_names(evaluate_gates(passing))


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
