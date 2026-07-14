from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any

import pytest
from benchmarks.truthbench.models import (
    DecisionRecord,
    GateResult,
    GoldCell,
    RunResult,
)
from benchmarks.truthbench.schema import TruthBenchSchemaError, validate_run


def _valid_payload() -> dict[str, Any]:
    cell = GoldCell.create("v1", "crm", "r7", "notes", "review")
    record = DecisionRecord.for_test(cell=cell, input_value="ordinary text")
    run = RunResult(
        run_id="run-1",
        profile="release",
        fixture_hashes=(("crm", "abc123"),),
        required_backends=("pandas",),
        records=(record,),
        gates=(GateResult(name="valid-value-corruption", passed=True),),
        summary=(("overall_passed", True), ("records", 1)),
        environment=(("python", "3.9+"),),
    )
    return run.to_dict()


def test_valid_serialized_run_passes_schema_and_integrity_validation() -> None:
    validate_run(_valid_payload())


def test_validator_accepts_the_declared_mapping_interface() -> None:
    validate_run(MappingProxyType(_valid_payload()))


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("records", 0),
        ("gates", 0),
        ("records", 0, "input"),
    ],
)
def test_unknown_schema_versions_are_rejected(path: tuple[str | int, ...]) -> None:
    payload = _valid_payload()
    target: Any = payload
    for component in path:
        target = target[component]
    target["schema_version"] = 2

    with pytest.raises(TruthBenchSchemaError, match="schema"):
        validate_run(payload)


@pytest.mark.parametrize("field", ["fixture_hashes", "required_backends", "gates"])
def test_required_run_components_cannot_be_absent_or_empty(field: str) -> None:
    for missing in (True, False):
        payload = _valid_payload()
        if missing:
            del payload[field]
        else:
            payload[field] = {} if field == "fixture_hashes" else []

        with pytest.raises(TruthBenchSchemaError):
            validate_run(payload)


def test_run_without_decision_records_is_partial_and_rejected() -> None:
    payload = _valid_payload()
    payload["records"] = []
    payload["summary"]["records"] = 0

    with pytest.raises(TruthBenchSchemaError):
        validate_run(payload)


def test_duplicate_record_ids_are_rejected_precisely() -> None:
    payload = _valid_payload()
    payload["records"].append(deepcopy(payload["records"][0]))
    payload["summary"]["records"] = 2

    with pytest.raises(TruthBenchSchemaError, match="duplicate decision record id"):
        validate_run(payload)


@pytest.mark.parametrize(
    "field",
    ["confidence", "trust_before", "trust_after", "trust_delta"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_record_scores_are_rejected(field: str, value: float) -> None:
    payload = _valid_payload()
    payload["records"][0][field] = value

    with pytest.raises(TruthBenchSchemaError, match="finite"):
        validate_run(payload)


def test_each_record_must_have_a_matching_fixture_hash() -> None:
    payload = _valid_payload()
    payload["fixture_hashes"] = {"finance": "def456"}

    with pytest.raises(TruthBenchSchemaError, match="fixture hash"):
        validate_run(payload)


def test_record_aggregate_must_match_record_count() -> None:
    payload = _valid_payload()
    payload["summary"]["records"] = 9

    with pytest.raises(
        TruthBenchSchemaError,
        match="record aggregate does not match records",
    ):
        validate_run(payload)


def test_gate_failure_aggregate_must_match_failures() -> None:
    payload = _valid_payload()
    payload["gates"][0]["failure_count"] = 1

    with pytest.raises(TruthBenchSchemaError, match="gate failure aggregate"):
        validate_run(payload)


def test_overall_passed_claim_cannot_hide_a_failed_gate() -> None:
    payload = _valid_payload()
    payload["gates"][0] = GateResult(
        name="valid-value-corruption",
        passed=False,
        failures=("record changed",),
    ).to_dict()

    with pytest.raises(TruthBenchSchemaError, match="overall gate claim is inconsistent"):
        validate_run(payload)


@pytest.mark.parametrize(
    ("path", "extra"),
    [
        ((), "unexpected_result"),
        (("records", 0), "unexpected_record"),
        (("gates", 0), "unexpected_gate"),
        (("environment",), "unexpected_environment"),
    ],
)
def test_closed_schema_levels_reject_unknown_properties(
    path: tuple[str | int, ...],
    extra: str,
) -> None:
    payload = _valid_payload()
    target: Any = payload
    for component in path:
        target = target[component]
    target[extra] = "not allowed"

    with pytest.raises(TruthBenchSchemaError, match="Additional properties"):
        validate_run(payload)


def test_gate_failures_must_be_non_empty_strings() -> None:
    payload = _valid_payload()
    payload["gates"][0]["failures"] = [{"message": "not the frozen contract"}]
    payload["gates"][0]["failure_count"] = 1

    with pytest.raises(TruthBenchSchemaError):
        validate_run(payload)
