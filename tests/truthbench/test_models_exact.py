from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from benchmarks.truthbench import (
    CaseExpectation,
    DecisionRecord,
    Disposition,
    GateResult,
    GoldCell,
    RunResult,
    TypedValue,
    canonical_json,
    encode_typed,
    exact_equal,
    stable_digest,
)


def test_disposition_contract_has_exactly_four_string_values() -> None:
    assert [(item.name, item.value) for item in Disposition] == [
        ("PRESERVE", "preserve"),
        ("REPAIR", "repair"),
        ("FLAG", "flag"),
        ("REVIEW", "review"),
    ]
    assert isinstance(Disposition.PRESERVE, str)


def test_gold_cell_ids_are_stable_and_match_the_public_contract() -> None:
    first = GoldCell.create("v1", "crm", "r7", "notes", "flag", sensitive=True)
    second = GoldCell.create("v1", "crm", "r7", "notes", Disposition.FLAG, sensitive=True)

    assert first == second
    assert first.cell_id == "v1:crm:r7:notes"
    assert GoldCell.create("v1", "crm", "r7", "email", "flag").cell_id != first.cell_id


def test_exact_values_do_not_use_gauntlet_canonicalization() -> None:
    assert not exact_equal("402.10", 402.1)
    assert not exact_equal(" AAPL", "AAPL")
    assert not exact_equal("AAPL", "aapl")
    assert exact_equal(pd.NA, pd.NA)


def test_exact_values_preserve_scalar_type_and_missing_kind() -> None:
    assert not exact_equal(402.1, np.float64(402.1))
    assert not exact_equal(None, pd.NA)
    assert not exact_equal(None, float("nan"))
    assert not exact_equal(pd.NA, float("nan"))
    assert exact_equal(float("nan"), float("nan"))
    assert exact_equal(np.float64(402.1), np.float64(402.1))


def test_exact_values_preserve_unicode_timestamp_and_identifier_representation() -> None:
    assert not exact_equal("Café", "Cafe\u0301")
    assert not exact_equal("00127", 127)
    assert exact_equal("00127", "00127")

    utc = pd.Timestamp("2026-01-15T12:00:00+00:00")
    kolkata = pd.Timestamp("2026-01-15T17:30:00+05:30")
    assert not exact_equal(utc, kolkata)
    assert not exact_equal(utc, datetime(2026, 1, 15, 12, tzinfo=timezone.utc))


def test_dtype_metadata_distinguishes_string_and_categorical_scalars() -> None:
    string_dtype = pd.StringDtype(storage="python")
    category_dtype = pd.CategoricalDtype(categories=["A", "B"], ordered=True)

    assert encode_typed("A", dtype=string_dtype) != encode_typed("A", dtype=category_dtype)
    assert exact_equal("A", "A", left_dtype=string_dtype, right_dtype=string_dtype)
    assert not exact_equal("A", "A", left_dtype=string_dtype, right_dtype=object)


def test_sensitive_record_never_serializes_raw_value() -> None:
    cell = GoldCell.create("v1", "crm", "r7", "notes", "flag", sensitive=True)
    record = DecisionRecord.for_test(
        cell=cell,
        input_value="tb.person+7@example.invalid",
    )

    payload = record.to_dict()

    assert "tb.person+7@example.invalid" not in json.dumps(payload)
    assert payload["input"]["display"] == "[REDACTED]"
    assert payload["input"]["value"] is None
    assert payload["input"]["digest"]


def test_decision_record_serializes_every_normalized_dimension() -> None:
    cell = GoldCell.create("v1", "finance", "r1", "price", "repair", expected_output=402.1)
    payload = DecisionRecord.for_test(cell=cell, input_value="402.10").to_dict()

    required = {
        "expected_disposition",
        "actual_disposition",
        "input",
        "input_type",
        "expected_output",
        "expected_output_type",
        "actual_output",
        "actual_output_type",
        "confidence",
        "rationale",
        "audit_required",
        "audit_complete",
        "audit_ids",
        "trust_before",
        "trust_after",
        "trust_delta",
        "requested_backend",
        "actual_backend",
        "fallback_events",
        "repeat_hash",
        "repeat_consistent",
    }
    assert required <= payload.keys()
    populated = {
        "expected_disposition",
        "input",
        "input_type",
        "expected_output",
        "expected_output_type",
    }
    assert all(payload[key] is None for key in required if key not in populated)


def test_oracle_and_result_models_are_frozen_and_json_safe() -> None:
    cell = GoldCell.create("v1", "crm", "r7", "notes", "review")
    case = CaseExpectation.create("v1", "crm", "row", "duplicate-r7", "review")
    record = DecisionRecord.for_test(cell=cell, input_value="ordinary text")
    gate = GateResult(name="valid-value-corruption", passed=True)
    run = RunResult(
        run_id="run-1",
        profile="release",
        fixture_hashes=(("crm", "abc123"),),
        required_backends=("pandas",),
        records=(record,),
        gates=(gate,),
        summary=(("overall_passed", True), ("records", 1)),
        environment=(("python", "3.9+"),),
    )

    for value in (encode_typed("x"), cell, case, record, gate, run):
        with pytest.raises(FrozenInstanceError):
            value.schema_version = 2  # type: ignore[misc]

    payload = run.to_dict()
    assert json.loads(canonical_json(payload)) == payload
    serialized = (
        cell.to_dict(),
        case.to_dict(),
        record.to_dict(),
        gate.to_dict(),
        payload,
    )
    assert all(item["schema_version"] == 1 for item in serialized)


def test_canonical_json_is_stable_and_rejects_unsafe_values() -> None:
    assert canonical_json({"b": 2, "a": [True, None]}) == '{"a":[true,null],"b":2}'
    assert stable_digest("AAPL", key=b"run-one") == stable_digest("AAPL", key=b"run-one")
    assert stable_digest("AAPL", key=b"run-one") != stable_digest("AAPL", key=b"run-two")

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            canonical_json({"unsafe": value})
    with pytest.raises(TypeError, match="unsupported"):
        canonical_json({"unsafe": object()})


def test_typed_values_round_trip_without_non_finite_json_numbers() -> None:
    values = [
        "402.10",
        402.1,
        np.float64(402.1),
        None,
        pd.NA,
        float("nan"),
        "Café",
        "Cafe\u0301",
        pd.Timestamp("2026-01-15T12:00:00+00:00"),
        "00127",
    ]

    for value in values:
        typed = encode_typed(value)
        payload = typed.to_dict()
        assert TypedValue.from_dict(json.loads(canonical_json(payload))) == typed
