from __future__ import annotations

from dataclasses import replace

import pandas as pd
from benchmarks.truthbench.exact import encode_typed
from benchmarks.truthbench.fixtures import build_fixture
from benchmarks.truthbench.fixtures.base import FixtureBuilder
from benchmarks.truthbench.models import CaseExpectation, Disposition
from benchmarks.truthbench.normalize import normalize_observation
from benchmarks.truthbench.surfaces.base import SurfaceObservation
from benchmarks.truthbench.surfaces.privacy import PrivacyAdapter
from benchmarks.truthbench.surfaces.validation import ValidationAdapter


def test_normalize_records_exact_outputs_audit_and_backend(minimal_fixture) -> None:
    output = minimal_fixture.frame.copy(deep=True)
    output.loc["r2", "amount"] = 2.5
    repair = next(cell for cell in minimal_fixture.cells if cell.disposition is Disposition.REPAIR)
    observation = SurfaceObservation(
        output_frame=output,
        raw_decisions={
            repair.cell_id: {
                "detected": True,
                "mutated": True,
                "confidence": 0.95,
                "rationale": "numeric coercion from declared policy",
                "rule_id": "numeric-format-v1",
            }
        },
        audit_sinks={"audit": {"record_ids": [repair.cell_id]}},
        trust={"trust_before": 0.4, "trust_after": 0.6},
        backend_disclosure={"requested": "pandas", "actual": "pandas"},
    )

    result = normalize_observation(
        minimal_fixture,
        observation,
        surface="cleaning",
        backend="pandas",
        repeat=0,
        run_id="run-1",
    )

    assert len(result.records) == len(minimal_fixture.cells)
    record = next(item for item in result.records if item.cell_id == repair.cell_id)
    assert record.actual_disposition is Disposition.REPAIR
    assert record.actual_output == encode_typed(2.5, dtype=output["amount"].dtype)
    assert record.detected is True and record.mutated is True
    assert record.audit_ids == (repair.cell_id,) and record.audit_complete is True
    assert record.confidence == 0.95
    assert record.rationale == "numeric coercion from declared policy"
    assert record.rule_id == "numeric-format-v1"
    assert record.trust_delta == 0.2
    assert record.requested_backend == record.actual_backend == "pandas"


def test_normalize_routes_flags_and_reviews_without_mutation() -> None:
    frame = pd.DataFrame({"value": ["uncertain", "invalid"]}, index=["flag", "review"])
    builder = FixtureBuilder("v1", "routing", frame)
    builder.inject("flag", "value", "uncertain", "flag", family="ambiguous")
    builder.inject("review", "value", "invalid", "review", family="policy-conflict")
    fixture = builder.build()
    observation = SurfaceObservation(
        output_frame=frame.copy(deep=True),
        raw_decisions={
            "v1:routing:flag:value": {"detected": True, "quarantined": True},
            "v1:routing:review:value": {"detected": True, "human_review": True},
        },
    )

    result = normalize_observation(
        fixture, observation, surface="validation", backend="pandas", repeat=0, run_id="r"
    )

    flag, review = result.records
    assert flag.actual_disposition is Disposition.FLAG
    assert flag.mutated is False and flag.quarantined is True
    assert review.actual_disposition is Disposition.REVIEW
    assert review.mutated is False and review.human_review is True


def test_normalize_sensitive_values_and_non_applicable_output(minimal_fixture) -> None:
    sensitive = replace(
        next(cell for cell in minimal_fixture.cells if cell.column == "name"),
        sensitive=True,
        canary_id="minimal-name",
    )
    fixture = replace(
        minimal_fixture,
        cells=tuple(
            sensitive if cell.cell_id == sensitive.cell_id else cell
            for cell in minimal_fixture.cells
        ),
        pii_canaries={"minimal-name": "TB-ALPHA"},
    )
    # The normalizer must neither expose the raw value nor pretend a validator
    # produced an output that its public surface does not expose.
    result = normalize_observation(
        fixture,
        SurfaceObservation(
            output_frame=None, raw_decisions={sensitive.cell_id: {"detected": True}}
        ),
        surface="privacy",
        backend="pandas",
        repeat=0,
        run_id="r",
    )
    record = next(item for item in result.records if item.cell_id == sensitive.cell_id)
    assert record.input.redacted and record.input.value is None
    assert record.actual_output is None
    assert record.detected is True and record.mutated is False


def test_normalize_fails_closed_on_missing_backend_disclosure_and_tracks_case_evidence(
    minimal_fixture,
) -> None:
    case = CaseExpectation.create("v1", "minimal", "row", "row-policy", "flag")
    fixture = replace(minimal_fixture, row_cases=(case,))
    result = normalize_observation(
        fixture,
        SurfaceObservation(raw_decisions={case.case_id: {"observed": True}}),
        surface="validation",
        backend="pandas",
        repeat=0,
        run_id="r",
    )
    assert {record.requested_backend for record in result.records} == {None}
    assert {record.actual_backend for record in result.records} == {None}
    assert len(result.cases) == 1
    assert result.cases[0].case_id == case.case_id and result.cases[0].observed is True


def test_real_validation_and_privacy_adapters_do_not_receive_synthetic_mutation_credit(
    minimal_fixture,
) -> None:
    validation = ValidationAdapter().observe(minimal_fixture, {"operation": "validate"})
    normalized_validation = normalize_observation(
        minimal_fixture,
        validation,
        surface="validation",
        backend="pandas",
        repeat=0,
        run_id="validation",
    )
    assert all(record.mutated is False for record in normalized_validation.records)

    pii_fixture = build_fixture("crm")
    privacy = PrivacyAdapter().observe(pii_fixture, {"operation": "detect_pii"})
    normalized_privacy = normalize_observation(
        pii_fixture,
        privacy,
        surface="privacy",
        backend="pandas",
        repeat=0,
        run_id="privacy",
    )
    assert all(record.input.redacted for record in normalized_privacy.records if record.sensitive)


def test_type_normalization_equivalence_is_narrow():
    from benchmarks.truthbench.exact import (  # noqa: PLC0415
        equivalent_after_type_normalization as equiv,
    )

    ts = encode_typed(pd.Timestamp("2026-01-15"))
    assert equiv(ts, encode_typed("2026-01-15"))          # canonical ISO
    assert not equiv(ts, encode_typed("01/02/2025"))      # ambiguous form
    assert not equiv(ts, encode_typed("2026-01"))         # partial date
    tz = encode_typed(pd.Timestamp("2026-01-15", tz="UTC"))
    assert not equiv(tz, encode_typed("2026-01-15"))      # timezone change
    assert equiv(encode_typed(2025), encode_typed("2025"))
    assert equiv(encode_typed(10.5), encode_typed("10.50"))
    assert not equiv(encode_typed(7), encode_typed("007"))   # leading zero
    assert not equiv(encode_typed(True), encode_typed("yes"))  # vocabulary
    assert not equiv(encode_typed(1234.56), encode_typed("$1,234.56"))
