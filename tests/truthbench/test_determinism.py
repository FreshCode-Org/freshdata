"""Normalized decision hashing and repeat-consistency contracts."""

from __future__ import annotations

import dataclasses

import pytest
from benchmarks.truthbench.determinism import (
    APPROVED_TELEMETRY,
    annotate_repeats,
    decision_hash,
)
from benchmarks.truthbench.exact import encode_typed
from benchmarks.truthbench.models import DecisionRecord, Disposition


def _record(repeat: int = 0, *, confidence: float | None = 0.9, run_id: str = "r1"):
    return DecisionRecord(
        record_id=f"{run_id}:cleaning:pandas:{repeat}:v1:demo:row-1:amount",
        run_id=run_id,
        fixture_id="v1:demo",
        case_id=None,
        cell_id="v1:demo:row-1:amount",
        domain="demo",
        row_id="row-1",
        column="amount",
        surface="cleaning",
        repeat=repeat,
        expected_disposition=Disposition.REPAIR,
        actual_disposition=Disposition.REPAIR,
        sensitive=False,
        input=encode_typed("1,5"),
        expected_output=encode_typed(1.5),
        actual_output=encode_typed(1.5),
        confidence=confidence,
        requested_backend="pandas",
        actual_backend="pandas",
    )


def test_hash_ignores_only_approved_telemetry():
    left = _record(repeat=0, run_id="run-a")
    right = _record(repeat=1, run_id="run-b")
    assert decision_hash(left) == decision_hash(right)


def test_hash_covers_decision_bearing_fields():
    base = _record()
    for change in (
        {"confidence": 0.5},
        {"actual_output": encode_typed(2.5)},
        {"actual_disposition": Disposition.PRESERVE},
        {"rationale": "different"},
        {"trust_after": 0.1},
        {"actual_backend": "polars"},
        {"audit_ids": ("x",)},
    ):
        changed = dataclasses.replace(base, **change)
        assert decision_hash(changed) != decision_hash(base), change


def test_decision_bearing_fields_cannot_be_excluded():
    record = _record()
    with pytest.raises(ValueError, match="cannot be excluded"):
        decision_hash(record, exclude={"confidence"})
    with pytest.raises(ValueError, match="cannot be excluded"):
        decision_hash(record, exclude={*APPROVED_TELEMETRY, "actual_output"})


def test_annotate_repeats_marks_consistent_groups():
    records = annotate_repeats(
        [_record(repeat=0), _record(repeat=1)], expected_repeats=(0, 1)
    )
    assert all(r.repeat_consistent is True for r in records)
    assert len({r.repeat_hash for r in records}) == 1
    assert all(r.normalized_decision_hash for r in records)


def test_annotate_repeats_fails_closed_on_difference_and_missing_repeat():
    differing = annotate_repeats(
        [_record(repeat=0), dataclasses.replace(_record(repeat=1), confidence=0.2)],
        expected_repeats=(0, 1),
    )
    assert all(r.repeat_consistent is False for r in differing)

    missing = annotate_repeats([_record(repeat=0)], expected_repeats=(0, 1))
    assert all(r.repeat_consistent is False for r in missing)


def test_annotate_repeats_requires_two_repeats():
    with pytest.raises(ValueError, match="two repeats"):
        annotate_repeats([_record()], expected_repeats=(0,))
