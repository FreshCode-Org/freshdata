"""Normalize heterogeneous public-surface evidence into TruthBench records.

This module deliberately has no cleaning policy.  It only captures what a
public surface did, preserves typed values, and leaves pass/fail decisions to
``gates``.  Missing evidence therefore remains missing rather than being
silently inferred as a successful repair.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .exact import encode_typed, exact_equal
from .models import DecisionRecord, Disposition
from .surfaces.base import SurfaceObservation

_SENSITIVE_KEY = b"truthbench-normalized-sensitive-v1"


@dataclass(frozen=True)
class CaseRecord:
    """Non-serialized row/schema evidence retained for gate context.

    ``RunResult`` intentionally serializes only ``DecisionRecord``.  Cases are
    contextual evidence for the runner and release gates, never an alternative
    public result schema.
    """

    case_id: str
    kind: str
    expected_disposition: Disposition
    observed: bool


@dataclass(frozen=True)
class NormalizationResult:
    records: tuple[DecisionRecord, ...]
    cases: tuple[CaseRecord, ...]
    input_mutated: bool = False


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _decision_for(raw: Any, cell_id: str) -> Mapping[str, Any]:
    """Extract one cell's public decision without assuming a report type."""

    source = _mapping(raw)
    direct = source.get(cell_id)
    if isinstance(direct, Mapping):
        return direct
    for name in ("decisions", "records", "actions", "findings"):
        values = source.get(name)
        if isinstance(values, Mapping) and isinstance(values.get(cell_id), Mapping):
            return values[cell_id]
        if isinstance(values, (list, tuple)):
            for item in values:
                if not isinstance(item, Mapping):
                    continue
                if item.get("cell_id") == cell_id or item.get("id") == cell_id:
                    return item
    return {}


def _audit_ids(sinks: Any, cell_id: str) -> tuple[str, ...]:
    """Collect matching IDs from public audit payloads, never their contents."""

    found: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"record_id", "cell_id", "id"} and item == cell_id:
                    found.append(cell_id)
                if (
                    key in {"record_ids", "cell_ids", "ids"}
                    and isinstance(item, (list, tuple))
                    and cell_id in item
                ):
                    found.append(cell_id)
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(sinks)
    return (cell_id,) if found else ()


def _bool(decision: Mapping[str, Any], name: str, default: bool | None = None) -> bool | None:
    value = decision.get(name, default)
    return value if isinstance(value, bool) else default


def _disposition(value: Any) -> Disposition | None:
    if value is None:
        return None
    try:
        return Disposition(value)
    except ValueError:
        return None


def _output_value(
    fixture: Any, cell: Any, observation: SurfaceObservation
) -> tuple[Any, Any] | None:
    frame = observation.output_frame
    if not isinstance(frame, pd.DataFrame):
        return None
    if cell.row_id not in frame.index or cell.column not in frame.columns:
        return None
    return frame.at[cell.row_id, cell.column], frame[cell.column].dtype


def normalize_observation(
    fixture: Any,
    observation: SurfaceObservation,
    *,
    surface: str,
    backend: str,
    repeat: int,
    run_id: str,
) -> NormalizationResult:
    """Produce one record for every labelled cell in ``fixture``.

    Validators may detect bad values without returning a frame.  In that case
    ``actual_output`` remains ``None`` and ``mutated`` remains false; gates can
    give detection credit without inventing a mutation.
    """

    frame = getattr(fixture, "frame", None)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("normalization requires a pandas TruthFixture frame")
    raw = observation.raw_decisions
    trust = _mapping(observation.trust)
    disclosure = _mapping(observation.backend_disclosure)
    records: list[DecisionRecord] = []
    for cell in fixture.cells:
        decision = _decision_for(raw, cell.cell_id)
        source_value = frame.at[cell.row_id, cell.column]
        source_dtype = frame[cell.column].dtype
        output = _output_value(fixture, cell, observation)
        actual_value, actual_dtype = output if output is not None else (None, None)
        changed = output is not None and not exact_equal(
            actual_value, source_value, left_dtype=actual_dtype, right_dtype=source_dtype
        )
        explicit_mutated = _bool(decision, "mutated")
        mutated = explicit_mutated if explicit_mutated is not None else changed
        detected = _bool(decision, "detected", False)
        quarantined = _bool(decision, "quarantined", False)
        human_review = _bool(decision, "human_review", False)
        actual_disposition = _disposition(decision.get("disposition"))
        if actual_disposition is None:
            if cell.disposition is Disposition.FLAG and (detected or quarantined):
                actual_disposition = Disposition.FLAG
            elif cell.disposition is Disposition.REVIEW and (detected or human_review):
                actual_disposition = Disposition.REVIEW
            elif cell.disposition is Disposition.REPAIR and (mutated or output is not None):
                actual_disposition = Disposition.REPAIR
            elif not mutated:
                actual_disposition = Disposition.PRESERVE
        audit_ids = _audit_ids(observation.audit_sinks, cell.cell_id)
        audit_required = (
            cell.disposition is not Disposition.PRESERVE or bool(mutated) or bool(detected)
        )
        confidence = decision.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = None
        trust_before = trust.get("trust_before")
        trust_after = trust.get("trust_after")
        if not isinstance(trust_before, (int, float)) or isinstance(trust_before, bool):
            trust_before = None
        if not isinstance(trust_after, (int, float)) or isinstance(trust_after, bool):
            trust_after = None
        records.append(
            DecisionRecord(
                record_id=f"{run_id}:{surface}:{backend}:{repeat}:{cell.cell_id}",
                run_id=run_id,
                fixture_id=f"{cell.fixture_version}:{cell.domain}",
                case_id=None,
                cell_id=cell.cell_id,
                domain=cell.domain,
                row_id=cell.row_id,
                column=cell.column,
                surface=surface,
                repeat=repeat,
                expected_disposition=cell.disposition,
                actual_disposition=actual_disposition,
                sensitive=cell.sensitive,
                input=encode_typed(
                    source_value,
                    dtype=source_dtype,
                    sensitive=cell.sensitive,
                    digest_key=_SENSITIVE_KEY if cell.sensitive else None,
                ),
                expected_output=cell.expected_output,
                actual_output=(
                    None
                    if output is None
                    else encode_typed(
                        actual_value,
                        dtype=actual_dtype,
                        sensitive=cell.sensitive,
                        digest_key=_SENSITIVE_KEY if cell.sensitive else None,
                    )
                ),
                confidence=float(confidence) if confidence is not None else None,
                risk=decision.get("risk") if isinstance(decision.get("risk"), str) else None,
                status=decision.get("status") if isinstance(decision.get("status"), str) else None,
                rule_id=(
                    decision.get("rule_id")
                    if isinstance(decision.get("rule_id"), str)
                    else decision.get("model_id")
                    if isinstance(decision.get("model_id"), str)
                    else None
                ),
                rationale=decision.get("rationale")
                if isinstance(decision.get("rationale"), str)
                else None,
                evidence_kinds=tuple(
                    str(item)
                    for item in decision.get("evidence_kinds", ())
                    if isinstance(item, str)
                )
                or None,
                mutated=mutated,
                detected=detected,
                quarantined=quarantined,
                human_review=human_review,
                audit_required=audit_required,
                audit_complete=(bool(audit_ids) if audit_required else True),
                audit_ids=audit_ids or None,
                trust_before=float(trust_before) if trust_before is not None else None,
                trust_after=float(trust_after) if trust_after is not None else None,
                trust_delta=(
                    round(float(trust_after - trust_before), 12)
                    if trust_before is not None and trust_after is not None
                    else None
                ),
                requested_backend=(
                    disclosure.get("requested")
                    if isinstance(disclosure.get("requested"), str)
                    else backend
                ),
                actual_backend=(
                    disclosure.get("actual")
                    if isinstance(disclosure.get("actual"), str)
                    else backend
                ),
            )
        )
    cases = tuple(
        CaseRecord(case.case_id, case.kind, case.disposition, False)
        for case in (*getattr(fixture, "row_cases", ()), *getattr(fixture, "schema_cases", ()))
    )
    return NormalizationResult(tuple(records), cases)


__all__ = ["CaseRecord", "NormalizationResult", "normalize_observation"]
