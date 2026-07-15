"""Deterministic reduction of reproduced TruthBench failures.

The minimizer removes background rows and columns while the failure's target
cell keeps failing, under a bounded evaluation budget.  The target cell and
its gold disposition are immutable: reduction that "fixes" the failure by
deleting the evidence is rejected by construction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .exact import canonical_json, stable_digest
from .privacy import SinkScanner

_HASH_KEY = b"truthbench-minimize-v1"

#: Maximum candidate frames evaluated per failure.
DEFAULT_BUDGET = 60


@dataclass(frozen=True)
class FailureCase:
    """A sanitized, reproducible minimal failure."""

    failure_id: str
    cell_id: str
    domain: str
    surface: str
    gate: str
    expected: Any
    actual: Any
    component: str
    frame_records: tuple[Mapping[str, Any], ...]
    schema: Mapping[str, Any]
    policy: Mapping[str, Any]
    reproduce_command: str
    evidence_hash: str
    evaluations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "cell_id": self.cell_id,
            "domain": self.domain,
            "surface": self.surface,
            "gate": self.gate,
            "expected": self.expected,
            "actual": self.actual,
            "component": self.component,
            "frame_records": [dict(row) for row in self.frame_records],
            "schema": dict(self.schema),
            "policy": dict(self.policy),
            "reproduce_command": self.reproduce_command,
            "evidence_hash": self.evidence_hash,
            "evaluations": self.evaluations,
        }


def _sanitize(frame: pd.DataFrame, fixture: Any) -> pd.DataFrame:
    canaries = dict(getattr(fixture, "pii_canaries", {}) or {})
    if not canaries:
        return frame
    scanner = SinkScanner.from_canaries(canaries, key=_HASH_KEY)
    return scanner.redact(frame)


def minimize_failure(
    fixture: Any,
    *,
    cell_id: str,
    gate: str,
    surface: str,
    expected: Any,
    actual: Any,
    component: str,
    still_fails: Callable[[pd.DataFrame], bool],
    budget: int = DEFAULT_BUDGET,
) -> FailureCase:
    """Reduce ``fixture.frame`` while ``still_fails`` holds for the target cell.

    ``still_fails`` receives a candidate frame (always containing the target
    cell's row and column) and must re-run the public surface to decide
    whether the original failure reproduces.  Reduction is greedy and
    deterministic: first drop row blocks, then single rows, then columns.
    """

    cell = next((c for c in fixture.cells if c.cell_id == cell_id), None)
    if cell is None:
        raise ValueError(f"unknown target cell: {cell_id}")

    frame = fixture.frame.copy(deep=True)
    evaluations = 0

    def attempt(candidate: pd.DataFrame) -> bool:
        nonlocal evaluations
        if evaluations >= budget:
            return False
        if cell.row_id not in candidate.index or cell.column not in candidate.columns:
            return False  # never remove the target cell
        evaluations += 1
        try:
            return bool(still_fails(candidate.copy(deep=True)))
        except Exception:
            return False

    # Pass 1: halve away row blocks (ddmin-style), keeping the target row.
    chunk = max(1, len(frame) // 2)
    while chunk >= 1:
        index = list(frame.index)
        position = 0
        while position < len(index):
            block = [
                row
                for row in index[position : position + chunk]
                if row != cell.row_id
            ]
            if block:
                candidate = frame.drop(index=block)
                if attempt(candidate):
                    frame = candidate
                    index = list(frame.index)
                    continue
            position += chunk
        if chunk == 1:
            break
        chunk //= 2

    # Pass 2: drop background columns one at a time.
    for column in list(frame.columns):
        if column == cell.column:
            continue
        candidate = frame.drop(columns=[column])
        if attempt(candidate):
            frame = candidate

    safe = _sanitize(frame, fixture)
    records = tuple(
        {"row_id": str(row), **{str(c): _cell_payload(safe.at[row, c]) for c in safe.columns}}
        for row in safe.index
    )
    evidence = {
        "cell_id": cell_id,
        "gate": gate,
        "surface": surface,
        "expected": expected,
        "actual": actual,
        "records": records,
    }
    evidence_hash = stable_digest(canonical_json(evidence), key=_HASH_KEY)
    failure_id = "tbf-" + hashlib.sha256(
        canonical_json(
            {"cell": cell_id, "gate": gate, "surface": surface}
        ).encode("utf-8")
    ).hexdigest()[:16]
    return FailureCase(
        failure_id=failure_id,
        cell_id=cell_id,
        domain=str(fixture.domain),
        surface=surface,
        gate=gate,
        expected=expected,
        actual=actual,
        component=component,
        frame_records=records,
        schema=dict(getattr(fixture, "schema", {}) or {}),
        policy=dict(getattr(fixture, "policy", {}) or {}),
        reproduce_command=(
            "PYTHONPATH=src python -m benchmarks.truthbench reproduce "
            f"--failure-id {failure_id} --domain {fixture.domain} "
            f"--cell-id '{cell_id}' --surface {surface}"
        ),
        evidence_hash=evidence_hash,
        evaluations=evaluations,
    )


def _cell_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


__all__ = ["DEFAULT_BUDGET", "FailureCase", "minimize_failure"]
