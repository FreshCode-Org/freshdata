"""Normalized decision hashing and repeat-consistency annotation.

The hash covers every decision-bearing ``DecisionRecord`` field.  Only the
approved telemetry fields — identifiers regenerated per run and the repeat
bookkeeping itself — may ever be excluded, and the exclusion list is closed:
asking to exclude anything else raises instead of silently narrowing the
comparison.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable, Sequence

from .exact import canonical_json
from .models import DecisionRecord

#: Fields regenerated per run/repeat that carry no decision content.  This set
#: is the *only* permissible exclusion surface.
APPROVED_TELEMETRY: frozenset[str] = frozenset(
    {
        "record_id",
        "run_id",
        "repeat",
        "repeat_hash",
        "repeat_consistent",
        "normalized_decision_hash",
    }
)

#: Every serialized field that participates in the normalized decision hash.
DECISION_FIELDS: tuple[str, ...] = tuple(
    sorted(
        set(DecisionRecord.__dataclass_fields__)
        - APPROVED_TELEMETRY
    )
)


def decision_hash(
    record: DecisionRecord, *, exclude: Iterable[str] = APPROVED_TELEMETRY
) -> str:
    """Stable SHA-256 over the record's decision-bearing content.

    ``exclude`` may only name approved telemetry fields; any attempt to drop a
    decision-bearing field (outputs, dispositions, confidence, rationale,
    audit evidence, trust, backend disclosure, ...) is rejected.
    """

    excluded = frozenset(exclude)
    illegal = excluded - APPROVED_TELEMETRY
    if illegal:
        raise ValueError(
            "decision-bearing fields cannot be excluded from the normalized "
            f"decision hash: {sorted(illegal)}"
        )
    payload = record.to_dict()
    body = {key: value for key, value in payload.items() if key not in excluded}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def annotate_repeats(
    records: Sequence[DecisionRecord], *, expected_repeats: Sequence[int]
) -> tuple[DecisionRecord, ...]:
    """Attach ``normalized_decision_hash``/``repeat_hash``/``repeat_consistent``.

    Records are grouped by ``(surface, requested_backend, cell_id)``.  A group
    is consistent only when it contains exactly one record per expected repeat
    and every repeat hashes identically.  Groups with missing repeats or any
    unexplained difference are annotated ``repeat_consistent=False`` so the
    ``default_nondeterminism`` gate fails closed.
    """

    expected = tuple(sorted({int(r) for r in expected_repeats}))
    if len(expected) < 2:
        raise ValueError("determinism verification requires at least two repeats")
    groups: dict[tuple[str, str | None, str], list[DecisionRecord]] = {}
    for record in records:
        groups.setdefault(
            (record.surface, record.requested_backend, record.cell_id), []
        ).append(record)

    annotated: list[DecisionRecord] = []
    for members in groups.values():
        hashes = {record.repeat: decision_hash(record) for record in members}
        repeats_seen = tuple(sorted(hashes))
        complete = repeats_seen == expected and len(members) == len(expected)
        consistent = complete and len(set(hashes.values())) == 1
        group_hash = hashlib.sha256(
            canonical_json([hashes[r] for r in repeats_seen]).encode("utf-8")
        ).hexdigest()
        for record in members:
            annotated.append(
                dataclasses.replace(
                    record,
                    normalized_decision_hash=hashes[record.repeat],
                    repeat_hash=group_hash,
                    repeat_consistent=consistent,
                )
            )
    return tuple(annotated)


__all__ = [
    "APPROVED_TELEMETRY",
    "DECISION_FIELDS",
    "annotate_repeats",
    "decision_hash",
]
