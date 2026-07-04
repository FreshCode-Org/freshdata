"""Human-verified release-gating eval labels.

Release gates may only be computed on labels where a human reviewer signed
off — teacher labels alone never gate a release. Verified label files are
JSONL with, per row, the payload plus ``human_verified``, ``reviewer`` and
``reviewed_at``; :func:`load_verified` refuses anything less.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import TRAINING_ROOT, read_jsonl, utc_now_iso, write_jsonl

DATA_DIR = TRAINING_ROOT / "eval" / "data"


class VerificationError(ValueError):
    """A release-gating label set is not fully human-verified."""


def verify_labels(
    labels: list[dict[str, Any]], *, reviewer: str, reviewed: list[bool] | None = None
) -> list[dict[str, Any]]:
    """Stamp labels as human-verified (used when a reviewer signs a batch)."""
    if not reviewer.strip():
        raise VerificationError("reviewer id/initials are required")
    flags = reviewed if reviewed is not None else [True] * len(labels)
    if len(flags) != len(labels):
        raise VerificationError("reviewed flags must match label count")
    now = utc_now_iso()
    out = []
    for label, flag in zip(labels, flags):
        stamped = dict(label)
        stamped["human_verified"] = bool(flag)
        stamped["reviewer"] = reviewer.strip()
        stamped["reviewed_at"] = now
        out.append(stamped)
    return out


def check_all_verified(labels: list[dict[str, Any]]) -> None:
    """Raise unless every label is human-verified with a reviewer recorded."""
    if not labels:
        raise VerificationError("release-gating eval set is empty")
    bad = [
        i for i, label in enumerate(labels)
        if label.get("human_verified") is not True
        or not str(label.get("reviewer", "")).strip()
    ]
    if bad:
        raise VerificationError(
            f"{len(bad)}/{len(labels)} release-gating labels are not human-verified "
            f"(first offenders: {bad[:5]}); teacher output alone cannot gate a release"
        )


def load_verified(path: Path | str) -> list[dict[str, Any]]:
    """Load a release-gating label file, enforcing 100% human verification."""
    labels = read_jsonl(path)
    check_all_verified(labels)
    return labels


def save_verified(path: Path | str, labels: list[dict[str, Any]]) -> Path:
    check_all_verified(labels)
    return write_jsonl(path, labels)
