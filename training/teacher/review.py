"""Human-review workflow for teacher outputs.

Policy (enforced, not advisory):

- at least **5%** of every task batch is sampled for human review;
- if teacher-human disagreement exceeds **3%**, the whole batch requires
  full review before any of it may be used;
- release-gating eval labels must be **100% human-verified** — an unreviewed
  teacher label can train a model but can never gate a release.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import utc_now_iso, write_json

SAMPLE_RATE = 0.05
DISAGREEMENT_THRESHOLD = 0.03


@dataclass(frozen=True)
class ReviewRecord:
    item_id: str
    reviewer: str
    agrees: bool
    reviewed_at: str
    disagreement_reason: str = ""
    corrected_payload: dict[str, Any] | None = None


@dataclass
class ReviewBatch:
    """Review state for one teacher task batch."""

    task_name: str
    items: list[dict[str, Any]]
    sample_ids: list[str] = field(default_factory=list)
    records: list[ReviewRecord] = field(default_factory=list)

    @staticmethod
    def item_id(index: int) -> str:
        return f"item_{index:05d}"

    def sample_for_review(self, *, rate: float = SAMPLE_RATE, seed: int = 0) -> list[str]:
        """Deterministically sample at least ``rate`` of items (min 1)."""
        if not self.items:
            self.sample_ids = []
            return []
        rng = random.Random(f"freshdata-review:{self.task_name}:{seed}")
        n = max(1, int(len(self.items) * rate + 0.999999))
        picks = sorted(rng.sample(range(len(self.items)), min(n, len(self.items))))
        self.sample_ids = [self.item_id(i) for i in picks]
        return self.sample_ids

    def record_review(
        self,
        item_id: str,
        *,
        reviewer: str,
        agrees: bool,
        disagreement_reason: str = "",
        corrected_payload: dict[str, Any] | None = None,
    ) -> ReviewRecord:
        if not reviewer.strip():
            raise ValueError("reviewer id/initials are required")
        if not agrees and not disagreement_reason.strip():
            raise ValueError("a disagreement must record its reason")
        record = ReviewRecord(
            item_id=item_id,
            reviewer=reviewer.strip(),
            agrees=agrees,
            reviewed_at=utc_now_iso(),
            disagreement_reason=disagreement_reason.strip(),
            corrected_payload=corrected_payload,
        )
        self.records.append(record)
        return record

    def disagreement_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if not r.agrees) / len(self.records)

    def requires_full_review(self) -> bool:
        return self.disagreement_rate() > DISAGREEMENT_THRESHOLD

    def reviewed_ids(self) -> set[str]:
        return {r.item_id for r in self.records}

    def usable_for_release_gating(self) -> bool:
        """True only when every item in the batch has a human review record."""
        return self.reviewed_ids() >= {self.item_id(i) for i in range(len(self.items))}

    def summary(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "n_items": len(self.items),
            "n_sampled": len(self.sample_ids),
            "n_reviewed": len(self.records),
            "sample_rate_required": SAMPLE_RATE,
            "disagreement_rate": round(self.disagreement_rate(), 6),
            "disagreement_threshold": DISAGREEMENT_THRESHOLD,
            "requires_full_review": self.requires_full_review(),
            "usable_for_release_gating": self.usable_for_release_gating(),
            "reviewers": sorted({r.reviewer for r in self.records}),
            "disagreements": [
                {"item_id": r.item_id, "reviewer": r.reviewer, "reason": r.disagreement_reason}
                for r in self.records if not r.agrees
            ],
        }

    def export_summary(self, path: Path | str) -> Path:
        return write_json(path, self.summary())
