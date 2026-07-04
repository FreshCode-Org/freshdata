"""Profile audit: what a profile learned, what it stores, what it will not do.

The audit is a first-class compartment (``audit.json`` inside the archive):
it must answer, without loading any data, whether raw sensitive literals are
stored, which artifacts were demoted and why, and where every rule came from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .types import DemotionRecord

if TYPE_CHECKING:  # pragma: no cover
    from .profile import LearningProfile

__all__ = ["ProfileAudit", "build_audit"]


@dataclass
class ProfileAudit:
    """Printable, serializable audit of a learning profile."""

    profile_id: str
    created_at: str
    privacy_mode: str
    contains_raw_values: bool
    rule_count: int
    rules_by_family: dict[str, int]
    advisory_rule_count: int
    value_map_columns: int
    value_map_entries: int
    masked_entries: int
    example_count: int
    masked_examples: int
    has_memory: bool
    has_vectors: bool
    embedding_model_id: str | None
    sensitive_columns: dict[str, str]
    protection_candidates: list[str]
    alignment: dict[str, Any]
    holdout_metrics: dict[str, Any]
    demotions: list[DemotionRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "created_at": self.created_at,
            "privacy_mode": self.privacy_mode,
            "contains_raw_values": self.contains_raw_values,
            "rule_count": self.rule_count,
            "rules_by_family": dict(self.rules_by_family),
            "advisory_rule_count": self.advisory_rule_count,
            "value_map_columns": self.value_map_columns,
            "value_map_entries": self.value_map_entries,
            "masked_entries": self.masked_entries,
            "example_count": self.example_count,
            "masked_examples": self.masked_examples,
            "has_memory": self.has_memory,
            "has_vectors": self.has_vectors,
            "embedding_model_id": self.embedding_model_id,
            "sensitive_columns": dict(self.sensitive_columns),
            "protection_candidates": list(self.protection_candidates),
            "alignment": dict(self.alignment),
            "holdout_metrics": dict(self.holdout_metrics),
            "demotions": [d.to_dict() for d in self.demotions],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProfileAudit:
        return cls(
            profile_id=str(data.get("profile_id", "")),
            created_at=str(data.get("created_at", "")),
            privacy_mode=str(data.get("privacy_mode", "mask")),
            contains_raw_values=bool(data.get("contains_raw_values", False)),
            rule_count=int(data.get("rule_count", 0)),
            rules_by_family=dict(data.get("rules_by_family", {})),
            advisory_rule_count=int(data.get("advisory_rule_count", 0)),
            value_map_columns=int(data.get("value_map_columns", 0)),
            value_map_entries=int(data.get("value_map_entries", 0)),
            masked_entries=int(data.get("masked_entries", 0)),
            example_count=int(data.get("example_count", 0)),
            masked_examples=int(data.get("masked_examples", 0)),
            has_memory=bool(data.get("has_memory", False)),
            has_vectors=bool(data.get("has_vectors", False)),
            embedding_model_id=data.get("embedding_model_id"),
            sensitive_columns=dict(data.get("sensitive_columns", {})),
            protection_candidates=list(data.get("protection_candidates", [])),
            alignment=dict(data.get("alignment", {})),
            holdout_metrics=dict(data.get("holdout_metrics", {})),
            demotions=[DemotionRecord.from_dict(d) for d in data.get("demotions", [])],
            notes=list(data.get("notes", [])),
        )

    def render(self) -> str:
        lines = [
            f"LearningProfile {self.profile_id}",
            f"  created:        {self.created_at}",
            f"  privacy:        {self.privacy_mode}"
            + ("  ** CONTAINS RAW SENSITIVE VALUES **" if self.contains_raw_values else ""),
            f"  rules:          {self.rule_count} "
            f"({self.advisory_rule_count} advisory/suggest-only)",
        ]
        for family, count in sorted(self.rules_by_family.items()):
            lines.append(f"    - {family}: {count}")
        lines.append(
            f"  value maps:     {self.value_map_columns} column(s), "
            f"{self.value_map_entries} entries, {self.masked_entries} masked"
        )
        lines.append(f"  examples:       {self.example_count} ({self.masked_examples} masked)")
        lines.append(f"  memory:         {'embedded' if self.has_memory else 'none'}")
        vector_note = (
            f"vectors via {self.embedding_model_id}" if self.has_vectors else "no vectors"
        )
        lines.append(f"  examples bank:  {vector_note}")
        if self.sensitive_columns:
            cols = ", ".join(f"{c} ({t})" for c, t in sorted(self.sensitive_columns.items()))
            lines.append(f"  sensitive cols: {cols}")
        if self.protection_candidates:
            lines.append(
                "  protection candidates (recorded, not enforced): "
                + ", ".join(self.protection_candidates)
            )
        if self.holdout_metrics.get("skipped_reason"):
            lines.append(f"  holdout:        {self.holdout_metrics['skipped_reason']}")
        elif self.holdout_metrics:
            lines.append(
                "  holdout:        "
                f"{self.holdout_metrics.get('correct', 0)}/"
                f"{self.holdout_metrics.get('proposed', 0)} proposed repairs correct, "
                f"{self.holdout_metrics.get('false_modifications', 0)} false, "
                f"{self.holdout_metrics.get('missed', 0)} missed"
            )
        if self.demotions:
            lines.append(f"  demotions ({len(self.demotions)}):")
            for demotion in self.demotions:
                lines.append(f"    - {demotion.target}: {demotion.outcome} — {demotion.reason}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.render()


def build_audit(profile: LearningProfile, **extra: Any) -> ProfileAudit:
    """Compute a fresh audit from an in-memory profile."""
    rules_by_family: dict[str, int] = {}
    advisory = 0
    for rule in profile.rules:
        family = str(rule.params.get("transform_family", rule.rule))
        rules_by_family[family] = rules_by_family.get(family, 0) + 1
        if rule.enforcement == "advisory":
            advisory += 1
    entries = [e for vm in profile.value_maps.values() for e in vm.entries]
    examples = profile.examples.examples if profile.examples is not None else []
    return ProfileAudit(
        profile_id=profile.profile_id,
        created_at=profile.manifest.created_at,
        privacy_mode=profile.manifest.privacy_mode,
        contains_raw_values=profile.manifest.contains_raw_values,
        rule_count=len(profile.rules),
        rules_by_family=rules_by_family,
        advisory_rule_count=advisory,
        value_map_columns=len(profile.value_maps),
        value_map_entries=len(entries),
        masked_entries=sum(1 for e in entries if e.masked),
        example_count=len(examples),
        masked_examples=sum(1 for e in examples if e.masked),
        has_memory=profile.memory is not None,
        has_vectors=bool(profile.examples is not None and profile.examples.vectors_path),
        embedding_model_id=(
            profile.examples.embedding_model_id if profile.examples is not None else None
        ),
        sensitive_columns=dict(extra.get("sensitive_columns", {})),
        protection_candidates=list(extra.get("protection_candidates", [])),
        alignment=dict(extra.get("alignment", {})),
        holdout_metrics=dict(extra.get("holdout_metrics", {})),
        demotions=list(extra.get("demotions", [])),
        notes=list(extra.get("notes", [])),
    )
