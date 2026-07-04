"""Profile diff and merge.

Diff is programmatic *and* printable; merge supports four strategies:

* ``union_min_precision`` (default) — union of both profiles, entries must
  meet the stricter of the two precision thresholds, direct conflicts are
  dropped and recorded (never silently resolved into a winner).
* ``prefer_self`` / ``prefer_other`` — the chosen side wins every conflict.
* ``error_on_conflict`` — any conflict raises with the full conflict list.

The merged profile always recomputes its manifest (hashes are recomputed at
save time) and keeps provenance from both parents in its audit notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from ..context.types import ColumnConstraint
from .audit import build_audit
from .types import (
    PROFILE_FORMAT_VERSION,
    ExampleBank,
    ProfileError,
    ProfileManifest,
    ValueMap,
    ValueMapEntry,
)

__all__ = ["ProfileDiff", "ProfileMergeError", "diff_profiles", "merge_profiles"]


class ProfileMergeError(ProfileError):
    """Raised by ``strategy="error_on_conflict"`` when profiles disagree."""


@dataclass
class ProfileDiff:
    """Structured comparison of two profiles (self vs other)."""

    added_rules: list[str] = field(default_factory=list)
    removed_rules: list[str] = field(default_factory=list)
    changed_rules: list[str] = field(default_factory=list)
    added_value_maps: list[str] = field(default_factory=list)
    removed_value_maps: list[str] = field(default_factory=list)
    conflicting_value_maps: list[str] = field(default_factory=list)
    privacy_differences: list[str] = field(default_factory=list)
    schema_differences: list[str] = field(default_factory=list)
    model_differences: list[str] = field(default_factory=list)
    memory_differences: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(
            getattr(self, name)
            for name in (
                "added_rules",
                "removed_rules",
                "changed_rules",
                "added_value_maps",
                "removed_value_maps",
                "conflicting_value_maps",
                "privacy_differences",
                "schema_differences",
                "model_differences",
                "memory_differences",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_rules": list(self.added_rules),
            "removed_rules": list(self.removed_rules),
            "changed_rules": list(self.changed_rules),
            "added_value_maps": list(self.added_value_maps),
            "removed_value_maps": list(self.removed_value_maps),
            "conflicting_value_maps": list(self.conflicting_value_maps),
            "privacy_differences": list(self.privacy_differences),
            "schema_differences": list(self.schema_differences),
            "model_differences": list(self.model_differences),
            "memory_differences": list(self.memory_differences),
        }

    def render(self) -> str:
        if self.is_empty:
            return "profiles are equivalent"
        lines = []
        sections = (
            ("rules added (other only)", self.added_rules),
            ("rules removed (self only)", self.removed_rules),
            ("rules changed", self.changed_rules),
            ("value maps added", self.added_value_maps),
            ("value maps removed", self.removed_value_maps),
            ("value map conflicts", self.conflicting_value_maps),
            ("privacy", self.privacy_differences),
            ("schema", self.schema_differences),
            ("models/vectors", self.model_differences),
            ("memory", self.memory_differences),
        )
        for title, items in sections:
            if items:
                lines.append(f"{title}:")
                lines.extend(f"  - {item}" for item in items)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.render()


def _rule_key(rule: ColumnConstraint) -> str:
    return rule.id


def _rule_payload(rule: ColumnConstraint) -> dict[str, Any]:
    payload = rule.to_dict()
    payload.pop("provenance", None)
    return payload


def diff_profiles(self_profile: Any, other_profile: Any) -> ProfileDiff:
    """Compare two LearningProfiles into a ProfileDiff."""
    diff = ProfileDiff()

    self_rules = {_rule_key(r): r for r in self_profile.rules}
    other_rules = {_rule_key(r): r for r in other_profile.rules}
    diff.added_rules = sorted(set(other_rules) - set(self_rules))
    diff.removed_rules = sorted(set(self_rules) - set(other_rules))
    diff.changed_rules = sorted(
        rule_id
        for rule_id in set(self_rules) & set(other_rules)
        if _rule_payload(self_rules[rule_id]) != _rule_payload(other_rules[rule_id])
    )

    self_maps = self_profile.value_maps
    other_maps = other_profile.value_maps
    diff.added_value_maps = sorted(set(other_maps) - set(self_maps))
    diff.removed_value_maps = sorted(set(self_maps) - set(other_maps))
    for column in sorted(set(self_maps) & set(other_maps)):
        for raw, conflict in _map_conflicts(self_maps[column], other_maps[column]):
            diff.conflicting_value_maps.append(f"{column}: {raw!r} -> {conflict}")

    if self_profile.manifest.privacy_mode != other_profile.manifest.privacy_mode:
        diff.privacy_differences.append(
            f"privacy_mode: {self_profile.manifest.privacy_mode} vs "
            f"{other_profile.manifest.privacy_mode}"
        )
    if self_profile.manifest.contains_raw_values != other_profile.manifest.contains_raw_values:
        diff.privacy_differences.append(
            f"contains_raw_values: {self_profile.manifest.contains_raw_values} vs "
            f"{other_profile.manifest.contains_raw_values}"
        )

    if self_profile.manifest.source_schema_hash != other_profile.manifest.source_schema_hash:
        diff.schema_differences.append("source schemas differ")
    if self_profile.manifest.dataset_signature != other_profile.manifest.dataset_signature:
        diff.schema_differences.append("dataset signatures differ")

    self_model = self_profile.examples.embedding_model_id if self_profile.examples else None
    other_model = other_profile.examples.embedding_model_id if other_profile.examples else None
    if self_model != other_model:
        diff.model_differences.append(f"embedding model: {self_model} vs {other_model}")

    self_memory = self_profile.memory
    other_memory = other_profile.memory
    if (self_memory is None) != (other_memory is None):
        diff.memory_differences.append(
            f"memory: {'embedded' if self_memory else 'none'} vs "
            f"{'embedded' if other_memory else 'none'}"
        )
    elif self_memory is not None and other_memory is not None:
        if self_memory.value_patterns != other_memory.value_patterns:
            diff.memory_differences.append("memory value_patterns differ")
        if self_memory.thresholds != other_memory.thresholds:
            diff.memory_differences.append("memory thresholds differ")
    return diff


def _map_conflicts(a: ValueMap, b: ValueMap) -> list[tuple[object, str]]:
    conflicts = []
    b_lookup = {e.raw_value: e for e in b.entries}
    for entry in a.entries:
        other = b_lookup.get(entry.raw_value)
        if other is not None and str(other.clean_value) != str(entry.clean_value):
            conflicts.append((entry.raw_value, f"{entry.clean_value!r} vs {other.clean_value!r}"))
    return conflicts


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

Strategy = Literal["union_min_precision", "prefer_self", "prefer_other", "error_on_conflict"]

_STRATEGIES = ("union_min_precision", "prefer_self", "prefer_other", "error_on_conflict")


def _merged_privacy(self_profile: Any, other_profile: Any, strategy: Strategy) -> str:
    """Privacy mode of the merged profile.

    ``prefer_self``/``prefer_other`` follow the preferred parent; everything
    else takes the *strictest* of the two modes ("mask" beats "none"), so a
    union merge can never weaken privacy. ``error_on_conflict`` raised before
    this point when the modes differed.
    """
    self_mode = str(self_profile.manifest.privacy_mode)
    other_mode = str(other_profile.manifest.privacy_mode)
    if strategy == "prefer_self":
        return self_mode
    if strategy == "prefer_other":
        return other_mode
    return "mask" if "mask" in (self_mode, other_mode) else self_mode


def merge_profiles(self_profile: Any, other_profile: Any, *, strategy: Strategy) -> Any:
    """Merge two profiles under the given conflict strategy."""
    from .profile import LearningProfile, _freshdata_version  # noqa: PLC0415 - cycle

    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown merge strategy {strategy!r}; choose one of {_STRATEGIES}")

    conflicts: list[str] = []
    notes: list[str] = [
        f"merged from {self_profile.profile_id} and {other_profile.profile_id} "
        f"(strategy={strategy})"
    ]

    if self_profile.manifest.privacy_mode != other_profile.manifest.privacy_mode:
        conflicts.append(
            "privacy_mode conflict: "
            f"{self_profile.manifest.privacy_mode} vs {other_profile.manifest.privacy_mode}"
        )
    if self_profile.manifest.source_schema_hash != other_profile.manifest.source_schema_hash:
        conflicts.append("incompatible schema signature (source_schema_hash differs)")

    rules, rule_conflicts = _merge_rules(self_profile.rules, other_profile.rules, strategy)
    conflicts.extend(rule_conflicts)

    min_precision = max(
        _profile_min_precision(self_profile), _profile_min_precision(other_profile)
    )
    value_maps, map_conflicts_found = _merge_value_maps(
        self_profile.value_maps, other_profile.value_maps, strategy, min_precision
    )
    conflicts.extend(map_conflicts_found)

    if strategy == "error_on_conflict" and conflicts:
        raise ProfileMergeError(
            "profiles conflict; refusing to merge:\n  - " + "\n  - ".join(conflicts)
        )
    if conflicts:
        notes.extend(f"conflict ({_resolution_word(strategy)}): {c}" for c in conflicts)

    examples = _merge_examples(self_profile.examples, other_profile.examples)
    memory = _merge_memory(self_profile.memory, other_profile.memory, strategy)

    privacy_mode = _merged_privacy(self_profile, other_profile, strategy)
    base = other_profile if strategy == "prefer_other" else self_profile
    manifest = ProfileManifest(
        profile_version=PROFILE_FORMAT_VERSION,
        freshdata_version=_freshdata_version(),
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_id=f"merge({self_profile.manifest.dataset_id},"
        f"{other_profile.manifest.dataset_id})",
        dataset_signature=base.manifest.dataset_signature,
        source_schema_hash=base.manifest.source_schema_hash,
        clean_schema_hash=base.manifest.clean_schema_hash,
        context_hash=base.manifest.context_hash,
        privacy_mode=privacy_mode,
        contains_raw_values=(
            self_profile.manifest.contains_raw_values or other_profile.manifest.contains_raw_values
        ),
        compartments=(),
        member_hashes={},  # recomputed at save time
    )
    merged = LearningProfile(
        manifest=manifest,
        rules=rules,
        value_maps=value_maps,
        examples=examples,
        memory=memory,
        vectors=None,  # vectors are model-tied; re-learn to regenerate them
    )
    parent_audits = [p.audit() for p in (self_profile, other_profile)]
    merged.audit_info = build_audit(
        merged,
        sensitive_columns={
            **parent_audits[0].sensitive_columns,
            **parent_audits[1].sensitive_columns,
        },
        protection_candidates=sorted(
            set(parent_audits[0].protection_candidates)
            | set(parent_audits[1].protection_candidates)
        ),
        alignment={"merged": True},
        holdout_metrics={},
        demotions=[*parent_audits[0].demotions, *parent_audits[1].demotions],
        notes=notes,
    )
    return merged


def _resolution_word(strategy: str) -> str:
    return {
        "union_min_precision": "dropped both sides",
        "prefer_self": "kept self",
        "prefer_other": "kept other",
    }.get(strategy, strategy)


def _profile_min_precision(profile: Any) -> float:
    precisions = [vm.min_precision for vm in profile.value_maps.values()]
    return max(precisions) if precisions else 0.0


def _merge_rules(
    self_rules: list[ColumnConstraint],
    other_rules: list[ColumnConstraint],
    strategy: str,
) -> tuple[list[ColumnConstraint], list[str]]:
    merged: dict[str, ColumnConstraint] = {r.id: r for r in self_rules}
    conflicts: list[str] = []
    for rule in other_rules:
        existing = merged.get(rule.id)
        if existing is None:
            merged[rule.id] = rule
            continue
        if _rule_payload(existing) == _rule_payload(rule):
            continue
        detail = _describe_rule_conflict(existing, rule)
        conflicts.append(detail)
        if strategy == "prefer_other":
            merged[rule.id] = rule
        elif strategy == "union_min_precision":
            del merged[rule.id]
        # prefer_self keeps existing; error_on_conflict raises in caller
    return list(merged.values()), conflicts


def _describe_rule_conflict(a: ColumnConstraint, b: ColumnConstraint) -> str:
    for param in ("region", "sentinels", "values", "strategy"):
        left, right = a.params.get(param), b.params.get(param)
        if left != right and (left is not None or right is not None):
            return f"rule {a.id}: {param} {left!r} vs {right!r}"
    return f"rule {a.id}: params differ ({a.params!r} vs {b.params!r})"


def _merge_value_maps(
    self_maps: dict[str, ValueMap],
    other_maps: dict[str, ValueMap],
    strategy: str,
    min_precision: float,
) -> tuple[dict[str, ValueMap], list[str]]:
    merged: dict[str, ValueMap] = {}
    conflicts: list[str] = []
    for column in sorted(set(self_maps) | set(other_maps)):
        a, b = self_maps.get(column), other_maps.get(column)
        if a is None or b is None:
            source = a or b
            assert source is not None
            merged[column] = _filtered_copy(source, strategy, min_precision)
            continue
        entries: dict[object, ValueMapEntry] = {e.raw_value: e for e in a.entries}
        for entry in b.entries:
            existing = entries.get(entry.raw_value)
            if existing is None:
                entries[entry.raw_value] = entry
            elif str(existing.clean_value) != str(entry.clean_value):
                conflicts.append(
                    f"value map '{column}': {entry.raw_value!r} -> "
                    f"{existing.clean_value!r} vs {entry.clean_value!r}"
                )
                if strategy == "prefer_other":
                    entries[entry.raw_value] = entry
                elif strategy == "union_min_precision":
                    del entries[entry.raw_value]
            else:
                # Same mapping seen in both: combine evidence.
                entries[entry.raw_value] = ValueMapEntry(
                    raw_value=existing.raw_value,
                    clean_value=existing.clean_value,
                    support=existing.support + entry.support,
                    precision=min(existing.precision, entry.precision),
                    transform_family=existing.transform_family,
                    masked=existing.masked or entry.masked,
                )
        result = ValueMap(
            column=column,
            entries=sorted(entries.values(), key=lambda e: (-e.support, str(e.raw_value))),
            min_precision=max(a.min_precision, b.min_precision),
            min_support=max(a.min_support, b.min_support),
            capped=a.capped or b.capped,
            masked=a.masked or b.masked,
        )
        merged[column] = _filtered_copy(result, strategy, min_precision)
        if not merged[column].entries:
            del merged[column]
    return merged, conflicts


def _filtered_copy(value_map: ValueMap, strategy: str, min_precision: float) -> ValueMap:
    if strategy != "union_min_precision":
        return value_map
    kept = [e for e in value_map.entries if e.masked or e.precision >= min_precision]
    return ValueMap(
        column=value_map.column,
        entries=kept,
        min_precision=max(value_map.min_precision, min_precision),
        min_support=value_map.min_support,
        capped=value_map.capped,
        masked=value_map.masked,
    )


def _merge_examples(a: ExampleBank | None, b: ExampleBank | None) -> ExampleBank | None:
    if a is None and b is None:
        return None
    seen = set()
    examples = []
    for bank in (a, b):
        if bank is None:
            continue
        for example in bank.examples:
            key = (example.column, str(example.raw_value), str(example.clean_value))
            if key not in seen:
                seen.add(key)
                examples.append(example)
    return ExampleBank(
        examples=examples,
        vectors_path=None,
        embedding_model_id=None,
        masked=all(e.masked for e in examples) if examples else True,
    )


def _merge_memory(a: Any, b: Any, strategy: str) -> Any:
    if a is None:
        return b
    if b is None or strategy == "prefer_self":
        return a
    if strategy == "prefer_other":
        return b
    # union: keep self's memory but fold in non-conflicting value patterns.
    merged_patterns = {c: dict(p) for c, p in a.value_patterns.items()}
    for column, patterns in b.value_patterns.items():
        target = merged_patterns.setdefault(column, {})
        for raw, clean in patterns.items():
            if raw in target and str(target[raw]) != str(clean):
                del target[raw]  # conflicting mapping: drop from replay
            elif raw not in target:
                target[raw] = clean
    a.value_patterns = merged_patterns
    return a
