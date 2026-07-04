"""Stage 4 of the learning pipeline: turn classified diffs into artifacts.

Artifacts come in five shapes:

* **rules** — :class:`~freshdata.context.types.ColumnConstraint` objects that
  speak the existing Phase-1 policy vocabulary (``valid_format``,
  ``locale_format``, ``allowed_values`` …) so replay lowers through the same
  ``ContextPolicy.lower()`` path as user context.  Families with no lowering
  are kept with ``enforcement="advisory"`` as auditable evidence.
* **config deltas** — ``extra_sentinels`` and ``dayfirst``.
* **value maps** — literal ``raw -> clean`` repairs with support/precision.
* **examples** — pairs that explain nothing reusable (never replayed).
* **embedded memory** — a :class:`~freshdata.CleaningMemory` built through
  the existing constructor so Phase-2 replay semantics stay untouched.

Hard rules: imputation never becomes a literal map, low-support patterns are
demoted to examples, sensitive literals are masked under ``privacy="mask"``,
and protected columns are skipped entirely unless explicitly allowed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..context.types import ColumnConstraint, Provenance
from ..memory import CleaningMemory, learn_cleaning_memory
from .privacy import mask_value
from .types import (
    AlignedPair,
    ClassifiedDiff,
    DiffSummary,
    ExamplePair,
    ValueMap,
    ValueMapEntry,
)

__all__ = ["ExtractionResult", "extract_artifacts"]

#: Families whose repairs are literal and column-local: they become value-map
#: entries.  Everything else replays algorithmically (or not at all).
_LITERAL_MAP_FAMILIES = frozenset({"reference_normalize", "allowed_value_map", "category_map"})

#: Families that emit a rule in the Phase-1 policy vocabulary (lowered at
#: replay), keyed to (rule, action, params-builder description).
_ADVISORY_FAMILIES = frozenset(
    {
        "date_parse",
        "currency_parse",
        "unit_strip",
        "boolean_synonym",
        "spelled_number",
        "numeric_rounding",
        "dtype_coercion",
        "case_fold",
        "whitespace",
    }
)

#: Provenance tier for learned constraints (0-2 are parser tiers, 3 is
#: memory; 4 marks paired-data learning).
_LEARNED_TIER = 4


@dataclass
class ExtractionResult:
    """Everything stage 4 learned, before holdout validation."""

    rules: list[ColumnConstraint] = field(default_factory=list)
    config_deltas: dict[str, Any] = field(default_factory=dict)
    value_maps: dict[str, ValueMap] = field(default_factory=dict)
    examples: list[ExamplePair] = field(default_factory=list)
    memory: CleaningMemory | None = None
    protection_candidates: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _learned_provenance(family: str, column: str, support: int, precision: float) -> Provenance:
    return Provenance(
        sentence=(
            f"learned[{family}] on '{column}': {support} training example(s), "
            f"precision {precision:.3f}"
        ),
        tier=_LEARNED_TIER,
        parse_confidence=round(precision, 4),
    )


def _rule(
    rule_id: str,
    column: str,
    rule: str,
    action: str,
    params: Mapping[str, Any],
    enforcement: str,
    support: int,
    precision: float,
    family: str,
) -> ColumnConstraint:
    merged = dict(params)
    merged["learned"] = True
    merged["support"] = support
    merged["precision"] = round(precision, 4)
    merged["transform_family"] = family
    return ColumnConstraint(
        id=rule_id,
        column=column,
        resolved_from=column,
        resolution_confidence=1.0,
        rule=rule,
        action=action,
        params=merged,
        enforcement=enforcement,
        provenance=_learned_provenance(family, column, support, precision),
    )


def _raw_occurrences(series: pd.Series) -> pd.Series:
    return series.astype(str).value_counts(dropna=False)


def _known_imputation_strategy(messy_col: pd.Series, clean_value: object) -> str | None:
    """Match an externally imputed literal to a known imputation model."""
    non_null = messy_col.dropna()
    if non_null.empty:
        return None
    numeric = pd.to_numeric(non_null, errors="coerce").dropna()
    target = pd.to_numeric(pd.Series([clean_value]), errors="coerce").iloc[0]
    if len(numeric) >= max(2, int(0.5 * len(non_null))) and pd.notna(target):
        for name, stat in (
            ("median", float(numeric.median())),
            ("mean", float(numeric.mean())),
        ):
            if abs(float(target) - stat) <= 1e-9 + 1e-6 * max(1.0, abs(stat)):
                return name
    mode = non_null.mode()
    if len(mode) > 0 and str(clean_value) == str(mode.iloc[0]):
        return "mode"
    return None


def extract_artifacts(
    classified: Mapping[str, list[ClassifiedDiff]],
    aligned: AlignedPair,
    summary: DiffSummary,
    *,
    dataset_id: str,
    sensitive: Mapping[str, str],
    salt: str,
    privacy: str = "mask",
    include_sensitive: bool = False,
    protected: frozenset[str] = frozenset(),
    min_support: int = 5,
    min_precision: float = 0.98,
    max_map_size: int = 2_000,
) -> ExtractionResult:
    """Extract reusable artifacts from classified diffs (train split only)."""
    result = ExtractionResult()
    messy = aligned.messy_aligned
    mask_literals = privacy == "mask" or not include_sensitive
    value_patterns: dict[str, dict[str, Any]] = {}
    learned_sentinels: list[str] = []

    for column, items in classified.items():
        if column in protected:
            result.notes.append(
                f"skipped learning on protected column '{column}' "
                "(pass include_sensitive=True and an explicit policy to override)"
            )
            continue
        semantic_type = sensitive.get(column)
        col_masked = semantic_type is not None and mask_literals
        raw_counts = (
            _raw_occurrences(messy[column])
            if column in getattr(messy, "columns", [])
            else pd.Series(dtype="int64")
        )

        by_family: dict[str, list[ClassifiedDiff]] = {}
        for item in items:
            by_family.setdefault(item.family, []).append(item)

        map_entries: list[ValueMapEntry] = []
        for family, group in by_family.items():
            total_support = sum(g.diff.support for g in group)

            if family == "unexplained":
                result.examples.extend(
                    _as_examples(group, column, semantic_type, salt, col_masked)
                )
                continue

            if family == "missing_imputation":
                _extract_imputation(result, group, messy, column, min_support)
                continue

            if family == "row_drop_evidence":  # produced only by keyed row diffs
                continue

            if total_support < min_support:
                result.examples.extend(
                    _as_examples(group, column, semantic_type, salt, col_masked)
                )
                result.notes.append(
                    f"'{column}': {family} seen only {total_support}x "
                    f"(< min_support={min_support}); kept as examples"
                )
                continue

            if family in _LITERAL_MAP_FAMILIES:
                map_entries.extend(
                    _literal_entries(group, raw_counts, family, col_masked, semantic_type, salt)
                )

            rule = _family_rule(
                f"learned:{column}:{family}",
                column,
                family,
                group,
                total_support,
                learned_sentinels,
                result.config_deltas,
                col_masked,
                allowed_values=_clean_allowed_values(aligned, column),
            )
            if rule is not None:
                result.rules.append(rule)

        if map_entries:
            replayable = [e for e in map_entries if not e.masked]
            capped = len(map_entries) > max_map_size
            if capped:
                map_entries = sorted(map_entries, key=lambda e: (-e.support, str(e.raw_value)))[
                    :max_map_size
                ]
            result.value_maps[column] = ValueMap(
                column=column,
                entries=map_entries,
                min_precision=min_precision,
                min_support=min_support,
                capped=capped,
                masked=col_masked,
            )
            if replayable:
                value_patterns[column] = {str(e.raw_value): e.clean_value for e in replayable}

    if learned_sentinels:
        result.config_deltas["extra_sentinels"] = tuple(sorted(set(learned_sentinels)))

    _detect_protection_candidates(result, aligned, summary, protected)

    result.memory = learn_cleaning_memory(
        messy if len(messy) else aligned.messy_aligned,
        [],
        dataset_id,
        value_patterns=value_patterns,
    )
    return result


_ALLOWED_RULE_CARDINALITY = 20


def _clean_allowed_values(aligned: AlignedPair, column: str) -> tuple[str, ...]:
    clean = aligned.clean_aligned
    if column not in getattr(clean, "columns", []):
        return ()
    col = clean[column]
    if not isinstance(col, pd.Series):
        return ()
    distinct = col.dropna().unique()
    if 0 < len(distinct) <= _ALLOWED_RULE_CARDINALITY:
        return tuple(sorted(str(v) for v in distinct))
    return ()


def _as_examples(
    group: list[ClassifiedDiff],
    column: str,
    semantic_type: str | None,
    salt: str,
    masked: bool,
) -> list[ExamplePair]:
    examples = []
    for item in group:
        raw, clean = item.diff.raw_value, item.diff.clean_value
        if masked:
            stype = semantic_type or "free_text"
            raw = mask_value(raw, salt=salt, semantic_type=stype)
            clean = mask_value(clean, salt=salt, semantic_type=stype)
        examples.append(
            ExamplePair(
                column=column,
                raw_value=raw,
                clean_value=clean,
                transform_family=item.family,
                support=item.diff.support,
                masked=masked,
            )
        )
    return examples


def _literal_entries(
    group: list[ClassifiedDiff],
    raw_counts: pd.Series,
    family: str,
    col_masked: bool,
    semantic_type: str | None,
    salt: str,
) -> list[ValueMapEntry]:
    entries = []
    for item in group:
        raw, clean = item.diff.raw_value, item.diff.clean_value
        occurrences = int(raw_counts.get(str(raw), item.diff.support))
        precision = item.diff.support / occurrences if occurrences else 0.0
        if col_masked:
            stype = semantic_type or "free_text"
            raw = mask_value(raw, salt=salt, semantic_type=stype)
            clean = mask_value(clean, salt=salt, semantic_type=stype)
        entries.append(
            ValueMapEntry(
                raw_value=raw,
                clean_value=clean,
                support=item.diff.support,
                precision=min(1.0, precision),
                transform_family=family,
                masked=col_masked,
            )
        )
    return entries


def _family_rule(
    rule_id: str,
    column: str,
    family: str,
    group: list[ClassifiedDiff],
    support: int,
    learned_sentinels: list[str],
    config_deltas: dict[str, Any],
    col_masked: bool,
    allowed_values: tuple[str, ...] = (),
) -> ColumnConstraint | None:
    precision = 1.0  # within-train precision; holdout eval refines it
    if family == "email_normalize":
        return _rule(
            rule_id,
            column,
            "valid_format",
            "normalize_or_flag",
            {"format": "email"},
            "soft",
            support,
            precision,
            family,
        )
    if family == "phone_normalize":
        region = str(group[0].params.get("region", "IN"))
        return _rule(
            rule_id,
            column,
            "locale_format",
            "normalize_or_flag",
            {"format": "phone", "region": region},
            "soft",
            support,
            precision,
            family,
        )
    if family == "date_dayfirst_inference":
        config_deltas["dayfirst"] = True
        return _rule(
            rule_id,
            column,
            "valid_format",
            "normalize_or_flag",
            {"format": "date", "dayfirst": True},
            "soft",
            support,
            precision,
            family,
        )
    if family == "sentinel_to_missing":
        if not col_masked:
            tokens = sorted(
                {str(item.params.get("sentinel", item.diff.raw_value)) for item in group}
            )
            learned_sentinels.extend(tokens)
            return _rule(
                rule_id,
                column,
                "sentinel",
                "to_missing",
                {"sentinels": tokens},
                "soft",
                support,
                precision,
                family,
            )
        return _rule(
            rule_id,
            column,
            "sentinel",
            "to_missing",
            {"sentinels": []},
            "advisory",
            support,
            precision,
            family,
        )
    if family in {"reference_normalize", "allowed_value_map"}:
        allowed = allowed_values
        for item in group:
            from_params = item.params.get("allowed_values")
            if from_params:
                allowed = tuple(str(v) for v in from_params)
                break
        params: dict[str, Any] = {}
        if allowed and not col_masked:
            params["values"] = tuple(allowed)
        return _rule(
            rule_id,
            column,
            "allowed_values",
            "map_or_flag",
            params,
            "soft" if params else "advisory",
            support,
            precision,
            family,
        )
    if family == "category_map":
        return _rule(
            rule_id,
            column,
            "category_map",
            "map_values",
            {},
            "advisory",
            support,
            precision,
            family,
        )
    if family in _ADVISORY_FAMILIES:
        return _rule(
            rule_id,
            column,
            "learned_evidence",
            "observe",
            {"family": family},
            "advisory",
            support,
            precision,
            family,
        )
    return None


def _extract_imputation(
    result: ExtractionResult,
    group: list[ClassifiedDiff],
    messy: pd.DataFrame,
    column: str,
    min_support: int,
) -> None:
    """Imputation becomes strategy evidence only — never a literal map."""
    total = sum(item.diff.support for item in group)
    strategies = {
        _known_imputation_strategy(messy[column], item.diff.clean_value)
        for item in group
        if column in messy.columns
    }
    strategies.discard(None)
    if len(strategies) == 1 and total >= min_support:
        strategy = next(iter(strategies))
        result.rules.append(
            _rule(
                f"learned:{column}:missing_imputation",
                column,
                "impute_missing",
                "suggest_strategy",
                {"strategy": strategy},
                "advisory",
                total,
                1.0,
                "missing_imputation",
            )
        )
    else:
        result.notes.append(
            f"'{column}': {total} imputed cell(s) did not match a known "
            "imputation model; dropped (imputation literals are never learned)"
        )


def _detect_protection_candidates(
    result: ExtractionResult,
    aligned: AlignedPair,
    summary: DiffSummary,
    protected: frozenset[str],
) -> None:
    """Record (never enforce) columns that look intentionally untouched."""
    from ..semantic.experts import column_name_is_identifier  # noqa: PLC0415

    changed = set(summary.column_diffs)
    for column in summary.schema_diffs.shared_columns:
        if column in changed or column in protected:
            continue
        if column_name_is_identifier(str(column)):
            result.protection_candidates.append(str(column))
