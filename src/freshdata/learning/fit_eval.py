"""Stage 5 of the learning pipeline: holdout validation and demotion.

The aligned pair is split deterministically into train/holdout rows.  A
candidate profile is extracted from the train rows only, replayed forward on
the holdout messy cells, and compared against the holdout clean cells.  Any
artifact whose measured precision falls below ``min_precision`` is demoted:

* rules and value maps below threshold become examples only,
* unsafe literal maps are dropped,
* imputation literals are dropped (they were never maps to begin with),
* ambiguous repairs become suggest-only (``enforcement="advisory"``).

Every demotion carries a human-readable reason into the profile audit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..context.types import ColumnConstraint
from ..semantic.formats import normalize_phone_in, repair_email
from .classify import _norm_token, _parse_date  # shared deterministic helpers
from .extract import ExtractionResult, extract_artifacts
from .types import (
    AlignedPair,
    AlignmentReport,
    DemotionRecord,
    ExamplePair,
    ValueMap,
    ValueMapEntry,
)

__all__ = ["HoldoutMetrics", "validate_and_demote"]

_MIN_ROWS_FOR_HOLDOUT = 5


@dataclass
class HoldoutMetrics:
    """Aggregate replay quality measured on the holdout rows."""

    holdout_rows: int = 0
    proposed: int = 0
    correct: int = 0
    false_modifications: int = 0
    missed: int = 0
    per_rule: dict[str, dict[str, float]] = field(default_factory=dict)
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "holdout_rows": self.holdout_rows,
            "proposed": self.proposed,
            "correct": self.correct,
            "false_modifications": self.false_modifications,
            "missed": self.missed,
            "per_rule": {k: dict(v) for k, v in self.per_rule.items()},
            "skipped_reason": self.skipped_reason,
        }


def _split_positions(
    n: int, holdout_fraction: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    order = rng.permutation(n)
    n_holdout = max(1, int(round(n * holdout_fraction)))
    if n_holdout >= n:
        n_holdout = n - 1
    return order[n_holdout:], order[:n_holdout]  # train, holdout


def _subset(aligned: AlignedPair, positions: np.ndarray) -> AlignedPair:
    report = aligned.alignment_report
    sub_report = AlignmentReport(
        mode=report.mode,
        key=report.key,
        matched_rows=int(len(positions)),
        unmatched_messy=0,
        unmatched_clean=0,
        duplicate_messy_keys=0,
        duplicate_clean_keys=0,
        warnings=list(report.warnings),
    )
    return AlignedPair(
        aligned.messy_aligned.iloc[positions],
        aligned.clean_aligned.iloc[positions],
        sub_report,
    )


# ---------------------------------------------------------------------------
# Forward replay of candidate artifacts on a single cell
# ---------------------------------------------------------------------------


def _rule_predict(rule: ColumnConstraint, raw: object) -> object:
    """Apply one learned rule forward; return raw unchanged when no repair."""
    if not isinstance(raw, str):
        return raw
    params = rule.params
    if rule.rule == "valid_format" and params.get("format") == "email":
        repaired = repair_email(raw)
        return repaired[0] if repaired is not None else raw
    if rule.rule == "locale_format" and params.get("format") == "phone":
        normalized = normalize_phone_in(raw)
        return normalized if normalized is not None else raw
    if rule.rule == "valid_format" and params.get("format") == "date":
        parsed = _parse_date(raw, dayfirst=bool(params.get("dayfirst", False)))
        return parsed.date().isoformat() if parsed is not None else raw
    if rule.rule == "sentinel" and params.get("sentinels"):
        if raw.strip() in set(params["sentinels"]):
            return None
        return raw
    if rule.rule == "allowed_values" and params.get("values"):
        values = {str(v) for v in params["values"]}
        if raw in values:
            return raw
        norm = _norm_token(raw)
        matches = [v for v in values if _norm_token(v) == norm]
        if len(matches) == 1:
            return matches[0]
        return raw
    return raw  # advisory families do not replay literally


def _values_equal(a: object, b: object) -> bool:
    if a is None and (b is None or pd.isna(b)):
        return True
    if b is None and pd.isna(a):
        return True
    try:
        if bool(pd.isna(a)) and bool(pd.isna(b)):
            return True
    except (TypeError, ValueError):
        pass
    if a == b:
        return True
    return str(a) == str(b)


def _evaluate(
    candidate: ExtractionResult,
    holdout: AlignedPair,
) -> HoldoutMetrics:
    """Replay candidate artifacts on holdout messy cells and score them."""
    metrics = HoldoutMetrics(holdout_rows=len(holdout.messy_aligned))
    messy = holdout.messy_aligned
    clean = holdout.clean_aligned

    rules_by_column: dict[str, list[ColumnConstraint]] = {}
    for rule in candidate.rules:
        if rule.column and rule.enforcement != "advisory":
            rules_by_column.setdefault(rule.column, []).append(rule)

    columns = set(rules_by_column) | set(candidate.value_maps)
    for column in columns:
        if column not in messy.columns or column not in clean.columns:
            continue
        value_map: ValueMap | None = candidate.value_maps.get(column)
        lookup = value_map.lookup() if value_map is not None else {}
        for raw, target in zip(messy[column].tolist(), clean[column].tolist()):
            predicted = raw
            source: str | None = None
            if lookup and raw in lookup:
                predicted = lookup[raw].clean_value
                source = f"value_map:{column}"
            else:
                for rule in rules_by_column.get(column, []):
                    attempt = _rule_predict(rule, raw)
                    if not _values_equal(attempt, raw):
                        predicted = attempt
                        source = f"rule:{rule.id}"
                        break

            changed = not _values_equal(predicted, raw)
            needed = not _values_equal(raw, target)
            if changed:
                metrics.proposed += 1
                bucket = metrics.per_rule.setdefault(
                    source or f"column:{column}",
                    {"proposed": 0, "correct": 0, "false": 0},
                )
                bucket["proposed"] += 1
                if _values_equal(predicted, target):
                    metrics.correct += 1
                    bucket["correct"] += 1
                else:
                    metrics.false_modifications += 1
                    bucket["false"] += 1
            elif needed:
                metrics.missed += 1

    for bucket in metrics.per_rule.values():
        proposed = bucket["proposed"]
        bucket["precision"] = bucket["correct"] / proposed if proposed else 0.0
    return metrics


def _apply_demotions(
    final: ExtractionResult,
    metrics: HoldoutMetrics,
    min_precision: float,
    salt_examples: list[ExamplePair],
) -> list[DemotionRecord]:
    """Demote artifacts that failed holdout validation (matched by id)."""
    records: list[DemotionRecord] = []

    failing_rules = {
        source.split(":", 1)[1]
        for source, bucket in metrics.per_rule.items()
        if source.startswith("rule:") and bucket["precision"] < min_precision
    }
    failing_maps = {
        source.split(":", 1)[1]
        for source, bucket in metrics.per_rule.items()
        if source.startswith("value_map:") and bucket["precision"] < min_precision
    }

    kept_rules: list[ColumnConstraint] = []
    for rule in final.rules:
        if rule.id in failing_rules:
            demoted = ColumnConstraint(
                id=rule.id,
                column=rule.column,
                resolved_from=rule.resolved_from,
                resolution_confidence=rule.resolution_confidence,
                rule=rule.rule,
                action=rule.action,
                params=dict(rule.params),
                enforcement="advisory",
                provenance=rule.provenance,
            )
            kept_rules.append(demoted)
            precision = metrics.per_rule.get(f"rule:{rule.id}", {}).get("precision")
            records.append(
                DemotionRecord(
                    target=f"rule:{rule.id}",
                    column=rule.column or "",
                    family=str(rule.params.get("transform_family", rule.rule)),
                    outcome="suggest_only",
                    reason=(
                        f"holdout precision {precision:.3f} < {min_precision}"
                        if precision is not None
                        else "failed holdout validation"
                    ),
                    precision=precision,
                    support=rule.params.get("support"),
                )
            )
        else:
            kept_rules.append(rule)
    final.rules = kept_rules

    for column in list(final.value_maps):
        if column not in failing_maps:
            continue
        value_map = final.value_maps.pop(column)
        precision = metrics.per_rule.get(f"value_map:{column}", {}).get("precision")
        for entry in value_map.entries:
            salt_examples.append(
                ExamplePair(
                    column=column,
                    raw_value=entry.raw_value,
                    clean_value=entry.clean_value,
                    transform_family=entry.transform_family,
                    support=entry.support,
                    masked=entry.masked,
                )
            )
        records.append(
            DemotionRecord(
                target=f"value_map:{column}",
                column=column,
                family="value_map",
                outcome="demoted_to_example",
                reason=(
                    f"holdout precision {precision:.3f} < {min_precision}; "
                    "entries kept as examples only"
                    if precision is not None
                    else "failed holdout validation; entries kept as examples only"
                ),
                precision=precision,
            )
        )
        if final.memory is not None:
            final.memory.value_patterns.pop(column, None)
    return records


def _drop_low_precision_entries(
    final: ExtractionResult, min_precision: float
) -> list[DemotionRecord]:
    """Within-train precision gate for individual literal map entries."""
    records: list[DemotionRecord] = []
    for column in list(final.value_maps):
        value_map = final.value_maps[column]
        kept: list[ValueMapEntry] = []
        dropped: list[ValueMapEntry] = []
        for entry in value_map.entries:
            (kept if entry.precision >= min_precision or entry.masked else dropped).append(entry)
        for entry in dropped:
            records.append(
                DemotionRecord(
                    target=f"value_map:{column}:{entry.raw_value!r}",
                    column=column,
                    family=entry.transform_family,
                    outcome="dropped",
                    reason=(
                        f"training precision {entry.precision:.3f} < {min_precision} "
                        "(raw value maps to conflicting clean values)"
                    ),
                    precision=entry.precision,
                    support=entry.support,
                )
            )
        if dropped:
            value_map.entries = kept
            if final.memory is not None and column in final.memory.value_patterns:
                final.memory.value_patterns[column] = {
                    str(e.raw_value): e.clean_value for e in kept if not e.masked
                }
        if not value_map.entries:
            del final.value_maps[column]
    return records


def validate_and_demote(
    aligned: AlignedPair,
    *,
    extract_kwargs: Mapping[str, Any],
    min_precision: float = 0.98,
    holdout_fraction: float = 0.2,
    random_state: int = 0,
) -> tuple[ExtractionResult, list[DemotionRecord], HoldoutMetrics]:
    """Run extract on train rows, score on holdout, demote, refit on full data.

    Returns the final (full-data, demotion-applied) extraction, the demotion
    records, and the holdout metrics.  With too few rows for a meaningful
    split, validation is skipped and noted rather than silently trusted.
    """
    from .classify import classify_diffs  # noqa: PLC0415 - avoid cycle at import
    from .diff import compute_diff  # noqa: PLC0415

    def _extract(pair: AlignedPair) -> ExtractionResult:
        summary = compute_diff(pair)
        classified = classify_diffs(summary, clean_frame=pair.clean_aligned)
        return extract_artifacts(classified, pair, summary, **extract_kwargs)

    final = _extract(aligned)
    records = _drop_low_precision_entries(final, min_precision)

    n = len(aligned.messy_aligned)
    if not aligned.alignment_report.row_level or n < _MIN_ROWS_FOR_HOLDOUT:
        metrics = HoldoutMetrics(
            skipped_reason=(
                f"holdout validation skipped: only {n} aligned row(s) "
                f"(need >= {_MIN_ROWS_FOR_HOLDOUT})"
                if aligned.alignment_report.row_level
                else "holdout validation skipped: no row-level alignment"
            )
        )
        final.notes.append(metrics.skipped_reason or "")
        return final, records, metrics

    train_pos, holdout_pos = _split_positions(n, holdout_fraction, random_state)
    candidate = _extract(_subset(aligned, train_pos))
    _drop_low_precision_entries(candidate, min_precision)
    metrics = _evaluate(candidate, _subset(aligned, holdout_pos))

    demoted_examples: list[ExamplePair] = []
    records.extend(_apply_demotions(final, metrics, min_precision, demoted_examples))
    final.examples.extend(demoted_examples)
    return final, records, metrics
