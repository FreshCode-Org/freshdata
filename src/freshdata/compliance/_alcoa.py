"""ALCOA+ attestation (FDA 2018 Data Integrity Q&A, MHRA 2018, WHO TRS 996)."""

from __future__ import annotations

from ._adapter import ComplianceContext
from ._base import GENERAL_CAVEAT, ComplianceConfig, FrameworkReport

FRAMEWORK_KEY = "alcoa_plus"
FRAMEWORK_NAME = "ALCOA+"

_ORIGINAL_THRESHOLD = 0.8
_ACCURATE_THRESHOLD = 0.7

PRINCIPLES = (
    "Attributable",
    "Legible",
    "Contemporaneous",
    "Original",
    "Accurate",
    "Complete",
    "Consistent",
    "Enduring",
    "Available",
)


def _mean_confidence(ctx: ComplianceContext) -> float:
    confidences = [a.confidence for a in ctx.actions]
    if not confidences:
        return 1.0
    return round(sum(confidences) / len(confidences), 4)


def generate_alcoa(ctx: ComplianceContext, config: ComplianceConfig) -> FrameworkReport:
    operator_id = config.operator_id or "system"
    warnings: list[str] = []
    errors: list[str] = []
    principles: dict[str, dict] = {}

    # Attributable — the only principle that can hard-fail the attestation.
    attributable_ok = bool(config.system_actor) and bool(operator_id)
    principles["Attributable"] = {
        "principle": "Attributable",
        "passed": attributable_ok,
        "evidence": {"system_actor": config.system_actor, "operator_id": operator_id},
    }
    if not attributable_ok:
        errors.append("Attributable: system_actor and operator_id must both be set.")

    # Legible — every action carries a human-readable rationale.
    missing_rationale = [a.column for a in ctx.actions if not a.rationale]
    legible_ok = not missing_rationale
    principles["Legible"] = {
        "principle": "Legible",
        "passed": legible_ok,
        "evidence": {
            "rationale_present_for_all_actions": legible_ok,
            "actions_missing_rationale": missing_rationale,
        },
    }
    if not legible_ok:
        warnings.append(
            f"Legible: {len(missing_rationale)} action(s) lack a rationale "
            "(non-engine representation repairs are deterministic but unannotated)."
        )

    # Contemporaneous — recorded at generation time, UTC.
    principles["Contemporaneous"] = {
        "principle": "Contemporaneous",
        "passed": True,
        "evidence": {"timestamp_utc": ctx.timestamp},
    }

    # Original — pre-images preserved for value-modifying actions.
    modify_actions = [a for a in ctx.actions if a.record_action_type == "MODIFY"]
    captured = sum(1 for a in modify_actions if a.original_captured)
    original_ratio = (captured / len(modify_actions)) if modify_actions else 1.0
    original_ok = original_ratio >= _ORIGINAL_THRESHOLD
    principles["Original"] = {
        "principle": "Original",
        "passed": original_ok,
        "evidence": {
            "original_values_captured": captured,
            "modify_action_count": len(modify_actions),
            "ratio": round(original_ratio, 4),
            "threshold": _ORIGINAL_THRESHOLD,
        },
    }
    if not original_ok:
        warnings.append(
            f"Original: only {original_ratio:.0%} of value-modifying actions retain a "
            f"pre-image (threshold {_ORIGINAL_THRESHOLD:.0%})."
        )

    # Accurate — mean engine confidence.
    mean_confidence = _mean_confidence(ctx)
    accurate_ok = mean_confidence >= _ACCURATE_THRESHOLD
    principles["Accurate"] = {
        "principle": "Accurate",
        "passed": accurate_ok,
        "evidence": {"mean_confidence": mean_confidence, "threshold": _ACCURATE_THRESHOLD},
    }
    if not accurate_ok:
        warnings.append(
            f"Accurate: mean confidence {mean_confidence} is below {_ACCURATE_THRESHOLD}."
        )

    # Complete — every known column was addressed.
    columns_with_actions = sorted({a.column for a in ctx.actions if a.column})
    verifiable = bool(ctx.all_columns)
    columns_without_actions = (
        [c for c in ctx.all_columns if c not in set(columns_with_actions)] if verifiable else []
    )
    complete_ok = not columns_without_actions
    principles["Complete"] = {
        "principle": "Complete",
        "passed": complete_ok,
        "evidence": {
            "columns_with_actions": columns_with_actions,
            "all_columns": list(ctx.all_columns),
            "columns_without_actions": columns_without_actions,
            "coverage_verifiable": verifiable,
        },
    }
    if not complete_ok:
        warnings.append(
            f"Complete: columns with no recorded action: {', '.join(columns_without_actions)}."
        )

    # Consistent / Enduring / Available — structural guarantees.
    principles["Consistent"] = {
        "principle": "Consistent",
        "passed": True,
        "evidence": {"single_timezone": "UTC", "timestamp_format": "ISO 8601"},
    }
    principles["Enduring"] = {
        "principle": "Enduring",
        "passed": True,
        "evidence": {"export_formats": ["json", "csv", "dict"]},
    }
    principles["Available"] = {
        "principle": "Available",
        "passed": True,
        "evidence": {"api_methods": [".to_dict()", ".to_json()", ".to_frame()"]},
    }

    action_evidence = [
        {
            "action_index": index,
            "column": action.column,
            "action_type": action.action_type,
            "attributable_to": config.system_actor,
            "rationale": action.rationale,
            "timestamp_utc": ctx.timestamp,
            "original_captured": action.original_captured,
            "confidence": action.confidence,
            "risk": action.risk,
        }
        for index, action in enumerate(ctx.actions)
    ]

    data = {
        "standard": "ALCOA+ data integrity principles",
        "principles": principles,
        "action_evidence": action_evidence,
        "caveat": GENERAL_CAVEAT,
    }
    return FrameworkReport(
        framework_key=FRAMEWORK_KEY,
        framework_name=FRAMEWORK_NAME,
        passed=not errors,
        warnings=warnings,
        errors=errors,
        data=data,
    )
