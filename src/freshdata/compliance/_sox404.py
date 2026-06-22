"""SOX Section 404 — transformation-control evidence pack."""

from __future__ import annotations

from ._adapter import ComplianceContext
from ._base import GENERAL_CAVEAT, ComplianceConfig, FrameworkReport, new_entry_id

FRAMEWORK_KEY = "sox_404"
FRAMEWORK_NAME = "SOX-404 transformation control"

#: SOX-grade Data Trust Score gate.
_TRUST_GATE = 80.0
_LOW_CONFIDENCE = 0.6
_PREVENTIVE_TYPES = frozenset({"flag", "sentinel_normalize", "rename", "strip_whitespace"})

_ATTESTATION_CAVEAT = (
    "This artifact supports compliance workflows but is not itself a certified "
    "compliance system. Review by a qualified control assessor is required."
)


def generate_sox404(ctx: ComplianceContext, config: ComplianceConfig) -> FrameworkReport:
    transformation_log: list[dict] = []
    for index, action in enumerate(ctx.actions):
        control_type = "Preventive" if action.action_type in _PREVENTIVE_TYPES else "Detective"
        transformation_log.append(
            {
                "entry_id": f"SOX-T-{index:04d}",
                "column": action.column,
                "action_type": action.action_type,
                "rationale": action.rationale,
                "risk": action.risk,
                "confidence": action.confidence,
                "rows_affected": action.row_count,
                "original_preserved": action.original_captured,
                "control_type": control_type,
            }
        )

    high_risk_entries = [e for e in transformation_log if e["risk"] == "high"]
    high_risk_columns = sorted({e["column"] for e in high_risk_entries if e["column"]})
    low_confidence_count = sum(1 for a in ctx.actions if a.confidence < _LOW_CONFIDENCE)
    requires_manual_review = bool(high_risk_entries)

    score = ctx.trust_score
    if score is not None:
        trust_gate_passed: bool | None = score >= _TRUST_GATE
    else:
        trust_gate_passed = None

    warnings: list[str] = []
    errors: list[str] = []
    if trust_gate_passed is None:
        warnings.append(
            "No Data Trust Score available; the SOX trust gate was not evaluated. "
            "Supply an EnterpriseResult or ComplianceConfig.trust_score."
        )
    elif trust_gate_passed is False:
        errors.append(
            f"Data Trust Score {ctx.trust_score:.1f} is below the SOX-grade threshold "
            f"of {_TRUST_GATE:.0f}."
        )
    for entry in high_risk_entries:
        errors.append(
            f"High-risk transformation on {entry['column']!r} requires manual control "
            f"review: {entry['rationale'] or entry['action_type']}"
        )

    exception_summary = {
        "high_risk_count": len(high_risk_entries),
        "high_risk_columns": high_risk_columns,
        "low_confidence_count": low_confidence_count,
        "requires_manual_review": requires_manual_review,
    }

    data = {
        "pack_id": new_entry_id("SOX404"),
        "generated_utc": ctx.timestamp,
        "control_objective": (
            "Ensure accuracy, completeness, and traceability of data transformations "
            "used in financial reporting processes."
        ),
        "control_owner": config.system_actor,
        "control_type": "Preventive + Detective",
        "control_description": (
            "Per-column automated cleaning with explainable audit trail. Each "
            "transformation records the column, action type, rationale, risk level, "
            "confidence score, and rows affected. A Data Trust Score gates pipeline "
            "continuation."
        ),
        "data_trust_score": ctx.trust_score,
        "trust_score_gate_passed": trust_gate_passed,
        "input_dataframe_hash": config.input_dataframe_hash or "not_provided",
        "total_transformations": len(ctx.actions),
        "high_risk_transformations": high_risk_entries,
        "transformation_log": transformation_log,
        "exception_summary": exception_summary,
        "attestation": {
            "statement": (
                "This transformation control evidence pack was generated automatically "
                "by freshdata. It documents all data modifications applied to the input "
                "dataset, the rationale for each, and the resulting Data Trust Score. It "
                "is intended to support internal control assessment under SOX Section 404."
            ),
            "system_actor": config.system_actor,
            "timestamp_utc": ctx.timestamp,
            "caveat": _ATTESTATION_CAVEAT,
        },
        "caveat": GENERAL_CAVEAT,
    }

    passed = (trust_gate_passed is not False) and not requires_manual_review
    return FrameworkReport(
        framework_key=FRAMEWORK_KEY,
        framework_name=FRAMEWORK_NAME,
        passed=passed,
        warnings=warnings,
        errors=errors,
        data=data,
    )
