"""21 CFR §11.10(e) — computer-generated, time-stamped audit trails."""

from __future__ import annotations

from ._adapter import ComplianceContext, NormalizedAction
from ._base import (
    GENERAL_CAVEAT,
    ComplianceConfig,
    FrameworkReport,
    new_entry_id,
)

FRAMEWORK_KEY = "21cfr_11"
FRAMEWORK_NAME = "21 CFR §11.10(e)"

#: Verbatim regulatory text the audit trail is designed to satisfy.
CONTROL_TEXT = (
    "Use of computer-generated, time-stamped audit trails to independently "
    "record the date and time of operator entries and actions that create, "
    "modify, or delete electronic records. Record changes shall not obscure "
    "previously recorded information and computer-generated audit trails shall "
    "be retained for a period at least as long as that required for the subject "
    "electronic records."
)


def _is_obscuring(action: NormalizedAction, *, strict: bool) -> bool:
    """Whether an action could obscure previously recorded information.

    Deletions are inherently non-obscuring once logged, and imputation/
    structural/intentional-mask changes retain (or never had) a prior value. A
    value overwrite with no retained pre-image (``not_captured``) always
    obscures. Normalising rewrites (whitespace trim, sentinel canonicalisation)
    obscure only when ``strict`` is set.
    """
    if action.record_action_type == "DELETE":
        return False
    if action.original_basis == "not_captured":
        return True
    return strict and action.original_basis == "normalized"


def generate_21cfr11(ctx: ComplianceContext, config: ComplianceConfig) -> FrameworkReport:
    """Emit a session header plus one audit entry per normalized action."""
    operator_id = config.operator_id or "system"
    strict = config.strict_cfr_normalization
    warnings: list[str] = []
    audit_entries: list[dict] = []

    for action in ctx.actions:
        obscuring = _is_obscuring(action, strict=strict)
        audit_entries.append(
            {
                "entry_id": new_entry_id("CFR11"),
                "session_id": ctx.session_id,
                "timestamp_utc": ctx.timestamp,
                "system_actor": config.system_actor,
                "operator_id": operator_id,
                "record_action_type": action.record_action_type,
                "column_affected": action.column,
                "column_role": action.column_role,
                "rows_affected": action.row_count,
                "original_value_class": "not_captured" if obscuring else "preserved",
                "original_value_basis": action.original_basis,
                "change_description": action.rationale or action.description,
                "risk_level": action.risk,
                "confidence": action.confidence,
                "non_obscuring_guarantee": not obscuring,
                "retention_days": config.retention_days,
            }
        )
        # A MODIFY that overwrote an existing value without retaining a pre-image
        # is the only way the audit trail could obscure prior information.
        if obscuring:
            warnings.append(
                f"{action.record_action_type} on {action.column!r} did not retain a "
                f"pre-image ({action.action_type}); prior value not independently recorded."
            )

    session_header = {
        "session_id": ctx.session_id,
        "session_start_utc": ctx.timestamp,
        "system_actor": config.system_actor,
        "operator_id": operator_id,
        "total_actions": len(audit_entries),
        "data_trust_score": ctx.trust_score,
        "framework": FRAMEWORK_NAME,
        "retention_days": config.retention_days,
    }

    passed = not warnings
    data = {
        "regulation": "21 CFR §11.10(e)",
        "control_text": CONTROL_TEXT,
        "session_header": session_header,
        "audit_entries": audit_entries,
        "caveat": GENERAL_CAVEAT,
    }
    return FrameworkReport(
        framework_key=FRAMEWORK_KEY,
        framework_name=FRAMEWORK_NAME,
        passed=passed,
        warnings=warnings,
        errors=[],
        data=data,
    )
