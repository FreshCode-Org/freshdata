"""GDPR Article 30 (record of processing) + Article 17 (erasure log)."""

from __future__ import annotations

from ._adapter import ComplianceContext
from ._base import (
    GENERAL_CAVEAT,
    ComplianceConfig,
    FrameworkReport,
    new_entry_id,
)

FRAMEWORK_KEY = "gdpr_30"
FRAMEWORK_NAME = "GDPR Article 30 + Article 17"

_ERASURE_TYPE = {
    "drop_col": "DROP_COLUMN",
    "drop_row": "DROP_ROW",
    "pii_mask": "PII_ANONYMISE",
}
_GROUNDS = {
    "drop_col": "Art.5(1)(c) data minimisation",
    "drop_row": "Art.5(1)(e) storage limitation (empty/irreparable record)",
    "pii_mask": "Art.5(1)(f) integrity and confidentiality",
}
_SECURITY_MEASURES = [
    "Hash-salt PII masking (SHA-256 + salt)",
    "Audit trail generation",
    "Access-gated Data Trust Score",
]


def _personal_data_categories(ctx: ComplianceContext) -> list[str]:
    """Columns carrying personal data: PII-masked or inferred PII role."""
    columns: set[str] = set(ctx.masked_columns)
    for action in ctx.actions:
        if action.column and (action.action_type == "pii_mask" or action.column_role == "pii"):
            columns.add(action.column)
    return sorted(columns)


def generate_gdpr(ctx: ComplianceContext, config: ComplianceConfig) -> FrameworkReport:
    personal_data_categories = _personal_data_categories(ctx)

    article_30 = {
        "record_id": new_entry_id("GDPR30"),
        "generated_utc": ctx.timestamp,
        "controller_name": config.controller_name or "Not specified",
        "controller_contact": config.controller_contact or "Not specified",
        "processing_activity": "Automated data cleaning and quality remediation",
        "processing_purpose": config.processing_purpose or "Data quality improvement",
        "legal_basis": config.legal_basis,
        "data_subject_categories": list(config.data_subject_categories),
        "personal_data_categories": personal_data_categories,
        # GDPR Art. 9 special categories (health/financial/etc.), inferred from the
        # source frame when supplied; informational, not identifiers.
        "special_category_columns": sorted(ctx.domain_sensitive_columns),
        "recipients": [],
        "recipients_note": (
            "No automated disclosure; downstream access governed by data controller."
        ),
        "third_country_transfers": False,
        "third_country_transfers_note": ("freshdata processes in-memory; no network transfer."),
        "retention_days": config.retention_days,
        "security_measures": list(_SECURITY_MEASURES),
        "automated_decision_making": True,
        "automated_decision_making_note": ("Column-level cleaning decisions are fully automated."),
        "safeguards": (
            "Role-gated decision engine: targets never modified; IDs never imputed. "
            "Confidence and risk scores logged per action."
        ),
    }

    erasure_log: list[dict] = []
    for action in ctx.actions:
        if action.action_type not in _ERASURE_TYPE:
            continue
        erasure_log.append(
            {
                "erasure_event_id": new_entry_id("GDPR17"),
                "timestamp_utc": ctx.timestamp,
                "column_name": action.column,
                "erasure_type": _ERASURE_TYPE[action.action_type],
                "rows_affected": action.row_count,
                "grounds_for_erasure": _GROUNDS[action.action_type],
                "confirmation": ("Action logged; original values not retained in output dataset"),
            }
        )

    warnings: list[str] = []
    if not personal_data_categories and config.data_subject_categories:
        warnings.append(
            "data_subject_categories were declared but no personal-data columns were "
            "detected; PII may be present but unmasked/undetected. Supply a source "
            "DataFrame or masked_columns for accurate Article 30 categorisation."
        )

    data = {
        "article_30": article_30,
        "article_17": {
            "right": "Art.17 GDPR — right to erasure",
            "total_erasures": len(erasure_log),
            "erasure_log": erasure_log,
        },
        "erasure_log": erasure_log,  # top-level alias for FrameworkReport.to_frame()
        "caveat": GENERAL_CAVEAT,
    }
    return FrameworkReport(
        framework_key=FRAMEWORK_KEY,
        framework_name=FRAMEWORK_NAME,
        passed=True,
        warnings=warnings,
        errors=[],
        data=data,
    )
