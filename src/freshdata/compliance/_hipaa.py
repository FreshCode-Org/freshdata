"""HIPAA Safe Harbor — 18-identifier coverage (45 CFR §164.514(b)(2)).

Detection is a *column-name heuristic only*: no cell values are ever scanned.
"""

from __future__ import annotations

from typing import Any

from ._adapter import ComplianceContext
from ._base import ComplianceConfig, ComplianceGapError, FrameworkReport, new_entry_id

FRAMEWORK_KEY = "hipaa_safe_harbor"
FRAMEWORK_NAME = "HIPAA Safe Harbor"

_HIPAA_CAVEAT = (
    "Detection is column-name heuristic only. A qualified privacy officer must "
    "verify that all 18 identifiers are absent or appropriately de-identified "
    "before asserting HIPAA Safe Harbor compliance. This report does not "
    "constitute a legal determination of de-identification."
)

#: The 18 Safe Harbor identifiers (45 CFR §164.514(b)(2)).
HIPAA_IDENTIFIERS: dict[str, dict[str, Any]] = {
    "names": {
        "id": 1,
        "description": "Names",
        "detection_hints": ["name", "first_name", "last_name", "full_name", "patient_name"],
    },
    "geographic": {
        "id": 2,
        "description": "Geographic subdivisions smaller than state",
        "detection_hints": ["address", "street", "city", "county", "zip", "postal"],
    },
    "dates": {
        "id": 3,
        "description": (
            "All elements of dates except year (for individuals >89, include age and "
            "all date elements)"
        ),
        "detection_hints": [
            "dob",
            "date_of_birth",
            "birth_date",
            "admission_date",
            "discharge_date",
            "death_date",
            "date",
        ],
    },
    "phone": {
        "id": 4,
        "description": "Telephone numbers",
        "detection_hints": ["phone", "telephone", "tel", "mobile", "cell"],
    },
    "fax": {
        "id": 5,
        "description": "Fax numbers",
        "detection_hints": ["fax"],
    },
    "email": {
        "id": 6,
        "description": "Email addresses",
        "detection_hints": ["email", "e_mail", "email_address"],
    },
    "ssn": {
        "id": 7,
        "description": "Social security numbers",
        "detection_hints": ["ssn", "social_security", "sin", "tax_id"],
    },
    "medical_record": {
        "id": 8,
        "description": "Medical record numbers",
        "detection_hints": ["mrn", "medical_record", "patient_id", "record_number"],
    },
    "health_plan": {
        "id": 9,
        "description": "Health plan beneficiary numbers",
        "detection_hints": ["beneficiary", "health_plan", "member_id", "plan_id"],
    },
    "account": {
        "id": 10,
        "description": "Account numbers",
        "detection_hints": ["account", "account_number", "acct"],
    },
    "certificate_license": {
        "id": 11,
        "description": "Certificate/license numbers",
        "detection_hints": ["license", "certificate", "cert_number", "lic_number"],
    },
    "vehicle_identifiers": {
        "id": 12,
        "description": "Vehicle identifiers and serial numbers, including license plates",
        "detection_hints": ["vin", "license_plate", "vehicle_id", "plate"],
    },
    "device_identifiers": {
        "id": 13,
        "description": "Device identifiers and serial numbers",
        "detection_hints": ["device_id", "serial_number", "imei", "mac_address"],
    },
    "urls": {
        "id": 14,
        "description": "Web universal resource locators (URLs)",
        "detection_hints": ["url", "website", "web_address", "link"],
    },
    "ip_addresses": {
        "id": 15,
        "description": "Internet protocol (IP) addresses",
        "detection_hints": ["ip", "ip_address", "ipv4", "ipv6"],
    },
    "biometric": {
        "id": 16,
        "description": "Biometric identifiers, including finger and voice prints",
        "detection_hints": ["fingerprint", "biometric", "voiceprint", "retina"],
    },
    "photos": {
        "id": 17,
        "description": "Full-face photographs and any comparable images",
        "detection_hints": ["photo", "image", "photograph", "face", "picture"],
    },
    "other_unique": {
        "id": 18,
        "description": "Any other unique identifying number, characteristic, or code",
        "detection_hints": ["uid", "unique_id", "identifier", "patient_code"],
    },
}


def _known_columns(ctx: ComplianceContext) -> list[str]:
    columns: set[str] = set(ctx.all_columns) | set(ctx.masked_columns)
    columns.update(a.column for a in ctx.actions if a.column)
    return sorted(columns)


def generate_hipaa(ctx: ComplianceContext, config: ComplianceConfig) -> FrameworkReport:
    known_columns = _known_columns(ctx)
    masked = set(ctx.masked_columns)

    identifier_coverage: dict[str, dict] = {}
    addressed = detected_not_addressed = not_detected = 0
    gaps: list[str] = []

    for key, spec in HIPAA_IDENTIFIERS.items():
        hints = spec["detection_hints"]
        columns_found = [
            col for col in known_columns if any(hint in col.lower() for hint in hints)
        ]
        columns_masked = [col for col in columns_found if col in masked]

        if not columns_found:
            status = "not_detected"
            not_detected += 1
        elif set(columns_found) <= masked:
            status = "addressed"
            addressed += 1
        else:
            status = "detected_not_addressed"
            detected_not_addressed += 1
            gaps.append(key)

        identifier_coverage[key] = {
            "identifier": key,
            "id": spec["id"],
            "description": spec["description"],
            "status": status,
            "columns_found": columns_found,
            "columns_masked": columns_masked,
        }

    total = len(HIPAA_IDENTIFIERS)
    summary = {
        "total_identifiers": total,
        "addressed": addressed,
        "detected_not_addressed": detected_not_addressed,
        "not_detected": not_detected,
        "coverage_pct": round(addressed / total * 100, 4),
    }

    if gaps and config.fail_on_hipaa_gap:
        raise ComplianceGapError(f"HIPAA Safe Harbor gaps detected: {gaps}")

    errors = [f"Identifier {key!r} detected but not addressed (no PII masking)." for key in gaps]

    data = {
        "report_id": new_entry_id("HIPAA"),
        "generated_utc": ctx.timestamp,
        "standard": "HIPAA Safe Harbor — 45 CFR §164.514(b)(2)",
        "identifier_coverage": identifier_coverage,
        "summary": summary,
        "gaps": gaps,
        "caveat": _HIPAA_CAVEAT,
    }
    return FrameworkReport(
        framework_key=FRAMEWORK_KEY,
        framework_name=FRAMEWORK_NAME,
        passed=not gaps,
        warnings=[],
        errors=errors,
        data=data,
    )
