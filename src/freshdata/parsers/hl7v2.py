"""HL7 v2.x ER7 (pipe-delimited) parser.

Parses the common ADT/ORU segments — MSH (message header), PID (patient), PV1 (visit),
OBX (observation) — into three frames (``patient`` / ``encounter`` / ``observation``)
shaped for the healthcare domain pack. Observation code systems are mapped to their
canonical URIs (LOINC ``http://loinc.org``, SNOMED ``http://snomed.info/sct``, ICD-10).

This is a structural parser for the common segments, not a full HL7 v2 conformance
engine: unrecognized segments are counted in :attr:`ParseResult.warnings`, and the OBX
component layout follows the usual ORU convention.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Parser, ParseResult

# Common HL7 coding-system identifiers -> canonical URI / label.
_CODE_SYSTEMS = {
    "LN": "http://loinc.org",
    "LOINC": "http://loinc.org",
    "SCT": "http://snomed.info/sct",
    "SNM": "http://snomed.info/sct",
    "SNOMED": "http://snomed.info/sct",
    "I10": "ICD-10",
    "ICD10": "ICD-10",
    "ICD-10": "ICD-10",
}

# PV1-2 patient class -> human label.
_PATIENT_CLASS = {"I": "inpatient", "O": "outpatient", "E": "emergency",
                  "P": "preadmit", "R": "recurring", "B": "obstetrics"}


def _comp(field: str, n: int) -> str:
    """1-based HL7 component *n* of a field (``DOE^JOHN`` -> 1='DOE')."""
    if not field:
        return ""
    parts = field.split("^")
    return parts[n - 1].strip() if 0 < n <= len(parts) else ""


def _field(fields: list[str], n: int) -> str:
    """Field *n* of a split segment (``fields[0]`` is the segment id)."""
    return fields[n].strip() if n < len(fields) else ""


class HL7v2Parser(Parser):
    """Parse HL7 v2 ER7 messages into patient/encounter/observation frames."""

    format = "hl7v2"
    suggested_domain = "healthcare"

    def parse(self, source: Any) -> ParseResult:
        text = self.read_text(source)
        # HL7 segments are CR-separated; tolerate LF / CRLF too.
        segments = [s for s in text.replace("\r\n", "\r").replace("\n", "\r").split("\r")
                    if s.strip()]

        patients: list[dict[str, Any]] = []
        encounters: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        orders: list[dict[str, Any]] = []
        warnings: list[str] = []
        unknown: dict[str, int] = {}

        msg_index = 0
        current_pid: str | None = None
        current_order: str | None = None
        message_type = ""

        for seg in segments:
            fields = seg.split("|")
            seg_id = fields[0].strip()

            if seg_id == "MSH":
                msg_index += 1
                current_pid = None
                current_order = None
                # MSH is offset by one (MSH-1 is the field separator itself), so
                # MSH-9 (message type, e.g. "ADT^A01") is fields[8].
                message_type = fields[8].strip() if len(fields) > 8 else ""
            elif seg_id == "PID":
                current_pid = _comp(_field(fields, 3), 1) or f"MSG{msg_index}"
                patients.append({
                    "patient_id": current_pid,
                    "family_name": _comp(_field(fields, 5), 1),
                    "given_name": _comp(_field(fields, 5), 2),
                    "birth_date": _field(fields, 7),
                    "gender": _field(fields, 8),
                })
            elif seg_id == "PV1":
                encounters.append({
                    "patient_id": current_pid or f"MSG{msg_index}",
                    "visit_number": _comp(_field(fields, 19), 1),
                    "class_code": _field(fields, 2),
                    "class": _PATIENT_CLASS.get(_field(fields, 2).upper(), _field(fields, 2)),
                    "location": _comp(_field(fields, 3), 1),
                })
            elif seg_id == "OBR":
                service = _field(fields, 4)
                current_order = (_comp(_field(fields, 3), 1)
                                 or _comp(_field(fields, 2), 1) or None)
                orders.append({
                    "patient_id": current_pid or f"MSG{msg_index}",
                    "order_id": current_order,
                    "placer_order": _comp(_field(fields, 2), 1),
                    "filler_order": _comp(_field(fields, 3), 1),
                    "service_code": _comp(service, 1),
                    "service_display": _comp(service, 2),
                    "service_system": _comp(service, 3),
                    "observed_at": _field(fields, 7),
                })
            elif seg_id == "OBX":
                system = _comp(_field(fields, 3), 3)
                observations.append({
                    "patient_id": current_pid or f"MSG{msg_index}",
                    "order_id": current_order,
                    "code": _comp(_field(fields, 3), 1),
                    "display": _comp(_field(fields, 3), 2),
                    "code_system": _CODE_SYSTEMS.get(system.upper(), system),
                    "value": _field(fields, 5),
                    "unit": _field(fields, 6),
                    "status": _field(fields, 11),
                    "observed_at": _field(fields, 14),
                })
            else:
                unknown[seg_id] = unknown.get(seg_id, 0) + 1

        if unknown:
            listed = ", ".join(f"{k}({v})" for k, v in sorted(unknown.items()))
            warnings.append(f"skipped unrecognized segment types: {listed}")
        if msg_index == 0:
            warnings.append("no MSH header found; input may not be HL7 v2 ER7")

        return ParseResult(
            format=self.format,
            frames={
                "patient": pd.DataFrame(patients),
                "encounter": pd.DataFrame(encounters),
                "order": pd.DataFrame(orders),
                "observation": pd.DataFrame(observations),
            },
            suggested_domain=self.suggested_domain,
            metadata={"messages": msg_index, "message_type": message_type},
            warnings=warnings,
        )
