"""HL7 v2.x ER7 (pipe-delimited) parser.

Parses the common ADT/ORU segments — MSH (message header), PID (patient), PV1 (visit),
OBX (observation) — into three frames (``patient`` / ``encounter`` / ``observation``)
shaped for the healthcare domain pack. Observation code systems are mapped to their
canonical URIs (LOINC ``http://loinc.org``, SNOMED ``http://snomed.info/sct``, ICD-10).

Delimiters are read from each message's MSH segment (HL7 v2 Chapter 2): MSH-1 is the
field separator and MSH-2 holds the component, repetition, escape and subcomponent
characters (``^~\\&`` when absent). They apply to every segment up to the next MSH.
Single-valued output columns take the first repetition of a field; OBX-5 (observation
value) is a repeating field and keeps every repetition. Fields are split first, then the
delimiter escape sequences ``\\F\\ \\S\\ \\T\\ \\R\\ \\E\\`` are decoded to the literal
characters. Other escapes (``\\H\\``, ``\\N\\``, ``\\X..\\``, ``\\.br\\`` ...) are left
verbatim. Whole-field values (e.g. OBX-5) are re-emitted with the standard ``^`` / ``&``
/ ``~`` component / subcomponent / repetition separators, so output does not depend on a
message's custom delimiters.

This is a structural parser for the common segments, not a full HL7 v2 conformance
engine: unrecognized segments are counted in :attr:`ParseResult.warnings`, and the OBX
component layout follows the usual ORU convention.
"""

from __future__ import annotations

from typing import Any, NamedTuple

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
_PATIENT_CLASS = {
    "I": "inpatient",
    "O": "outpatient",
    "E": "emergency",
    "P": "preadmit",
    "R": "recurring",
    "B": "obstetrics",
}

_KNOWN_SEGMENTS = frozenset({"PID", "PV1", "OBR", "OBX"})


class _Delimiters(NamedTuple):
    """The five HL7 v2 message delimiters declared by MSH-1 / MSH-2."""

    field: str = "|"
    component: str = "^"
    repetition: str = "~"
    escape: str = "\\"
    subcomponent: str = "&"

    @classmethod
    def from_msh(cls, segment: str) -> _Delimiters:
        """Read MSH-1 (``segment[3]``) and MSH-2; missing characters use the defaults."""
        default = cls()
        if len(segment) < 4:
            return default
        field_sep = segment[3]
        encoding = segment[4:].split(field_sep, 1)[0]
        chars = [encoding[i] if i < len(encoding) else d for i, d in enumerate(default[1:])]
        return cls(field_sep, *chars)

    def decode(self, text: str) -> str:
        """Decode the delimiter escape sequences; leave any other escape verbatim."""
        esc = self.escape
        if esc not in text:
            return text
        literal = {
            "F": self.field,
            "S": self.component,
            "T": self.subcomponent,
            "R": self.repetition,
            "E": esc,
        }
        out: list[str] = []
        i = 0
        while True:
            start = text.find(esc, i)
            end = text.find(esc, start + 1) if start != -1 else -1
            if end == -1:
                out.append(text[i:])
                return "".join(out)
            out.append(text[i:start])
            code = text[start + 1 : end]
            out.append(literal.get(code, text[start : end + 1]))
            i = end + 1

    def value(self, field: str, *, repeating: bool = False) -> str:
        """Decoded *field* re-emitted with the standard ``^`` / ``&`` / ``~`` separators.

        Only the first repetition is kept unless *repeating* is true, in which case every
        repetition is kept and joined with ``~``. Splitting happens before escapes are
        decoded, so an escaped delimiter (e.g. ``\\R\\``) never splits the value.
        """
        reps = field.strip().split(self.repetition)
        return "~".join(
            "^".join(
                "&".join(self.decode(sub) for sub in comp.split(self.subcomponent))
                for comp in rep.split(self.component)
            ).strip()
            for rep in (reps if repeating else reps[:1])
        )

    def component_of(self, field: str, n: int) -> str:
        """1-based component *n* of the first repetition (``DOE^JOHN`` -> 1='DOE')."""
        if not field:
            return ""
        parts = field.strip().split(self.repetition, 1)[0].split(self.component)
        return self.decode(parts[n - 1]).strip() if 0 < n <= len(parts) else ""


def _field(fields: list[str], n: int) -> str:
    """Raw field *n* of a split segment (``fields[0]`` is the segment id)."""
    return fields[n] if n < len(fields) else ""


class HL7v2Parser(Parser):
    """Parse HL7 v2 ER7 messages into patient/encounter/observation frames."""

    format = "hl7v2"
    suggested_domain = "healthcare"

    def parse(self, source: Any) -> ParseResult:
        text = self.read_text(source)
        # HL7 segments are CR-separated; tolerate LF / CRLF too.
        segments = [
            s.lstrip()
            for s in text.replace("\r\n", "\r").replace("\n", "\r").split("\r")
            if s.strip()
        ]

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
        delim = _Delimiters()

        for seg in segments:
            seg_id = seg[:3]
            is_msh = seg_id == "MSH" and (len(seg) == 3 or not seg[3].isalnum())

            if is_msh:
                delim = _Delimiters.from_msh(seg)
                msg_index += 1
                current_pid = None
                current_order = None
                # MSH is offset by one (MSH-1 is the field separator itself), so
                # MSH-2 is fields[1] and MSH-9 (message type, e.g. "ADT^A01") is fields[8].
                message_type = delim.value(_field(seg.split(delim.field), 8))
                continue

            d = delim
            fields = seg.split(d.field)
            if seg_id not in _KNOWN_SEGMENTS or (len(seg) > 3 and seg[3] != d.field):
                key = fields[0].strip()
                unknown[key] = unknown.get(key, 0) + 1
            elif seg_id == "PID":
                current_pid = d.component_of(_field(fields, 3), 1) or f"MSG{msg_index}"
                patients.append(
                    {
                        "patient_id": current_pid,
                        "family_name": d.component_of(_field(fields, 5), 1),
                        "given_name": d.component_of(_field(fields, 5), 2),
                        "birth_date": d.value(_field(fields, 7)),
                        "gender": d.value(_field(fields, 8)),
                    }
                )
            elif seg_id == "PV1":
                class_code = d.value(_field(fields, 2))
                encounters.append(
                    {
                        "patient_id": current_pid or f"MSG{msg_index}",
                        "visit_number": d.component_of(_field(fields, 19), 1),
                        "class_code": class_code,
                        "class": _PATIENT_CLASS.get(class_code.upper(), class_code),
                        "location": d.component_of(_field(fields, 3), 1),
                    }
                )
            elif seg_id == "OBR":
                service = _field(fields, 4)
                current_order = (
                    d.component_of(_field(fields, 3), 1)
                    or d.component_of(_field(fields, 2), 1)
                    or None
                )
                orders.append(
                    {
                        "patient_id": current_pid or f"MSG{msg_index}",
                        "order_id": current_order,
                        "placer_order": d.component_of(_field(fields, 2), 1),
                        "filler_order": d.component_of(_field(fields, 3), 1),
                        "service_code": d.component_of(service, 1),
                        "service_display": d.component_of(service, 2),
                        "service_system": d.component_of(service, 3),
                        "observed_at": d.value(_field(fields, 7)),
                    }
                )
            else:  # OBX
                code = _field(fields, 3)
                system = d.component_of(code, 3)
                observations.append(
                    {
                        "patient_id": current_pid or f"MSG{msg_index}",
                        "order_id": current_order,
                        "code": d.component_of(code, 1),
                        "display": d.component_of(code, 2),
                        "code_system": _CODE_SYSTEMS.get(system.upper(), system),
                        # OBX-5 repeats: keep every value, joined with "~".
                        "value": d.value(_field(fields, 5), repeating=True),
                        "unit": d.value(_field(fields, 6)),
                        "status": d.value(_field(fields, 11)),
                        "observed_at": d.value(_field(fields, 14)),
                    }
                )

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
