"""FHIR R4 JSON parser.

Parses a FHIR R4 Bundle, a single resource, a list of resources, a JSON string, or a
``pathlib.Path`` into flattened DataFrames keyed by resource: ``patient``,
``observation``, ``encounter``, ``condition``, ``medication_request``. The flattened
columns line up with the healthcare domain pack's resource validators, so a parsed frame
can go straight into ``fd.clean(frame, domain="healthcare")``.

Only predictable R4 fields are flattened; resource types the parser does not handle are
counted and surfaced in :attr:`ParseResult.warnings` rather than dropped silently.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd

from .base import Parser, ParseResult


def _as_list(value: Any) -> list[Any]:
    """Coerce a FHIR repeating element: a dict becomes ``[dict]``, a non-list ``[]``."""
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce a FHIR complex element: a list yields its first item if that is a dict."""
    items = _as_list(value)
    return items[0] if items and isinstance(items[0], dict) else {}


def _coding(concept: Any) -> tuple[Any, Any, Any]:
    """Return ``(system, code, display)`` from a FHIR CodeableConcept's first coding."""
    if not isinstance(concept, dict):
        return None, None, None
    first = _as_dict(concept.get("coding"))
    return first.get("system"), first.get("code"), first.get("display") or concept.get("text")


def _coding_code(concept: Any) -> Any:
    return _coding(concept)[1]


def _ref_id(reference: Any) -> Any:
    """Logical id of a FHIR reference (``{"reference": "Patient/123"}`` -> ``"123"``)."""
    if not isinstance(reference, dict):
        return None
    ref = reference.get("reference")
    return ref.split("/")[-1] if isinstance(ref, str) and ref else None


def _flatten_patient(r: dict[str, Any]) -> dict[str, Any]:
    addr = _as_dict(r.get("address"))
    deceased_dt = r.get("deceasedDateTime")
    deceased = r.get("deceasedBoolean")
    if deceased is None and deceased_dt is not None:
        deceased = True
    return {
        "patient_id": r.get("id"),
        "birth_date": r.get("birthDate"),
        "gender": r.get("gender"),
        "deceased": deceased,
        "deceased_date": deceased_dt,
        "marital_status": _coding_code(r.get("maritalStatus")),
        "address_postal_code": addr.get("postalCode"),
        "address_country": addr.get("country"),
    }


def _flatten_observation(r: dict[str, Any]) -> dict[str, Any]:
    system, code, display = _coding(r.get("code"))
    vq = _as_dict(r.get("valueQuantity"))
    interpretation = _as_dict(r.get("interpretation"))
    return {
        "observation_id": r.get("id"),
        "patient_id": _ref_id(r.get("subject")),
        "status": r.get("status"),
        "code_system": system,
        "code_value": code,
        "display": display,
        "effective_date": (r.get("effectiveDateTime")
                           or _as_dict(r.get("effectivePeriod")).get("start")),
        "value_quantity": vq.get("value"),
        "value_unit": vq.get("unit") or vq.get("code"),
        "value_string": r.get("valueString"),
        "interpretation": _coding_code(interpretation),
    }


def _flatten_encounter(r: dict[str, Any]) -> dict[str, Any]:
    cls = _as_dict(r.get("class"))
    period = _as_dict(r.get("period"))
    rsystem, rcode, _ = _coding(_as_dict(r.get("reasonCode")))
    hosp = _as_dict(r.get("hospitalization"))
    return {
        "encounter_id": r.get("id"),
        "patient_id": _ref_id(r.get("subject")),
        "status": r.get("status"),
        "class_code": cls.get("code"),
        "period_start": period.get("start"),
        "period_end": period.get("end"),
        "reason_code": rcode,
        "reason_code_system": rsystem,
        "hospitalization_admit_source": _coding_code(hosp.get("admitSource")),
        "service_type": _coding_code(r.get("serviceType")),
    }


def _flatten_condition(r: dict[str, Any]) -> dict[str, Any]:
    system, code, display = _coding(r.get("code"))
    return {
        "condition_id": r.get("id"),
        "patient_id": _ref_id(r.get("subject")),
        "clinical_status": _coding_code(r.get("clinicalStatus")),
        "verification_status": _coding_code(r.get("verificationStatus")),
        "category": _coding_code(_as_dict(r.get("category"))),
        "code_system": system,
        "code_value": code,
        "display": display,
        "onset_date": r.get("onsetDateTime") or _as_dict(r.get("onsetPeriod")).get("start"),
        "recorded_date": r.get("recordedDate"),
    }


def _flatten_medication_request(r: dict[str, Any]) -> dict[str, Any]:
    system, code, display = _coding(r.get("medicationCodeableConcept"))
    return {
        "medication_request_id": r.get("id"),
        "patient_id": _ref_id(r.get("subject")),
        "status": r.get("status"),
        "intent": r.get("intent"),
        "medication_system": system,
        "medication_code": code,
        "medication_display": display,
        "authored_on": r.get("authoredOn"),
        "requester": _ref_id(r.get("requester")),
    }


# resourceType -> (frame name, flattener)
_FLATTENERS = {
    "Patient": ("patient", _flatten_patient),
    "Observation": ("observation", _flatten_observation),
    "Encounter": ("encounter", _flatten_encounter),
    "Condition": ("condition", _flatten_condition),
    "MedicationRequest": ("medication_request", _flatten_medication_request),
}
_FRAME_NAMES = [frame for frame, _ in _FLATTENERS.values()]


class FHIRParser(Parser):
    """Parse FHIR R4 JSON (Bundle / resource / list) into flattened resource frames."""

    format = "fhir"
    suggested_domain = "healthcare"

    def parse(self, source: Any) -> ParseResult:
        warnings: list[str] = []
        try:
            data = self._load_json(source)
        except (ValueError, TypeError) as exc:
            return ParseResult(self.format, {name: pd.DataFrame() for name in _FRAME_NAMES},
                               self.suggested_domain, {}, [f"invalid FHIR JSON: {exc}"])

        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in _FRAME_NAMES}
        counts: Counter[str] = Counter()
        unsupported: Counter[str] = Counter()
        bundle_id = bundle_type = None

        if isinstance(data, dict) and data.get("resourceType") == "Bundle":
            bundle_id = data.get("id")
            bundle_type = data.get("type")

        for res in self._iter_resources(data, warnings):
            rtype = res.get("resourceType")
            if not isinstance(rtype, str):
                warnings.append(
                    f"resource with non-string resourceType ({type(rtype).__name__}) skipped")
                continue
            counts[rtype] += 1
            handler = _FLATTENERS.get(rtype)
            if handler is None:
                unsupported[rtype] += 1
                continue
            frame_name, flatten = handler
            try:
                row = flatten(res)
            except (AttributeError, KeyError, TypeError, IndexError, ValueError) as exc:
                warnings.append(
                    f"{rtype} resource {res.get('id')!r} skipped: malformed "
                    f"({type(exc).__name__})")
                continue
            rows[frame_name].append(row)

        if unsupported:
            listed = ", ".join(f"{k}({v})" for k, v in sorted(unsupported.items()))
            warnings.append(f"skipped unsupported resource types: {listed}")
        if not counts:
            warnings.append("no FHIR resources found")

        return ParseResult(
            format=self.format,
            frames={name: pd.DataFrame(data_rows) for name, data_rows in rows.items()},
            suggested_domain=self.suggested_domain,
            metadata={
                "resource_types": dict(counts),
                "bundle_id": bundle_id,
                "bundle_type": bundle_type,
                "total_resources": sum(counts.values()),
            },
            warnings=warnings,
        )

    def _load_json(self, source: Any) -> Any:
        """Return parsed JSON, accepting a dict/list directly or a Path/text/bytes."""
        if isinstance(source, (dict, list)):
            return source
        return json.loads(self.read_text(source))

    def _iter_resources(self, data: Any, warnings: list[str]):
        """Yield resource dicts from a Bundle, a single resource, or a list."""
        if isinstance(data, dict) and data.get("resourceType") == "Bundle":
            for entry in _as_list(data.get("entry")):
                resource = entry.get("resource") if isinstance(entry, dict) else None
                if isinstance(resource, dict):
                    yield resource
        elif isinstance(data, dict) and data.get("resourceType"):
            yield data
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("resourceType"):
                    yield item
                else:
                    warnings.append("list item without a resourceType skipped")
        else:
            warnings.append("input is not a FHIR Bundle, resource, or list of resources")
