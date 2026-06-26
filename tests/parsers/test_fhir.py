"""Tests for the FHIR R4 JSON parser."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import freshdata as fd
from freshdata.parsers.fhir import (
    FHIRParser,
    _coding,
    _flatten_condition,
    _flatten_encounter,
    _flatten_medication_request,
    _flatten_observation,
    _flatten_patient,
    _ref_id,
)


def _bundle() -> dict:
    return {
        "resourceType": "Bundle", "id": "b1", "type": "collection",
        "entry": [
            {"resource": {
                "resourceType": "Patient", "id": "p1", "birthDate": "1970-01-01",
                "gender": "male", "maritalStatus": {"coding": [{"code": "M"}]},
                "address": [{"postalCode": "10001", "country": "US"}],
            }},
            {"resource": {
                "resourceType": "Observation", "id": "o1", "status": "final",
                "subject": {"reference": "Patient/p1"},
                "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4",
                                     "display": "Heart rate"}]},
                "effectiveDateTime": "2024-01-01T00:00:00Z",
                "valueQuantity": {"value": 72, "unit": "/min", "code": "/min"},
            }},
            {"resource": {
                "resourceType": "Encounter", "id": "e1", "status": "finished",
                "subject": {"reference": "Patient/p1"}, "class": {"code": "IMP"},
                "period": {"start": "2024-01-01", "end": "2024-01-03"},
            }},
            {"resource": {
                "resourceType": "Condition", "id": "c1",
                "subject": {"reference": "Patient/p1"},
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10",
                                     "code": "E11", "display": "Diabetes"}]},
                "onsetDateTime": "2020-01-01", "recordedDate": "2020-02-01",
            }},
            {"resource": {
                "resourceType": "MedicationRequest", "id": "m1", "status": "active",
                "intent": "order", "subject": {"reference": "Patient/p1"},
                "medicationCodeableConcept": {"coding": [{"system": "rxnorm", "code": "860975",
                                                          "display": "Metformin"}]},
                "authoredOn": "2024-01-02",
            }},
            {"resource": {"resourceType": "Provenance", "id": "x1"}},
        ],
    }


def test_bundle_parses_all_five_frames():
    result = fd.parse_domain(_bundle(), format="fhir")
    assert result.suggested_domain == "healthcare"
    assert {n: len(df) for n, df in result.frames.items()} == {
        "patient": 1, "observation": 1, "encounter": 1, "condition": 1, "medication_request": 1,
    }
    assert result.metadata["bundle_id"] == "b1"
    assert result.metadata["total_resources"] == 6


def test_patient_columns_align_with_validator():
    patient = fd.parse_domain(_bundle(), format="fhir").frames["patient"].iloc[0]
    assert patient["patient_id"] == "p1"
    assert patient["birth_date"] == "1970-01-01"
    assert patient["gender"] == "male"
    assert patient["marital_status"] == "M"
    assert patient["address_postal_code"] == "10001"


def test_observation_flattening_and_reference_strip():
    obs = fd.parse_domain(_bundle(), format="fhir").frames["observation"].iloc[0]
    assert obs["patient_id"] == "p1"          # subject reference "Patient/p1" -> "p1"
    assert obs["code_system"] == "http://loinc.org"
    assert obs["code_value"] == "8867-4"
    assert obs["value_quantity"] == 72
    assert obs["value_unit"] == "/min"


def test_condition_and_medication_flattening():
    frames = fd.parse_domain(_bundle(), format="fhir").frames
    cond = frames["condition"].iloc[0]
    assert cond["clinical_status"] == "active"
    assert cond["code_value"] == "E11"
    assert "icd-10" in cond["code_system"]
    med = frames["medication_request"].iloc[0]
    assert med["status"] == "active" and med["intent"] == "order"
    assert med["medication_code"] == "860975"


def test_unsupported_resource_type_is_warned_not_dropped():
    result = fd.parse_domain(_bundle(), format="fhir")
    assert any("Provenance" in w for w in result.warnings)
    assert result.metadata["resource_types"]["Provenance"] == 1


def test_single_resource_input():
    result = fd.parse_domain({"resourceType": "Patient", "id": "solo"}, format="fhir")
    assert len(result.frames["patient"]) == 1


def test_list_of_resources_input():
    result = fd.parse_domain(
        [{"resourceType": "Patient", "id": "a"}, {"resourceType": "Patient", "id": "b"}],
        format="fhir")
    assert list(result.frames["patient"]["patient_id"]) == ["a", "b"]


def test_json_string_and_path_inputs(tmp_path):
    text = json.dumps(_bundle())
    assert fd.parse_domain(text, format="fhir").metadata["total_resources"] == 6
    p = tmp_path / "bundle.json"
    p.write_text(text)
    assert fd.parse_domain(str(p), format="fhir").metadata["total_resources"] == 6


def test_invalid_json_warns_not_raises():
    result = FHIRParser().parse("{not valid json")
    assert all(df.empty for df in result.frames.values())
    assert any("invalid FHIR JSON" in w for w in result.warnings)


def test_empty_bundle_warns():
    result = fd.parse_domain({"resourceType": "Bundle", "entry": []}, format="fhir")
    assert any("no FHIR resources" in w for w in result.warnings)


# -- end-to-end: parse then clean ---------------------------------------------------


def test_clean_domain_file_parses_and_validates_condition(tmp_path):
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(_bundle()))
    out, report = fd.clean_domain_file(
        str(p), format="fhir", domain="healthcare", frame="condition", return_report=True)
    assert isinstance(out, pd.DataFrame)
    assert report.domain == "healthcare"
    assert report.domain_trust_score is not None


@pytest.mark.parametrize("frame", ["patient", "observation", "encounter",
                                   "condition", "medication_request"])
def test_each_parsed_frame_cleans_under_healthcare(frame):
    frames = fd.parse_domain(_bundle(), format="fhir").frames
    out, report = fd.clean(frames[frame], domain="healthcare", return_report=True, verbose=False)
    assert report.domain == "healthcare"
    assert isinstance(out, pd.DataFrame)


# -- _coding() helper unit tests ----------------------------------------------------


def test_coding_returns_none_for_non_dict():
    assert _coding(None) == (None, None, None)
    assert _coding("string") == (None, None, None)
    assert _coding(42) == (None, None, None)


def test_coding_returns_none_for_empty_coding_list():
    concept = {"coding": []}
    system, code, display = _coding(concept)
    assert system is None and code is None and display is None


def test_coding_extracts_first_coding():
    concept = {
        "coding": [
            {"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"},
            {"system": "http://other.org", "code": "OTHER"},
        ]
    }
    system, code, display = _coding(concept)
    assert system == "http://loinc.org"
    assert code == "8867-4"
    assert display == "Heart rate"


def test_coding_falls_back_to_concept_text_when_no_display():
    concept = {
        "coding": [{"system": "http://loinc.org", "code": "8867-4"}],
        "text": "Heart rate text",
    }
    _, _, display = _coding(concept)
    assert display == "Heart rate text"


def test_coding_handles_non_dict_first_coding_entry():
    # Codings list contains a non-dict item; falls back to empty dict.
    concept = {"coding": ["not-a-dict"]}
    system, code, display = _coding(concept)
    assert system is None and code is None


def test_coding_missing_coding_key_returns_text_as_display():
    # When "coding" is absent, system and code are None; display falls back to "text".
    concept = {"text": "some text"}
    system, code, display = _coding(concept)
    assert system is None and code is None
    assert display == "some text"


def test_coding_no_coding_and_no_text_returns_all_nones():
    concept = {}
    system, code, display = _coding(concept)
    assert system is None and code is None and display is None


# -- _ref_id() helper unit tests ----------------------------------------------------


def test_ref_id_extracts_id_after_slash():
    assert _ref_id({"reference": "Patient/abc123"}) == "abc123"


def test_ref_id_no_slash_returns_whole_string():
    # A plain ID without a slash: split("/")[-1] is the full string.
    assert _ref_id({"reference": "plainId"}) == "plainId"


def test_ref_id_non_dict_returns_none():
    assert _ref_id(None) is None
    assert _ref_id("Patient/p1") is None
    assert _ref_id(123) is None


def test_ref_id_missing_reference_key_returns_none():
    assert _ref_id({}) is None
    assert _ref_id({"display": "Dr. Smith"}) is None


def test_ref_id_empty_reference_string_returns_none():
    assert _ref_id({"reference": ""}) is None


# -- _flatten_patient edge cases ----------------------------------------------------


def test_flatten_patient_deceased_datetime_sets_deceased_true():
    r = {"id": "p1", "deceasedDateTime": "2023-05-01"}
    row = _flatten_patient(r)
    assert row["deceased"] is True
    assert row["deceased_date"] == "2023-05-01"


def test_flatten_patient_deceased_boolean_true():
    r = {"id": "p1", "deceasedBoolean": True}
    row = _flatten_patient(r)
    assert row["deceased"] is True
    assert row["deceased_date"] is None


def test_flatten_patient_no_address_uses_empty_dict():
    r = {"id": "p1"}
    row = _flatten_patient(r)
    assert row["address_postal_code"] is None
    assert row["address_country"] is None


def test_flatten_patient_address_is_non_dict_entry():
    # If the first address entry is not a dict, falls back gracefully.
    r = {"id": "p1", "address": ["not-a-dict"]}
    row = _flatten_patient(r)
    assert row["address_postal_code"] is None


# -- _flatten_observation edge cases ------------------------------------------------


def test_flatten_observation_uses_effective_period_when_no_datetime():
    r = {
        "id": "o1", "status": "final",
        "subject": {"reference": "Patient/p1"},
        "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
        "effectivePeriod": {"start": "2024-01-01", "end": "2024-01-02"},
    }
    row = _flatten_observation(r)
    assert row["effective_date"] == "2024-01-01"


def test_flatten_observation_value_string_populated():
    r = {
        "id": "o1", "status": "final",
        "subject": {"reference": "Patient/p1"},
        "code": {"coding": [{"system": "http://loinc.org", "code": "X"}]},
        "valueString": "positive",
    }
    row = _flatten_observation(r)
    assert row["value_string"] == "positive"
    assert row["value_quantity"] is None


def test_flatten_observation_value_unit_falls_back_to_code():
    # valueQuantity has "code" but not "unit" -> use "code" as unit.
    r = {
        "id": "o1", "status": "final",
        "subject": {"reference": "Patient/p1"},
        "code": {"coding": [{"system": "http://loinc.org", "code": "X"}]},
        "valueQuantity": {"value": 5.0, "code": "mg"},
    }
    row = _flatten_observation(r)
    assert row["value_unit"] == "mg"


# -- _flatten_condition edge cases --------------------------------------------------


def test_flatten_condition_onset_period_fallback():
    r = {
        "id": "c1", "subject": {"reference": "Patient/p1"},
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006"}]},
        "onsetPeriod": {"start": "2019-06-01"},
        "recordedDate": "2019-07-01",
    }
    row = _flatten_condition(r)
    assert row["onset_date"] == "2019-06-01"


def test_flatten_condition_no_category_returns_none():
    r = {
        "id": "c1", "subject": {"reference": "Patient/p1"},
        "code": {"coding": [{"code": "I10"}]},
    }
    row = _flatten_condition(r)
    assert row["category"] is None


def test_flatten_condition_verification_status_extracted():
    r = {
        "id": "c1", "subject": {"reference": "Patient/p1"},
        "code": {"coding": [{"code": "I10"}]},
        "verificationStatus": {"coding": [{"code": "confirmed"}]},
    }
    row = _flatten_condition(r)
    assert row["verification_status"] == "confirmed"


# -- _flatten_medication_request edge cases -----------------------------------------


def test_flatten_medication_request_requester_extracted():
    r = {
        "id": "m1", "status": "active", "intent": "order",
        "subject": {"reference": "Patient/p1"},
        "requester": {"reference": "Practitioner/dr1"},
        "medicationCodeableConcept": {"coding": [{"code": "860975"}]},
        "authoredOn": "2024-01-02",
    }
    row = _flatten_medication_request(r)
    assert row["requester"] == "dr1"


def test_flatten_medication_request_missing_medication_concept():
    r = {
        "id": "m1", "status": "active", "intent": "order",
        "subject": {"reference": "Patient/p1"},
    }
    row = _flatten_medication_request(r)
    assert row["medication_code"] is None
    assert row["medication_system"] is None


# -- _iter_resources edge cases -----------------------------------------------------


def test_iter_resources_list_item_without_resource_type_warns():
    result = FHIRParser().parse([{"id": "no-type-here"}, {"resourceType": "Patient", "id": "p1"}])
    assert len(result.frames["patient"]) == 1
    assert any("resourceType skipped" in w for w in result.warnings)


def test_iter_resources_non_bundle_dict_without_resource_type_warns():
    result = FHIRParser().parse({"id": "x", "type": "something"})
    assert all(df.empty for df in result.frames.values())
    assert any("not a FHIR Bundle" in w for w in result.warnings)


def test_iter_resources_bundle_entry_missing_resource_key_is_skipped():
    data = {
        "resourceType": "Bundle", "id": "b2", "type": "collection",
        "entry": [
            {"fullUrl": "Patient/p1"},           # no "resource" key
            {"resource": {"resourceType": "Patient", "id": "p1"}},
        ],
    }
    result = FHIRParser().parse(data)
    assert len(result.frames["patient"]) == 1


# -- bundle metadata ----------------------------------------------------------------


def test_bundle_type_stored_in_metadata():
    result = fd.parse_domain(_bundle(), format="fhir")
    assert result.metadata["bundle_type"] == "collection"


def test_non_bundle_has_null_bundle_metadata():
    result = fd.parse_domain({"resourceType": "Patient", "id": "solo"}, format="fhir")
    assert result.metadata["bundle_id"] is None
    assert result.metadata["bundle_type"] is None


def test_multiple_unsupported_types_all_counted():
    data = {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Provenance", "id": "prov1"}},
            {"resource": {"resourceType": "Provenance", "id": "prov2"}},
            {"resource": {"resourceType": "AuditEvent", "id": "ae1"}},
        ],
    }
    result = FHIRParser().parse(data)
    assert result.metadata["resource_types"]["Provenance"] == 2
    assert result.metadata["resource_types"]["AuditEvent"] == 1
    # Both unsupported types appear in the warning message.
    warning_text = " ".join(result.warnings)
    assert "Provenance" in warning_text and "AuditEvent" in warning_text


# -- registry -----------------------------------------------------------------------


def test_fhir_format_in_registry_available():
    from freshdata.parsers.registry import available, get_parser
    assert "fhir" in available()
    parser = get_parser("fhir")
    assert isinstance(parser, FHIRParser)
