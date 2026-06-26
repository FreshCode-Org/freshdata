"""Tests for the FHIR R4 JSON parser."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import freshdata as fd
from freshdata.parsers.fhir import FHIRParser


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
