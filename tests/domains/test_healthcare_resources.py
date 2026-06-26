"""Tests for the new FHIR healthcare resources: Condition, MedicationRequest, and the
UCUM unit check on Observation."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains.healthcare.validator import SUPPORTED_RESOURCES


def _clean(df, resource):
    return fd.clean(df, domain="healthcare", fhir_resource=resource,
                    return_report=True, verbose=False)


def _violated(rep, rule_id):
    return any(f["rule_id"] == rule_id and f["status"] == "violated"
               for f in rep.domain_findings)


# -- Condition ----------------------------------------------------------------------


@pytest.fixture
def good_condition():
    return pd.DataFrame({
        "condition_id": ["c1", "c2"],
        "patient_id": ["p1", "p2"],
        "clinical_status": ["active", "resolved"],
        "verification_status": ["confirmed", "confirmed"],
        "code_system": ["http://hl7.org/fhir/sid/icd-10"] * 2,
        "code_value": ["E11", "I10"],
        "onset_date": ["2020-01-01", "2021-06-01"],
        "recorded_date": ["2020-02-01", "2021-06-15"],
    })


def test_valid_condition_passes(good_condition):
    out, rep = _clean(good_condition, "Condition")
    assert rep.domain == "healthcare"
    assert not [f for f in rep.domain_findings if f["status"] == "violated"]


def test_condition_requires_id_and_patient(good_condition):
    df = good_condition.copy()
    df.loc[0, "condition_id"] = None
    df.loc[1, "patient_id"] = None
    out, rep = _clean(df, "Condition")
    assert _violated(rep, "HC-C001") and _violated(rep, "HC-C002")
    assert pd.isna(out.loc[0, "condition_id"])  # IDs never imputed


def test_condition_clinical_status_validated(good_condition):
    df = good_condition.copy()
    df.loc[0, "clinical_status"] = "bogus"
    _, rep = _clean(df, "Condition")
    assert _violated(rep, "HC-C004")


def test_condition_icd10_code_checked_against_common_set(good_condition):
    df = good_condition.copy()
    df.loc[0, "code_value"] = "ZZZ99"   # not in icd10_common
    _, rep = _clean(df, "Condition")
    assert _violated(rep, "HC-C005")
    hc005 = next(f for f in rep.domain_findings if f["rule_id"] == "HC-C005")
    assert hc005["severity"] == "warning"  # sample set -> review, not hard error


def test_condition_icd10_check_skips_non_icd10_systems(good_condition):
    df = good_condition.copy()
    df["code_system"] = "http://snomed.info/sct"   # not ICD-10 -> HC-C005 should not fire
    df.loc[0, "code_value"] = "44054006"
    _, rep = _clean(df, "Condition")
    assert not _violated(rep, "HC-C005")


# -- MedicationRequest --------------------------------------------------------------


@pytest.fixture
def good_medication():
    return pd.DataFrame({
        "medication_request_id": ["m1", "m2"],
        "patient_id": ["p1", "p2"],
        "status": ["active", "completed"],
        "intent": ["order", "plan"],
        "medication_code": ["860975", "197361"],
        "authored_on": ["2024-01-02", "2024-02-03"],
    })


def test_valid_medication_request_passes(good_medication):
    _, rep = _clean(good_medication, "MedicationRequest")
    assert not [f for f in rep.domain_findings if f["status"] == "violated"]


def test_medication_status_and_intent_validated(good_medication):
    df = good_medication.copy()
    df.loc[0, "status"] = "weird"
    df.loc[1, "intent"] = "nope"
    _, rep = _clean(df, "MedicationRequest")
    assert _violated(rep, "HC-M003") and _violated(rep, "HC-M004")


def test_medication_request_id_never_imputed(good_medication):
    df = good_medication.copy()
    df.loc[0, "medication_request_id"] = None
    out, rep = _clean(df, "MedicationRequest")
    assert _violated(rep, "HC-M001")
    assert pd.isna(out.loc[0, "medication_request_id"])


# -- Observation UCUM ---------------------------------------------------------------


def test_observation_unit_validated_against_ucum():
    obs = pd.DataFrame({
        "observation_id": ["o1", "o2"],
        "patient_id": ["p1", "p2"],
        "status": ["final", "final"],
        "code_system": ["http://loinc.org", "http://loinc.org"],
        "code_value": ["8867-4", "8867-4"],
        "value_quantity": [72, 80],
        "value_unit": ["/min", "bpm"],   # bpm is not a UCUM atom
    })
    _, rep = _clean(obs, "Observation")
    assert _violated(rep, "HC-O010")


# -- auto-detection -----------------------------------------------------------------


def test_condition_frame_auto_detected(good_condition):
    # No fhir_resource passed: detection must pick Condition over Observation despite
    # the shared code_* columns.
    out, rep = fd.clean(good_condition, domain="healthcare", return_report=True, verbose=False)
    assert rep.domain == "healthcare"
    # A Condition-specific rule ran (proves the Condition validator was selected).
    assert any(f["rule_id"].startswith("HC-C") for f in rep.domain_findings)


def test_medication_frame_auto_detected(good_medication):
    _, rep = fd.clean(good_medication, domain="healthcare", return_report=True, verbose=False)
    assert any(f["rule_id"].startswith("HC-M") for f in rep.domain_findings)


def test_new_resources_are_supported():
    assert {"Condition", "MedicationRequest"} <= set(SUPPORTED_RESOURCES)
