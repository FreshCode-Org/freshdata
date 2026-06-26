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


# -- Condition additional rules -----------------------------------------------------


def test_condition_code_value_required(good_condition):
    df = good_condition.copy()
    df.loc[0, "code_value"] = None
    _, rep = _clean(df, "Condition")
    assert _violated(rep, "HC-C003")
    hc003 = next(f for f in rep.domain_findings if f["rule_id"] == "HC-C003")
    assert hc003["severity"] == "error"


def test_condition_onset_date_invalid_format_fires_warning(good_condition):
    df = good_condition.copy()
    df.loc[0, "onset_date"] = "yesterday"   # not ISO 8601
    _, rep = _clean(df, "Condition")
    assert _violated(rep, "HC-C006")
    hc006 = next(f for f in rep.domain_findings if f["rule_id"] == "HC-C006")
    assert hc006["severity"] == "warning"


def test_condition_onset_date_valid_partial_date_passes(good_condition):
    # FHIR allows partial dates like "2020" and "2020-06" as valid.
    df = good_condition.copy()
    df.loc[0, "onset_date"] = "2020"
    df.loc[1, "onset_date"] = "2021-06"
    _, rep = _clean(df, "Condition")
    assert not _violated(rep, "HC-C006")


def test_condition_patient_id_never_imputed(good_condition):
    df = good_condition.copy()
    df.loc[0, "patient_id"] = None
    out, rep = _clean(df, "Condition")
    assert _violated(rep, "HC-C002")
    assert pd.isna(out.loc[0, "patient_id"])


@pytest.mark.parametrize("status", ["active", "recurrence", "relapse",
                                     "inactive", "remission", "resolved"])
def test_all_valid_condition_clinical_statuses_pass(status, good_condition):
    df = good_condition.copy()
    df["clinical_status"] = status
    _, rep = _clean(df, "Condition")
    assert not _violated(rep, "HC-C004")


def test_condition_icd10_system_without_hyphen_also_flagged(good_condition):
    # "icd10" (no hyphen) should also be detected as an ICD-10 system.
    df = good_condition.copy()
    df["code_system"] = "http://example.org/icd10"
    df.loc[0, "code_value"] = "ZZZ99"   # not in icd10_common
    _, rep = _clean(df, "Condition")
    assert _violated(rep, "HC-C005")


def test_condition_icd10_null_system_skips_check(good_condition):
    # If code_system is null, HC-C005 should not fire (no system to gate on).
    df = good_condition.copy()
    df["code_system"] = None
    df["code_value"] = "ZZZ99"
    _, rep = _clean(df, "Condition")
    assert not _violated(rep, "HC-C005")


# -- MedicationRequest additional rules ---------------------------------------------


def test_medication_patient_id_required(good_medication):
    df = good_medication.copy()
    df.loc[0, "patient_id"] = None
    out, rep = _clean(df, "MedicationRequest")
    assert _violated(rep, "HC-M002")
    assert pd.isna(out.loc[0, "patient_id"])  # IDs never imputed


def test_medication_code_missing_is_warning_not_error(good_medication):
    df = good_medication.copy()
    df["medication_code"] = None
    _, rep = _clean(df, "MedicationRequest")
    assert _violated(rep, "HC-M005")
    hc005 = next(f for f in rep.domain_findings if f["rule_id"] == "HC-M005")
    assert hc005["severity"] == "warning"


def test_medication_authored_on_invalid_date_fires_warning(good_medication):
    df = good_medication.copy()
    df.loc[0, "authored_on"] = "not-a-date"
    _, rep = _clean(df, "MedicationRequest")
    assert _violated(rep, "HC-M006")
    hc006 = next(f for f in rep.domain_findings if f["rule_id"] == "HC-M006")
    assert hc006["severity"] == "warning"


@pytest.mark.parametrize("status", [
    "active", "on-hold", "cancelled", "completed",
    "entered-in-error", "stopped", "draft", "unknown",
])
def test_all_valid_medication_statuses_pass(status, good_medication):
    df = good_medication.copy()
    df["status"] = status
    _, rep = _clean(df, "MedicationRequest")
    assert not _violated(rep, "HC-M003")


@pytest.mark.parametrize("intent", [
    "proposal", "plan", "order", "original-order",
    "reflex-order", "filler-order", "instance-order", "option",
])
def test_all_valid_medication_intents_pass(intent, good_medication):
    df = good_medication.copy()
    df["intent"] = intent
    _, rep = _clean(df, "MedicationRequest")
    assert not _violated(rep, "HC-M004")


# -- Observation UCUM additional ----------------------------------------------------


def test_observation_valid_ucum_units_pass_hc_o010():
    obs = pd.DataFrame({
        "observation_id": ["o1", "o2"],
        "patient_id": ["p1", "p2"],
        "status": ["final", "final"],
        "code_system": ["http://loinc.org", "http://loinc.org"],
        "code_value": ["8867-4", "8480-6"],
        "value_quantity": [72, 120],
        "value_unit": ["/min", "mm[Hg]"],   # both are valid UCUM units
    })
    _, rep = _clean(obs, "Observation")
    assert not _violated(rep, "HC-O010")


def test_observation_null_unit_does_not_violate_hc_o010():
    # Null units are handled by HC-O007 (quantity without unit), not HC-O010.
    obs = pd.DataFrame({
        "observation_id": ["o1"],
        "patient_id": ["p1"],
        "status": ["final"],
        "code_system": ["http://loinc.org"],
        "code_value": ["8867-4"],
        "value_quantity": [None],
        "value_string": ["positive"],
        "value_unit": [None],
    })
    _, rep = _clean(obs, "Observation")
    assert not _violated(rep, "HC-O010")


# -- SUPPORTED_RESOURCES completeness -----------------------------------------------


def test_supported_resources_contains_all_five():
    assert set(SUPPORTED_RESOURCES) == {
        "Patient", "Observation", "Encounter", "Condition", "MedicationRequest"
    }


# -- reference loading sanity checks ------------------------------------------------


def test_condition_reference_values_load():
    from freshdata.domains.healthcare.validator import HealthcareValidator
    v = HealthcareValidator(fhir_resource="Condition")
    codes = v.load_reference_values("condition_clinical_status")
    assert set(codes) == {"active", "recurrence", "relapse", "inactive", "remission", "resolved"}


def test_medication_status_reference_values_load():
    from freshdata.domains.healthcare.validator import HealthcareValidator
    v = HealthcareValidator(fhir_resource="MedicationRequest")
    codes = v.load_reference_values("medicationrequest_status")
    assert "active" in codes and "cancelled" in codes and "unknown" in codes


def test_medication_intent_reference_values_load():
    from freshdata.domains.healthcare.validator import HealthcareValidator
    v = HealthcareValidator(fhir_resource="MedicationRequest")
    codes = v.load_reference_values("medicationrequest_intent")
    assert "order" in codes and "plan" in codes and "proposal" in codes


def test_icd10_common_reference_values_load():
    from freshdata.domains.healthcare.validator import HealthcareValidator
    v = HealthcareValidator(fhir_resource="Condition")
    codes = v.load_reference_values("icd10_common")
    assert "E11" in codes and "I10" in codes


# -- auto-detection edge cases ------------------------------------------------------


def test_condition_and_medication_together_raises_ambiguous():
    """A frame with both Condition and MedicationRequest columns is ambiguous."""
    from freshdata.domains.healthcare import AmbiguousFHIRResourceError
    df = pd.DataFrame({
        "condition_id": ["c1"],
        "medication_request_id": ["m1"],
        "patient_id": ["p1"],
        "clinical_status": ["active"],
        "status": ["active"],
        "intent": ["order"],
    })
    with pytest.raises(AmbiguousFHIRResourceError):
        fd.clean(df, domain="healthcare", return_report=True, verbose=False)
