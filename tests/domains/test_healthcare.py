"""Conformance tests for the healthcare (FHIR / US Core) domain pack."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains import UnknownDomainError
from freshdata.domains.healthcare import (
    AmbiguousFHIRResourceError,
    HealthcareValidator,
    UnsupportedFHIRResourceError,
)


def _violated(rep, rule_id: str) -> bool:
    return any(f["rule_id"] == rule_id and f["status"] == "violated" for f in rep.domain_findings)


@pytest.fixture
def good_patient() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "birth_date": ["1980-05-03", "1992-11-20", "1955-02-15"],
        "gender": ["male", "female", "other"],
        "deceased": [False, False, True],
        "deceased_date": [None, None, "2022-06-01"],
        "marital_status": ["M", "S", "W"],
        "address_postal_code": ["10001", "94016", "60601"],
        "address_country": ["US", "US", "GB"],
    })


@pytest.fixture
def good_observation() -> pd.DataFrame:
    return pd.DataFrame({
        "observation_id": ["O1", "O2"],
        "patient_id": ["P1", "P2"],
        "status": ["final", "amended"],
        "code_system": ["http://loinc.org", "http://loinc.org"],
        "code_value": ["8867-4", "8480-6"],
        "effective_date": ["2024-01-10T09:00:00", "2024-01-11"],
        "value_quantity": [72.0, 120.0],
        "value_unit": ["/min", "mm[Hg]"],
        "interpretation": ["N", "H"],
    })


@pytest.fixture
def good_encounter() -> pd.DataFrame:
    return pd.DataFrame({
        "encounter_id": ["E1", "E2"],
        "patient_id": ["P1", "P2"],
        "status": ["finished", "finished"],
        "class_code": ["AMB", "IMP"],
        "period_start": ["2024-01-10T09:00:00", "2024-02-01T08:00:00"],
        "period_end": ["2024-01-10T10:00:00", "2024-02-03T12:00:00"],
        "reason_code_system": ["http://snomed.info/sct", None],
    })


# -- happy paths -----------------------------------------------------------

@pytest.mark.parametrize("resource,fixture", [
    ("Patient", "good_patient"),
    ("Observation", "good_observation"),
    ("Encounter", "good_encounter"),
])
def test_happy_path_passes(resource, fixture, request):
    df = request.getfixturevalue(fixture)
    out, rep = fd.clean(df, domain="healthcare", fhir_resource=resource,
                        return_report=True, verbose=False)
    assert rep.domain == "healthcare"
    assert rep.domain_trust_score >= 0.95
    assert not [f for f in rep.domain_findings
                if f["status"] == "violated" and f["severity"] == "error"]
    assert out.shape[0] == df.shape[0]


# -- required-field violations --------------------------------------------

@pytest.mark.parametrize("resource,fixture,field", [
    ("Patient", "good_patient", "patient_id"),
    ("Patient", "good_patient", "birth_date"),
    ("Patient", "good_patient", "gender"),
    ("Observation", "good_observation", "observation_id"),
    ("Observation", "good_observation", "patient_id"),
    ("Observation", "good_observation", "status"),
    ("Observation", "good_observation", "code_value"),
    ("Encounter", "good_encounter", "encounter_id"),
    ("Encounter", "good_encounter", "patient_id"),
    ("Encounter", "good_encounter", "status"),
])
def test_required_field_missing(resource, fixture, field, request):
    df = request.getfixturevalue(fixture)
    validator = HealthcareValidator(fhir_resource=resource)
    actual = validator.detect_columns(df).actual(field)
    report = validator.validate(df.drop(columns=[actual]))
    assert field in report.mapping.unmapped_required
    assert any("MISSING_REQUIRED_FIELD" in r.message and r.violated for r in report.results)


# -- format violations -----------------------------------------------------

def test_patient_format_violations(good_patient):
    df = good_patient.copy()
    df.loc[0, "birth_date"] = "1980-13-40"   # HC-P003: impossible date
    df.loc[1, "birth_date"] = "2099-01-01"   # HC-P004: future
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Patient",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-P003")
    assert _violated(rep, "HC-P004")


def test_partial_birth_date_is_info_not_error(good_patient):
    df = good_patient.copy()
    df["birth_date"] = df["birth_date"].astype(object)
    df.loc[0, "birth_date"] = "1980"      # FHIR partial date — valid, info only
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Patient",
                      return_report=True, verbose=False)
    assert not _violated(rep, "HC-P003")      # partial is accepted
    assert _violated(rep, "HC-P003I")         # surfaced as info
    p003i = next(f for f in rep.domain_findings if f["rule_id"] == "HC-P003I")
    assert p003i["severity"] == "info"


def test_observation_format_violations(good_observation):
    df = good_observation.copy()
    df.loc[0, "effective_date"] = "last tuesday"   # HC-O005
    df.loc[1, "value_quantity"] = -5.0             # HC-O008: negative
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Observation",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-O005")
    assert _violated(rep, "HC-O008")


def test_encounter_format_violation(good_encounter):
    df = good_encounter.copy()
    df.loc[0, "period_start"] = "yesterday"   # HC-E004
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Encounter",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-E004")


# -- reference violations --------------------------------------------------

def test_patient_reference_violations(good_patient):
    df = good_patient.copy()
    df.loc[0, "gender"] = "ambiguous"   # HC-P005: not a gender code, uncoercible
    df.loc[1, "address_country"] = "ZZ"  # HC-P008: not ISO 3166-1
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Patient",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-P005")
    assert _violated(rep, "HC-P008")


def test_observation_reference_violations(good_observation):
    df = good_observation.copy()
    df.loc[0, "status"] = "in-flight"               # HC-O002
    df.loc[0, "code_system"] = "http://example.org"  # HC-O003 (warning)
    df.loc[1, "code_value"] = "0000-0"              # HC-O004 (warning, LOINC system)
    df.loc[0, "interpretation"] = "ZZ"              # HC-O009
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Observation",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-O002")
    assert _violated(rep, "HC-O003")
    assert _violated(rep, "HC-O004")
    assert _violated(rep, "HC-O009")


def test_encounter_reference_violations(good_encounter):
    df = good_encounter.copy()
    df.loc[0, "status"] = "teleported"               # HC-E002
    df.loc[1, "class_code"] = "SPACE"                # HC-E003
    df.loc[0, "reason_code_system"] = "http://nope"  # HC-E007 (warning)
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Encounter",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-E002")
    assert _violated(rep, "HC-E003")
    assert _violated(rep, "HC-E007")


# -- business / cross-field violations ------------------------------------

def test_patient_business_violations(good_patient):
    df = good_patient.copy()
    df.loc[2, "deceased_date"] = "1900-01-01"   # HC-P006: before birth (1955)
    df.loc[0, "deceased"] = True
    df.loc[0, "deceased_date"] = "2099-01-01"   # HC-P007: future death
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Patient",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-P006")
    assert _violated(rep, "HC-P007")


def test_observation_business_violations(good_observation):
    df = good_observation.copy()
    df["value_string"] = [None, None]
    df.loc[0, "value_quantity"] = None   # HC-O006: no value at all
    df.loc[1, "value_unit"] = None       # HC-O007: quantity without unit
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Observation",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-O006")
    assert _violated(rep, "HC-O007")


def test_encounter_business_violations(good_encounter):
    df = good_encounter.copy()
    df.loc[0, "period_end"] = "2024-01-10T08:00:00"   # HC-E005: end before start
    df.loc[1, "period_start"] = "2099-01-01T00:00:00"  # HC-E006: future, finished
    df.loc[1, "period_end"] = "2099-01-02T00:00:00"
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Encounter",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-E005")
    assert _violated(rep, "HC-E006")


def test_encounter_long_duration_is_warning(good_encounter):
    df = good_encounter.copy()
    df.loc[0, "period_end"] = "2025-06-01T10:00:00"   # > 365 days from 2024-01-10
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Encounter",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-E008")
    e008 = next(f for f in rep.domain_findings if f["rule_id"] == "HC-E008")
    assert e008["severity"] == "warning"


def test_implausible_age_is_warning(good_patient):
    df = good_patient.copy()
    df.loc[0, "birth_date"] = "1820-01-01"   # > 150 years -> HC-P009 warning
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Patient",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-P009")


# -- repair audit ----------------------------------------------------------

def test_repair_audit_gender_case_coercion(good_patient):
    df = good_patient.copy()
    df.loc[0, "gender"] = "Male"      # case fix -> "male"
    df.loc[1, "gender"] = "FEMALE"    # case fix -> "female"
    out, rep = fd.clean(df, domain="healthcare", fhir_resource="Patient",
                        return_report=True, verbose=False)
    applied = [a for a in rep.domain_repairs
               if a["rule_id"] == "HC-P005" and a["status"] == "applied"]
    assert applied and applied[0]["from"] == "Male" and applied[0]["to"] == "male"
    assert applied[0]["strategy"] == "coerce"
    assert out.loc[0, "gender"] == "male" and out.loc[1, "gender"] == "female"


def test_uncoercible_gender_is_unresolvable(good_patient):
    df = good_patient.copy()
    df.loc[0, "gender"] = "M"   # not a case variant of any code
    out, rep = fd.clean(df, domain="healthcare", fhir_resource="Patient",
                        return_report=True, verbose=False)
    assert out.loc[0, "gender"] == "M"
    assert any(a["rule_id"] == "HC-P005" and a["status"] == "unresolvable"
               for a in rep.domain_repairs)


# -- ID safety -------------------------------------------------------------

def test_id_safety_null_patient_id_never_filled(good_observation):
    df = good_observation.copy()
    df.loc[0, "patient_id"] = None
    out, rep = fd.clean(df, domain="healthcare", fhir_resource="Observation",
                        return_report=True, verbose=False)
    assert pd.isna(out.loc[0, "patient_id"])
    assert _violated(rep, "HC-O001")
    assert not [a for a in rep.domain_repairs
                if a["column"] == "patient_id" and a["status"] == "applied"]


def test_id_safety_null_observation_id_never_filled(good_observation):
    df = good_observation.copy()
    df.loc[1, "observation_id"] = None
    out, _ = fd.clean(df, domain="healthcare", fhir_resource="Observation",
                      return_report=True, verbose=False)
    assert pd.isna(out.loc[1, "observation_id"])


# -- PHI redaction ---------------------------------------------------------

def test_phi_masked_by_default(good_patient):
    df = good_patient.copy()
    df["birth_date"] = df["birth_date"].astype(object)
    df.loc[0, "birth_date"] = "1980"   # partial date -> HC-P003I flags birth_date (PHI)
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Patient",
                      return_report=True, verbose=False)
    p003i = [a for a in rep.domain_repairs if a["rule_id"] == "HC-P003I"]
    assert p003i and p003i[0]["column"] == "birth_date"
    assert p003i[0]["from"] == "[PHI]"
    assert "1980" not in str(rep.domain_repairs)


def test_phi_shown_with_opt_in(good_patient):
    df = good_patient.copy()
    df["birth_date"] = df["birth_date"].astype(object)
    df.loc[0, "birth_date"] = "1980"
    _, rep = fd.clean(df, domain="healthcare", fhir_resource="Patient",
                      audit_include_phi=True, return_report=True, verbose=False)
    p003i = [a for a in rep.domain_repairs if a["rule_id"] == "HC-P003I"]
    assert p003i and p003i[0]["from"] == "1980"


# -- resource routing / auto-detection ------------------------------------

@pytest.mark.parametrize("resource,fixture", [
    ("Patient", "good_patient"),
    ("Observation", "good_observation"),
    ("Encounter", "good_encounter"),
])
def test_autodetect_resource(resource, fixture, request):
    df = request.getfixturevalue(fixture)
    validator = HealthcareValidator()
    validator.validate(df)
    assert validator.fhir_resource == resource


def test_autodetect_end_to_end(good_patient):
    out, rep = fd.clean(good_patient, domain="healthcare", return_report=True, verbose=False)
    assert rep.domain_trust_score >= 0.95


def test_ambiguous_resource_raises():
    df = pd.DataFrame({"patient_id": ["P1"], "status": ["final"]})
    with pytest.raises(AmbiguousFHIRResourceError):
        HealthcareValidator().validate(df)


def test_unsupported_resource_raises():
    with pytest.raises(UnsupportedFHIRResourceError) as exc:
        HealthcareValidator(fhir_resource="DiagnosticReport")
    assert "Patient" in exc.value.supported


# -- regression guards -----------------------------------------------------

def test_validation_never_mutates_input(good_patient):
    before = good_patient.copy()
    HealthcareValidator(fhir_resource="Patient").validate(good_patient)
    pd.testing.assert_frame_equal(good_patient, before)


def test_unknown_domain_lists_available(good_patient):
    with pytest.raises(UnknownDomainError) as exc:
        fd.clean(good_patient, domain="unknown_xyz")
    assert "healthcare" in exc.value.available


def test_standalone_import():
    assert HealthcareValidator(fhir_resource="Patient").domain_name == "healthcare"
