"""End-to-end conformance tests for all seven domain packs via ``fd.clean``.

These exercise the full public entrypoint (generic clean -> domain validate ->
repair -> findings merged into the :class:`CleanReport`) on realistic, multi-row
datasets that mirror real-world data for each domain — finance ledgers, GS1
catalogs, GTFS feeds, FHIR Patient/Observation frames, Ed-Fi enrollments, ADAPT
field operations, and EIDR/DDEX media metadata.

They assert against the real report API: per-rule outcomes live in
``report.domain_findings`` (JSON-friendly dicts, one per rule) and repair-log
entries in ``report.domain_repairs``; ``report.domain``/``report.domain_trust_score``
carry the pack name and 0-1 trust score.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains.healthcare import HealthcareValidator
from freshdata.domains.registry import UnknownDomainError


def _violated(report, rule_id: str) -> bool:
    """True iff *rule_id* is present and flagged as violated in the findings."""
    return any(f["rule_id"] == rule_id and f["status"] == "violated"
               for f in report.domain_findings)


def _applied(report, rule_id=None, frm=None, to=None) -> bool:
    """True iff an applied repair matches the given rule id / from / to value."""
    for r in report.domain_repairs:
        if r["status"] != "applied":
            continue
        if rule_id is not None and r["rule_id"] != rule_id:
            continue
        if frm is not None and r["from"] != frm:
            continue
        if to is not None and r["to"] != to:
            continue
        return True
    return False


# --------------------------------------------------------------------- finance

def test_finance_conformance():
    finance_df = pd.DataFrame({
        "transaction_id": ["TXN-001", "TXN-001", "TXN-002", "TXN-003", "TXN-004", None],
        "date": ["2024-03-15", "2024-03-15", "15/04/2024", "2024-13-01",
                 "2099-01-01", "2024-01-10"],
        "account_code": ["1000", "2000", "1001", "3000", "ACCT", "X"],
        "debit": [1500.00, 0.00, 750.50, 200.00, 0.00, 500.00],
        "credit": [0.00, 1500.00, 750.50, 0.00, 300.00, 500.00],
        "currency": ["USD", "USD", "GBP", "XYZ", "EUR", "USD"],
        "description": ["Invoice payment", "Invoice payment", "Supplier refund",
                        "Office supplies", "Future booking", "Salary payment"],
        "entity_id": ["E001", "E001", "E002", "E003", "E004", "E005"],
    })
    df_out, rep = fd.clean(finance_df, domain="finance", return_report=True, verbose=False)

    # ID column is never imputed, even where null.
    assert pd.isna(df_out["transaction_id"].iloc[-1])
    assert _violated(rep, "FIN-001")            # null transaction_id
    assert _violated(rep, "FIN-005")            # currency XYZ not ISO 4217
    assert _violated(rep, "FIN-009")            # future date 2099-01-01
    assert rep.domain == "finance"
    assert rep.domain_trust_score is not None


# ---------------------------------------------------------------------- retail

def test_retail_gs1_conformance():
    retail_df = pd.DataFrame({
        "gtin": ["00012345678905", "0-001-2345-6789", "1234567", "00012345678900",
                 "5901234123457", None],
        "gln": ["1234567890128", "1234567890128", None, None, None, None],
        "product_description": ["Organic Whole Milk 1L", "Premium Coffee Blend 500g",
                                "X" * 201, "Energy Drink 250ml", "Pasta Fusilli 500g",
                                "Canned Tomatoes 400g"],
        "brand_name": ["FarmFresh", "BeanBar", "BuzzCo", "BarillaIT", "Mutti", "DelMonte"],
        "net_content": [1.0, 0.5, 0.25, None, 0.5, 0.4],
        "net_content_uom": ["LTR", "KGM", "LTR", "KGM", None, "KGM"],
        "country_of_origin": ["DE", "IT", "US", "ZZ", "IT", "US"],
        "gpc_brick_code": ["10000266", "10001270", "1234567", None, "10001336", "10001050"],
    })
    df_out, rep = fd.clean(retail_df, domain="retail", return_report=True, verbose=False)

    assert pd.isna(df_out["gtin"].iloc[5])              # null GTIN never filled
    assert _violated(rep, "GS1-001")                    # null GTIN
    assert _violated(rep, "GS1-003")                    # wrong check digit
    assert _violated(rep, "GS1-007")                    # description > 200 chars
    assert len(df_out["product_description"].iloc[2]) == 201   # never truncated
    assert rep.domain_trust_score is not None


# ------------------------------------------------------------------- transport

@pytest.fixture
def gtfs_stops() -> pd.DataFrame:
    return pd.DataFrame({
        "stop_id": ["S001", "S002", "S003", "S001", None],
        "stop_name": ["Central Station", "Airport Terminal", "Park & Ride",
                      "Central Station Dup", "Unknown Stop"],
        "stop_lat": [51.5074, 200.0, -33.8688, 51.5074, 40.7128],
        "stop_lon": [-0.1278, -73.9857, 151.2093, -0.1278, -74.0060],
    })


@pytest.fixture
def gtfs_routes() -> pd.DataFrame:
    return pd.DataFrame({
        "route_id": ["R001", "R002", "R003", "R002"],
        "route_short_name": ["1", "Airport Express", "Night Bus", "Airport Express Dup"],
        "route_type": [3, 2, 99, 1],
        "route_long_name": ["City Circle", "City to Airport", "Night Service", "Airport Dup"],
    })


def test_transport_single_file_mode(gtfs_stops):
    df_out, rep = fd.clean(gtfs_stops, domain="transport", gtfs_file="stops",
                           return_report=True, verbose=False)
    assert _violated(rep, "GTFS-S002")                  # stop_lat 200.0 out of range
    assert _violated(rep, "GTFS-S004")                  # duplicate stop_id
    assert pd.isna(df_out["stop_id"].iloc[4])           # null stop_id not filled


def test_transport_full_feed_mode(gtfs_stops, gtfs_routes):
    feed = {"stops": gtfs_stops, "routes": gtfs_routes}
    feed_out, rep = fd.clean(feed, domain="transport", return_report=True, verbose=False)
    assert set(feed_out) == {"stops", "routes"}         # dict in -> dict out
    assert _violated(rep, "GTFS-R002")                  # invalid route_type 99
    assert _violated(rep, "GTFS-R003")                  # duplicate route_id
    assert rep.domain_trust_score is not None


# ------------------------------------------------------------------ healthcare

@pytest.fixture
def fhir_patients() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": ["P001", "P002", None, "P004", "P005"],
        "birth_date": ["1985-06-15", "2090-01-01", "1972-11-30", "1800-03-22", "1995"],
        "gender": ["male", "Female", "UNKNOWN", "apache", "other"],
        "deceased": [False, False, True, False, False],
        "deceased_date": [None, None, "1960-01-01", None, None],
        "address_country": ["US", "GB", "DE", "ZZ", "CA"],
        "marital_status": ["M", "S", "W", "D", None],
    })


@pytest.fixture
def fhir_observations() -> pd.DataFrame:
    return pd.DataFrame({
        "observation_id": ["O001", "O002", "O003", None, "O005"],
        "patient_id": ["P001", "P002", "P003", "P004", "P005"],
        "status": ["final", "preliminary", "bad-status", "final", "cancelled"],
        "code_system": ["http://loinc.org", "http://loinc.org", "http://loinc.org",
                        "http://unknown.org", None],
        "code_value": ["8867-4", "29463-7", "FAKE-999", "8310-5", "2345-7"],
        "effective_date": ["2024-01-15", "2024-02-20", "not-a-date",
                           "2024-03-01", "2024-04-10"],
        "value_quantity": [72.0, 68.5, None, 37.2, -1.0],
        "value_unit": ["beats/min", "kg", None, "Cel", None],
        "value_string": [None, None, None, None, None],
    })


def test_healthcare_patient_conformance(fhir_patients):
    _, rep = fd.clean(fhir_patients, domain="healthcare", fhir_resource="Patient",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-P001")            # null patient_id
    assert _violated(rep, "HC-P004")            # future birth_date
    assert _violated(rep, "HC-P005")            # invalid gender 'apache'
    assert _violated(rep, "HC-P006")            # deceased_date before birth_date
    assert _violated(rep, "HC-P008")            # invalid country ZZ


def test_healthcare_phi_redacted_by_default(fhir_patients):
    """PHI cell values are masked as ``[PHI]`` in the audit trail by default.

    Verified on the value channel (repair ``from``/``to`` for PHI columns); the
    raw patient_id values P001.. are deliberately *not* used as leak sentinels
    because they collide with rule ids (HC-P004, HC-P005, ...).
    """
    _, rep = fd.clean(fhir_patients, domain="healthcare", fhir_resource="Patient",
                      return_report=True, verbose=False)
    birth_dates = {"1985-06-15", "2090-01-01", "1972-11-30", "1800-03-22", "1995"}
    phi_repairs = [r for r in rep.domain_repairs if r["column"] == "birth_date"]
    assert phi_repairs, "expected birth_date (PHI) repair-log entries"
    assert any(r["from"] == "[PHI]" or r["to"] == "[PHI]" for r in phi_repairs)
    assert not any(r["from"] in birth_dates or r["to"] in birth_dates for r in phi_repairs)


def test_healthcare_phi_exposed_when_opted_in(fhir_patients):
    _, rep = fd.clean(fhir_patients, domain="healthcare", fhir_resource="Patient",
                      audit_include_phi=True, return_report=True, verbose=False)
    birth_dates = {"1985-06-15", "2090-01-01", "1972-11-30", "1800-03-22", "1995"}
    phi_repairs = [r for r in rep.domain_repairs if r["column"] == "birth_date"]
    assert any(r["from"] in birth_dates or r["to"] in birth_dates for r in phi_repairs)


def test_healthcare_observation_conformance(fhir_observations):
    _, rep = fd.clean(fhir_observations, domain="healthcare", fhir_resource="Observation",
                      return_report=True, verbose=False)
    assert _violated(rep, "HC-O001")            # null observation_id
    assert _violated(rep, "HC-O002")            # invalid status 'bad-status'
    assert _violated(rep, "HC-O008")            # negative value_quantity


def test_healthcare_resource_auto_detection(fhir_patients):
    # Equivalent of the spec's ``report.domain_resource == "Patient"``: the
    # validator resolves the resource, and Patient-specific rules (HC-P*) run.
    validator = HealthcareValidator()
    validator.validate(fhir_patients)
    assert validator.fhir_resource == "Patient"

    _, rep = fd.clean(fhir_patients, domain="healthcare", return_report=True, verbose=False)
    assert any(f["rule_id"].startswith("HC-P") for f in rep.domain_findings)


# ------------------------------------------------------------------- education

def test_education_edfi_conformance():
    education_df = pd.DataFrame({
        "student_unique_id": ["STU001", "STU002", None, "STU004", "STU005", "STU006"],
        "school_id": ["SCH100", "SCH100", "SCH200", "SCH200", None, "SCH300"],
        "school_year": [2024, 2024, 2024, 1850, 2024, 2030],
        "grade_level": ["Eighth grade", "Twelfth grade", "Fifth grade",
                        "InvalidGrade", "Kindergarten", "First grade"],
        "enrollment_date": ["2023-09-05"] * 6,
        "exit_withdrawal_date": [None, "2024-06-15", None, "2023-08-01", None, None],
        "exit_withdrawal_type": [None, "Graduated", None, None, None, None],
        "assessment_title": ["State Math", "State Reading", None,
                             "State Science", "State Math", None],
        "assessment_score": [85.5, 92.0, None, -10.0, 78.0, None],
        "assessment_score_type": ["Scale score", "Raw score", None,
                                  "Made Up Type", "Percentile", None],
    })
    _, rep = fd.clean(education_df, domain="education", return_report=True, verbose=False)
    assert _violated(rep, "ED-001")             # null student_unique_id
    assert _violated(rep, "ED-003")             # school_year 1850 out of range
    assert _violated(rep, "ED-004")             # invalid grade_level
    assert _violated(rep, "ED-007")             # exit before enrollment
    assert _violated(rep, "ED-010")             # exit date without type
    assert _violated(rep, "ED-014")             # negative assessment_score

    # student id is PHI: not exposed in the audit by default. STU002/4/5/6 do
    # not collide with any ED-* rule id, so they are reliable leak sentinels.
    audit = str(rep.to_dict())
    assert not any(v in audit for v in ("STU002", "STU004", "STU005", "STU006"))


# ----------------------------------------------------------------- agriculture

def test_agriculture_adapt_conformance():
    agriculture_df = pd.DataFrame({
        "field_id": ["F001", "F002", "F003", None, "F005", "F006"],
        "operation_id": ["OP001", "OP002", "OP003", "OP004", "OP005", "OP006"],
        "operation_type": ["Planting", "Harvesting", "Irrigation",
                           "Flying", "Harvesting", "Sampling"],
        "operation_date": ["2024-04-15", "2024-10-20", "2099-06-01",
                           "2024-05-10", "2024-10-25", "not-a-date"],
        "crop_code": ["1.1.1", "1.1.1", None, None, "9999999", "1.1.1"],
        "area": [50.0, 50.0, 12.5, None, 200.0, None],
        "area_unit": ["hectares", "ACR", "HAR", None, "HAR", None],
        "yield_value": [None, 5200.0, None, 250.0, None, None],
        "yield_unit": [None, "KGM", None, "KGM", None, None],
        "soil_ph": [6.8, 7.2, 5.5, 15.0, 6.1, None],
        "soil_om_pct": [3.2, 4.1, 2.8, 25.0, None, 1.8],
        "season_year": [2024, 2024, 2024, 2024, 2023, 2024],
    })
    ag_out, rep = fd.clean(agriculture_df, domain="agriculture",
                           return_report=True, verbose=False)
    assert _violated(rep, "AG-001")             # null field_id
    assert _violated(rep, "AG-003")             # future operation_date
    assert _violated(rep, "AG-004")             # invalid operation_type 'Flying'
    assert _violated(rep, "AG-012")             # yield on non-harvest operation
    assert _violated(rep, "AG-014")             # soil_ph 15.0 > 14
    assert _violated(rep, "AG-017")             # soil_om_pct 25.0 too high

    # Unit coercion repair: "hectares" -> "HAR", recorded in the repair log.
    coerced = ag_out[ag_out["field_id"] == "F001"]
    assert coerced["area_unit"].iloc[0] == "HAR"
    assert _applied(rep, "AG-007", frm="hectares", to="HAR")


# -------------------------------------------------------------------- media

def test_media_eidr_content_conformance():
    eidr_df = pd.DataFrame({
        "eidr_id": ["10.5240/7791-8534-2C23-9030-8610-5", "10.5240/XXXX-XXXX",
                    "10.9999/7791-8534-2C23-9030-8610-5",
                    "10.5240/1234-5678-9ABC-DEF0-1234-5", None],
        "title": ["Inception", "Unknown", "Bad Prefix", "Interstellar", "No ID"],
        "content_type": ["Movie", "Episode", "Movie", "Movie", "Season"],
        "release_date": ["2010-07-16", "2023", "not-a-date", "2014-11-07", "2022"],
        "country_of_origin": ["US", "US", "GB", "ZZ", "US"],
        "language": ["en", "en", "en", "en", "xx"],
        "runtime_seconds": [8820, 2700, 5400, 10140, None],
        "series_eidr_id": [None, "10.5240/ABCD-EFGH-1234-5678-IJKL-2", None, None, None],
    })
    _, rep = fd.clean(eidr_df, domain="media", media_type="content",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-C001")            # null eidr_id
    assert _violated(rep, "MD-C002")            # invalid EIDR DOI format
    assert _violated(rep, "MD-C005")            # invalid country ZZ
    assert _violated(rep, "MD-C006")            # invalid language 'xx'


def test_media_ddex_release_conformance():
    ddex_df = pd.DataFrame({
        "release_id": ["REL001", "REL002", None, "REL004", "REL005"],
        "icpn": ["012345678901", "5099902895529", "00000000000000",
                 "012345678900", None],
        "release_type": ["Album", "Single", "EP", "Compilation", "Podcast"],
        "title": ["Greatest Hits", "Summer Love", "EP Vol 1", "Best of", "Interviews"],
        "language": ["en", "es", "fr", "zz", "en"],
        "party_role": ["MainArtist", "FeaturedArtist", None, "InvalidRole", "Composer"],
        "party_id": ["ISNI:0000000", None, None, "ISNI:1111111", "ISNI:2222222"],
        "territory": ["US", "Worldwide", "GB", "ZZ", "CA"],
        "track_count": [14, 1, 5, 20, None],
        "release_date": ["2024-01-15", "2024-06-01", "not-a-date",
                         "2023-12-01", "2024-03-15"],
    })
    _, rep = fd.clean(ddex_df, domain="media", media_type="release",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-R001")            # null release_id
    assert _violated(rep, "MD-R003")            # invalid release_type 'Podcast'
    assert _violated(rep, "MD-R004")            # invalid language 'zz'
    assert _violated(rep, "MD-R005")            # invalid party_role
    assert _violated(rep, "MD-R010")            # party_role without party_id


# ----------------------------------------------------------- regression / errors

def test_generic_clean_has_no_domain_findings():
    generic_df = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C001"],
        "age": [25, None, 41, 25],
        "salary": ["$1,200.50", "$2,000", "N/A", "$1,200.50"],
        "segment": ["A", "B", None, "A"],
    })
    cleaned, rep = fd.clean(generic_df, target_column="segment",
                            id_columns=("customer_id",), return_report=True, verbose=False)
    assert cleaned is not None and rep is not None
    assert rep.domain is None
    assert rep.domain_findings == []
    assert rep.domain_repairs == []


def test_generic_clean_is_deterministic():
    plain = pd.DataFrame({"a": [1, 2, 2, None], "b": ["x", "y", "y", "z"]})
    pd.testing.assert_frame_equal(fd.clean(plain.copy(), verbose=False),
                                  fd.clean(plain.copy(), verbose=False))


def test_unknown_domain_raises():
    with pytest.raises(UnknownDomainError) as exc:
        fd.clean(pd.DataFrame({"a": [1]}), domain="unknown_xyz", verbose=False)
    assert "unknown_xyz" in str(exc.value).lower()


def test_healthcare_ambiguity_is_a_controlled_error():
    # A frame carrying both Observation- and Encounter-specific signals cannot
    # be auto-resolved and must raise a typed error, never a generic crash.
    from freshdata.domains.healthcare import AmbiguousFHIRResourceError
    ambiguous = pd.DataFrame({
        "patient_id": ["P1", "P2"],
        "observation_id": ["O1", "O2"],   # Observation signal
        "value_quantity": [72.0, 68.5],   # Observation signal
        "class_code": ["AMB", "IMP"],     # Encounter signal
        "period_start": ["2024-01-01", "2024-02-01"],  # Encounter signal
    })
    with pytest.raises(AmbiguousFHIRResourceError):
        fd.clean(ambiguous, domain="healthcare", verbose=False)
