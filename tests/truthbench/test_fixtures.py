from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.truthbench import Disposition
from benchmarks.truthbench.fixtures import build_fixture
from benchmarks.truthbench.fixtures.base import FixtureBuilder, FixtureError

DOMAINS = (
    "crm",
    "education",
    "finance",
    "government",
    "healthcare",
    "insurance",
    "logistics",
    "retail",
)


def _cell(fixture, row_id: str, column: str):
    return next(cell for cell in fixture.cells if cell.row_id == row_id and cell.column == column)


def _assert_cell(fixture, row_id: str, column: str, family: str, value, disposition: Disposition):
    cell = _cell(fixture, row_id, column)
    assert cell.family == family
    assert cell.disposition is disposition
    assert fixture.frame.at[row_id, column] == value
    return cell


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_fixture_has_complete_oracle_and_required_dispositions(domain: str) -> None:
    fixture = build_fixture(domain, seed=1729)
    assert fixture.frame.shape[0] == 16
    assert {cell.disposition for cell in fixture.cells} == set(Disposition)
    assert len(fixture.row_cases) >= 2
    assert len(fixture.schema_cases) >= 4
    assert fixture.policy["reference_date"] == "2026-01-15"
    assert fixture.policy["timezone"] == "UTC"


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_fixture_is_byte_deterministic_for_approved_seeds(domain: str) -> None:
    for seed in (1729, 2718):
        first = build_fixture(domain, seed=seed)
        second = build_fixture(domain, seed=seed)
        assert first.fixture_hash == second.fixture_hash
        assert first.to_dict() == second.to_dict()


def test_finance_content_families_are_labeled_on_actual_cells() -> None:
    fixture = build_fixture("finance", seed=1729)
    _assert_cell(fixture, "fin-03", "price", "semantic-apple-price", "apple", Disposition.REVIEW)
    _assert_cell(fixture, "fin-04", "company", "apple-company", "Apple", Disposition.PRESERVE)
    # Corrected oracle: a protected column value is preserved, not reviewed.
    _assert_cell(
        fixture, "fin-04", "ticker", "protected-ticker-valid", "AAPL", Disposition.PRESERVE
    )
    _assert_cell(fixture, "fin-02", "price", "zero-price", "0.00", Disposition.PRESERVE)
    _assert_cell(fixture, "fin-05", "price", "negative-value", -4.5, Disposition.FLAG)
    _assert_cell(fixture, "fin-06", "price", "extreme-value", 9999999999.99, Disposition.FLAG)
    _assert_cell(
        fixture, "fin-07", "price", "indian-grouped-currency", "₹1,23,456.70", Disposition.REPAIR
    )
    _assert_cell(
        fixture, "fin-10", "trade_date", "ambiguous-date-format", "01/02/2025", Disposition.REVIEW
    )
    pii = _assert_cell(
        fixture,
        "fin-11",
        "memo",
        "invisible-pii-memo",
        "tb.finance+memo@example.invalid",
        Disposition.FLAG,
    )
    assert pii.sensitive
    tail = _assert_cell(
        fixture,
        "fin-16",
        "account_id",
        "tail-row-account-canary",
        "TB-FIN-ACCOUNT-TAIL",
        Disposition.FLAG,
    )
    assert tail.sensitive
    assert sum(cell.family != "background" for cell in fixture.cells) >= 12


def test_finance_currency_conflict_zero_width_memo_and_exact_case_families() -> None:
    fixture = build_fixture("finance", seed=1729)
    currencies = {
        fixture.frame.at[cell.row_id, cell.column]
        for cell in fixture.cells
        if cell.family == "supported-currency"
    }
    assert currencies == {"EUR", "INR"}
    assert {fixture.frame.at[row, "currency"] for row in fixture.frame.index} >= {
        "USD",
        "EUR",
        "INR",
    }
    zero_width = next(cell for cell in fixture.cells if cell.family == "zero-width-memo")
    assert "\u200b" in fixture.frame.at[zero_width.row_id, zero_width.column]
    assert zero_width.disposition is Disposition.FLAG
    assert {case.family for case in fixture.row_cases} == {"exact-duplicate", "removed-row"}
    assert {case.family for case in fixture.schema_cases} == {
        "added-column",
        "removed-column",
        "renamed-column",
        "reordered-columns",
        "type-drifted-column",
    }


def test_healthcare_content_families_use_valid_rare_reference_codes() -> None:
    fixture = build_fixture("healthcare", seed=1729)
    rare_icd = _assert_cell(
        fixture, "hc-04", "diagnosis_code", "rare-icd-valid", "Z79.4", Disposition.PRESERVE
    )
    rare_loinc = _assert_cell(
        fixture, "hc-04", "loinc", "rare-loinc-valid", "9843-4", Disposition.PRESERVE
    )
    references = (
        Path(__file__).parents[2] / "src" / "freshdata" / "domains" / "healthcare" / "reference"
    )
    assert (
        fixture.frame.at["hc-04", "diagnosis_code"]
        in json.loads((references / "icd10_common.json").read_text())["codes"]
    )
    assert (
        fixture.frame.at["hc-04", "loinc"]
        in json.loads((references / "loinc_common.json").read_text())["codes"]
    )
    assert re.fullmatch(
        r"[A-TV-Z][0-9]{2}(?:\.[0-9]{1,4})?", fixture.frame.at["hc-04", "diagnosis_code"]
    )
    assert re.fullmatch(r"[0-9]{1,5}-[0-9]", fixture.frame.at["hc-04", "loinc"])
    assert rare_icd.disposition is rare_loinc.disposition is Disposition.PRESERVE
    _assert_cell(
        fixture, "hc-04", "temperature", "celsius-fahrenheit-conflict", 98.6, Disposition.REVIEW
    )
    _assert_cell(fixture, "hc-01", "dose", "dose-unit-valid", "5 mg", Disposition.PRESERVE)
    # Corrected oracle: a float repair contradicted the same column's
    # dose-unit-valid PRESERVE strings; a cross-unit dose conversion is a
    # clinical decision routed to a human.
    _assert_cell(
        fixture, "hc-04", "dose", "mg-mcg-unit-conversion", "5000 mcg", Disposition.REVIEW
    )
    _assert_cell(fixture, "hc-05", "event_date", "partial-date", "2025-01", Disposition.REVIEW)
    _assert_cell(
        fixture, "hc-06", "event_date", "fhir-date", "2025-01-15T12:00:00Z", Disposition.REPAIR
    )
    _assert_cell(fixture, "hc-07", "event_date", "impossible-date", "2025-02-30", Disposition.FLAG)
    _assert_cell(
        fixture, "hc-08", "patient_name", "decomposed-unicode", "Jose\u0301", Disposition.REPAIR
    )
    for row_id, value, family in (
        ("hc-09", "TB-HC-PHI-0001", "phi-in-notes"),
        ("hc-11", "TB-HC-MRN-NOTE", "mrn-in-notes"),
        ("hc-12", "555-0112", "phone-in-notes"),
    ):
        cell = _assert_cell(fixture, row_id, "notes", family, value, Disposition.FLAG)
        assert cell.sensitive
    assert sum(cell.family != "background" for cell in fixture.cells) >= 12


def test_healthcare_protected_dob_tail_and_exact_case_families() -> None:
    fixture = build_fixture("healthcare", seed=1729)
    dob = _assert_cell(
        fixture,
        "hc-15",
        "dob",
        "protected-dob-repair-conflict",
        "01/01/1980",
        Disposition.PRESERVE,
    )
    tail = _assert_cell(
        fixture,
        "hc-16",
        "mrn",
        "mrn-tail-canary",
        "TB-HC-MRN-TAIL",
        Disposition.PRESERVE,
    )
    assert dob.column in fixture.protected_columns
    assert tail.sensitive
    assert {case.family for case in fixture.row_cases} == {"exact-duplicate", "removed-row"}
    assert {case.family for case in fixture.schema_cases} == {
        "added-column",
        "removed-column",
        "renamed-column",
        "reordered-columns",
        "type-drifted-column",
    }


def test_retail_content_families_are_labeled_on_actual_cells_and_schema_cases() -> None:
    fixture = build_fixture("retail", seed=1729)
    _assert_cell(fixture, "ret-01", "sku", "leading-zero-sku", "001234", Disposition.PRESERVE)
    _assert_cell(
        fixture, "ret-01", "gtin", "leading-zero-gtin", "00012345678905", Disposition.PRESERVE
    )
    _assert_cell(fixture, "ret-02", "price", "free-item", "0.00", Disposition.PRESERVE)
    _assert_cell(fixture, "ret-06", "quantity", "return-quantity", -2, Disposition.FLAG)
    _assert_cell(
        fixture, "ret-03", "price", "mixed-decimal-grouping", "1.234,56", Disposition.REPAIR
    )
    _assert_cell(fixture, "ret-04", "price", "mixed-currency", "₹1,23,456.70", Disposition.REVIEW)
    _assert_cell(
        fixture,
        "ret-03",
        "product_name",
        "html-entity-mojibake",
        "CafÃ© &amp; Tea",
        Disposition.REVIEW,
    )
    _assert_cell(
        fixture, "ret-04", "product_name", "multilingual-product", "茶", Disposition.PRESERVE
    )
    for row_id, column, value, family in (
        ("ret-07", "review", "customer@example.invalid", "email-in-review"),
        ("ret-08", "card", "TB-RETAIL-CARD-REVIEW", "card-in-review"),
    ):
        cell = _assert_cell(fixture, row_id, column, family, value, Disposition.REVIEW)
        assert cell.sensitive
    assert {case.family for case in fixture.schema_cases} >= {
        "added-column",
        "removed-column",
        "renamed-column",
        "reordered-columns",
        "type-drifted-column",
    }
    assert sum(cell.family != "background" for cell in fixture.cells) >= 12


def test_crm_content_families_are_labeled_on_actual_cells() -> None:
    fixture = build_fixture("crm", seed=1729)
    _assert_cell(
        fixture, "crm-02", "first_name", "combining-unicode-name", "Jose\u0301", Disposition.REPAIR
    )
    _assert_cell(
        fixture,
        "crm-03",
        "email",
        "reserved-invalid-email",
        "miyuki@example.invalid",
        Disposition.PRESERVE,
    )
    _assert_cell(fixture, "crm-04", "phone", "spaced-phone", "555 0101", Disposition.REPAIR)
    _assert_cell(fixture, "crm-05", "country", "ambiguous-country", "US/CA", Disposition.REVIEW)
    _assert_cell(fixture, "crm-06", "language", "ambiguous-language", "en/fr", Disposition.REVIEW)
    _assert_cell(
        fixture, "crm-07", "signup_date", "ambiguous-date", "01/02/2025", Disposition.REVIEW
    )
    _assert_cell(
        fixture,
        "crm-08",
        "lifecycle",
        "lifecycle-contradiction",
        "lead|churned",
        Disposition.REVIEW,
    )
    _assert_cell(
        fixture,
        "crm-11",
        "email",
        "zero-width-email",
        "lead\u200b@example.invalid",
        Disposition.FLAG,
    )
    ssn = _assert_cell(
        fixture, "crm-10", "notes", "hidden-ssn", "TB-CRM-SSN-0001", Disposition.FLAG
    )
    assert ssn.sensitive
    _assert_cell(
        fixture, "crm-03", "lead_source", "apple-lead-source", "apple", Disposition.PRESERVE
    )
    _assert_cell(fixture, "crm-03", "employer", "apple-employer", "Apple", Disposition.PRESERVE)
    protected = _assert_cell(
        fixture,
        "crm-16",
        "customer_id",
        "protected-customer-id",
        "TB-CRM-CUSTOMER-TAIL",
        Disposition.PRESERVE,
    )
    assert protected.sensitive
    assert sum(cell.family != "background" for cell in fixture.cells) >= 12


def test_crm_has_exact_row_and_schema_case_family_sets() -> None:
    fixture = build_fixture("crm", seed=1729)
    assert {case.family for case in fixture.row_cases} == {"exact-duplicate", "removed-row"}
    assert {case.family for case in fixture.schema_cases} == {
        "added-column",
        "removed-column",
        "renamed-column",
        "reordered-columns",
        "type-drifted-column",
    }


def test_logistics_content_families_are_labeled_on_actual_cells() -> None:
    fixture = build_fixture("logistics", seed=1729)
    _assert_cell(
        fixture, "log-04", "destination_code", "rare-unlocode-valid", "INBOM", Disposition.PRESERVE
    )
    # Corrected oracle: converting weight to kg would contradict the row's
    # own weight_unit column ('lb', PRESERVE); cross-unit conversion is
    # routed to a human.
    _assert_cell(fixture, "log-02", "weight", "kg-lb-conversion", "10 lb", Disposition.REVIEW)
    _assert_cell(fixture, "log-02", "weight_unit", "weight-unit-lb", "lb", Disposition.PRESERVE)
    _assert_cell(
        fixture, "log-03", "temperature", "celsius-fahrenheit-conflict", 98.6, Disposition.REVIEW
    )
    _assert_cell(
        fixture, "log-03", "temperature_unit", "temperature-unit-f", "F", Disposition.PRESERVE
    )
    _assert_cell(
        fixture,
        "log-05",
        "delivery_window",
        "cross-timezone-window",
        "2026-01-15 23:30-2026-01-16 01:00",
        Disposition.REVIEW,
    )
    _assert_cell(
        fixture,
        "log-05",
        "timezone",
        "cross-timezone-window",
        "Asia/Kolkata→America/New_York",
        Disposition.REVIEW,
    )
    _assert_cell(
        fixture,
        "log-06",
        "transport_time",
        "twentyfour-hour-transport",
        "24:00",
        Disposition.REPAIR,
    )
    for row_id, column, family in (
        ("log-09", "address", "address-pii"),
        ("log-15", "tracking_status", "late-tracking-canary"),
    ):
        cell = _assert_cell(
            fixture, row_id, column, family, fixture.frame.at[row_id, column], Disposition.FLAG
        )
        if family == "address-pii":
            assert cell.sensitive
    protected = _assert_cell(
        fixture,
        "log-16",
        "shipment_id",
        "protected-shipment-id-conflict",
        "TB-LOG-SHIPMENT-TAIL",
        Disposition.PRESERVE,
    )
    assert protected.sensitive


def test_government_content_families_are_labeled_on_actual_cells() -> None:
    fixture = build_fixture("government", seed=1729)
    _assert_cell(
        fixture, "gov-01", "district_id", "leading-zero-district-id", "007", Disposition.PRESERVE
    )
    _assert_cell(
        fixture, "gov-02", "case_id", "leading-zero-case-id", "000123", Disposition.PRESERVE
    )
    _assert_cell(
        fixture,
        "gov-03",
        "agency",
        "indian-international-grouping",
        "भारत सरकार / Government of India",
        Disposition.PRESERVE,
    )
    _assert_cell(
        fixture,
        "gov-05",
        "fiscal_year",
        "fiscal-calendar-year-conflict",
        "2025-26",
        Disposition.REVIEW,
    )
    _assert_cell(
        fixture,
        "gov-07",
        "language",
        "multilingual-agency",
        "हिन्दी / English",
        Disposition.PRESERVE,
    )
    _assert_cell(
        fixture, "gov-06", "encoding", "mixed-legacy-encoding", "CafÃ©", Disposition.REPAIR
    )
    restricted = _assert_cell(
        fixture,
        "gov-09",
        "notes",
        "restricted-national-id",
        "TB-GOV-NATIONAL-ID-0001",
        Disposition.FLAG,
    )
    assert restricted.sensitive
    _assert_cell(
        fixture,
        "gov-11",
        "retention_policy",
        "contradictory-retention-repair-policy",
        "retain 7 years",
        Disposition.REVIEW,
    )
    tail = _assert_cell(
        fixture,
        "gov-16",
        "case_id",
        "protected-case-id-conflict",
        "TB-GOV-CASE-TAIL",
        Disposition.PRESERVE,
    )
    assert tail.sensitive


def test_education_content_families_are_labeled_on_actual_cells() -> None:
    fixture = build_fixture("education", seed=1729)
    _assert_cell(
        fixture, "edu-01", "student_id", "leading-zero-student-id", "000123", Disposition.PRESERVE
    )
    _assert_cell(
        fixture, "edu-02", "grade_letter", "letter-grade-scale", "A-", Disposition.PRESERVE
    )
    _assert_cell(fixture, "edu-03", "score_percent", "zero-score", 0, Disposition.PRESERVE)
    _assert_cell(
        fixture, "edu-04", "school_year", "school-year-ambiguity", "2025/26", Disposition.REVIEW
    )
    _assert_cell(
        fixture,
        "edu-05",
        "enrollment_date",
        "enrollment-date-ordering",
        "2026-02-01",
        Disposition.REVIEW,
    )
    _assert_cell(fixture, "edu-08", "gpa", "gpa-scale", 4.0, Disposition.PRESERVE)
    guardian = _assert_cell(
        fixture,
        "edu-09",
        "guardian_email",
        "guardian-contact-pii",
        "guardian@example.invalid",
        Disposition.FLAG,
    )
    assert guardian.sensitive
    phone = _assert_cell(
        fixture,
        "edu-10",
        "guardian_phone",
        "guardian-contact-pii",
        "555-0110",
        Disposition.FLAG,
    )
    assert phone.sensitive
    ferpa = _assert_cell(
        fixture,
        "edu-10",
        "ferpa_notes",
        "ferpa-sensitive-notes",
        "TB-EDU-FERPA-0001",
        Disposition.FLAG,
    )
    assert ferpa.sensitive
    protected = _assert_cell(
        fixture,
        "edu-16",
        "grade_letter",
        "protected-grade-policy-conflict",
        "A",
        Disposition.PRESERVE,
    )
    assert protected.column in fixture.protected_columns


def test_insurance_content_families_are_labeled_on_actual_cells() -> None:
    fixture = build_fixture("insurance", seed=1729)
    _assert_cell(
        fixture, "ins-01", "policy_number", "policy-id-format", "00012345", Disposition.PRESERVE
    )
    _assert_cell(
        fixture, "ins-02", "claim_id", "claim-id-format", "CLM-000123", Disposition.PRESERVE
    )
    _assert_cell(
        fixture,
        "ins-04",
        "reserve_currency",
        "premium-reserve-currency-conflict",
        "EUR",
        Disposition.REVIEW,
    )
    _assert_cell(
        fixture, "ins-05", "reserve", "negative-reserve-review", -250.0, Disposition.REVIEW
    )
    _assert_cell(
        fixture,
        "ins-06",
        "report_date",
        "incident-report-date-ordering",
        "2025-01-01",
        Disposition.REVIEW,
    )
    _assert_cell(
        fixture,
        "ins-07",
        "state",
        "state-transition-contradiction",
        "open|closed",
        Disposition.REVIEW,
    )
    claimant = _assert_cell(
        fixture,
        "ins-09",
        "claimant_name",
        "claimant-pii",
        "TB-INS-CLAIMANT-0001",
        Disposition.FLAG,
    )
    assert claimant.sensitive
    medical = _assert_cell(
        fixture,
        "ins-10",
        "loss_description",
        "medical-loss-text",
        "TB-INS-MEDICAL-0001",
        Disposition.FLAG,
    )
    assert medical.sensitive
    protected = _assert_cell(
        fixture,
        "ins-16",
        "policy_number",
        "protected-policy-number-conflict",
        "TB-INS-POLICY-TAIL",
        Disposition.PRESERVE,
    )
    assert protected.sensitive


def test_eight_domain_corpus_contains_required_trap_categories() -> None:
    expected_families = {
        "logistics": {
            "rare-unlocode-valid",
            "kg-lb-conversion",
            "weight-unit-lb",
            "celsius-fahrenheit-conflict",
            "temperature-unit-f",
            "cross-timezone-window",
            "twentyfour-hour-transport",
            "address-pii",
            "late-tracking-canary",
            "protected-shipment-id-conflict",
        },
        "government": {
            "leading-zero-district-id",
            "leading-zero-case-id",
            "indian-international-grouping",
            "multilingual-agency",
            "fiscal-calendar-year-conflict",
            "restricted-national-id",
            "mixed-legacy-encoding",
            "contradictory-retention-repair-policy",
            "protected-case-id-conflict",
        },
        "education": {
            "leading-zero-student-id",
            "letter-grade-scale",
            "percentage-scale",
            "gpa-scale",
            "school-year-ambiguity",
            "zero-score",
            "enrollment-date-ordering",
            "guardian-contact-pii",
            "ferpa-sensitive-notes",
            "protected-grade-policy-conflict",
        },
        "insurance": {
            "policy-id-format",
            "claim-id-format",
            "grouped-premium",
            "premium-reserve-currency-conflict",
            "negative-reserve-review",
            "incident-report-date-ordering",
            "state-transition-contradiction",
            "claimant-pii",
            "medical-loss-text",
            "protected-policy-number-conflict",
        },
    }
    for domain, required in expected_families.items():
        actual = {
            cell.family
            for cell in build_fixture(domain, seed=1729).cells
            if cell.family and cell.family != "background"
        }
        assert required <= actual


def test_required_domain_families_match_actual_values_and_dispositions() -> None:
    logistics = build_fixture("logistics", seed=1729)
    for row, column, family, logistics_value, disposition in (
        ("log-02", "weight", "kg-lb-conversion", "10 lb", Disposition.REVIEW),
        ("log-02", "weight_unit", "weight-unit-lb", "lb", Disposition.PRESERVE),
        ("log-03", "temperature_unit", "temperature-unit-f", "F", Disposition.PRESERVE),
    ):
        _assert_cell(logistics, row, column, family, logistics_value, disposition)

    government = build_fixture("government", seed=1729)
    for row, column, family, government_value, disposition in (
        ("gov-01", "district_id", "leading-zero-district-id", "007", Disposition.PRESERVE),
        ("gov-02", "case_id", "leading-zero-case-id", "000123", Disposition.PRESERVE),
        (
            "gov-03",
            "agency",
            "indian-international-grouping",
            "भारत सरकार / Government of India",
            Disposition.PRESERVE,
        ),
        ("gov-07", "language", "multilingual-agency", "हिन्दी / English", Disposition.PRESERVE),
        ("gov-05", "fiscal_year", "fiscal-calendar-year-conflict", "2025-26", Disposition.REVIEW),
        (
            "gov-16",
            "case_id",
            "protected-case-id-conflict",
            "TB-GOV-CASE-TAIL",
            Disposition.PRESERVE,
        ),
    ):
        _assert_cell(government, row, column, family, government_value, disposition)

    education = build_fixture("education", seed=1729)
    for row, column, family, education_value, disposition in (
        ("edu-01", "student_id", "leading-zero-student-id", "000123", Disposition.PRESERVE),
        ("edu-02", "grade_letter", "letter-grade-scale", "A-", Disposition.PRESERVE),
        ("edu-07", "score_percent", "percentage-scale", "95%", Disposition.REPAIR),
        ("edu-08", "gpa", "gpa-scale", 4.0, Disposition.PRESERVE),
        ("edu-16", "grade_letter", "protected-grade-policy-conflict", "A", Disposition.PRESERVE),
    ):
        _assert_cell(education, row, column, family, education_value, disposition)
    percentage = _cell(education, "edu-07", "score_percent")
    assert percentage.expected_output is not None
    assert percentage.expected_output.display == "95.0"

    insurance = build_fixture("insurance", seed=1729)
    for row, column, family, value, disposition in (
        ("ins-01", "policy_number", "policy-id-format", "00012345", Disposition.PRESERVE),
        ("ins-02", "claim_id", "claim-id-format", "CLM-000123", Disposition.PRESERVE),
        ("ins-03", "premium", "grouped-premium", "1,000.00", Disposition.REPAIR),
        ("ins-09", "claimant_name", "claimant-pii", "TB-INS-CLAIMANT-0001", Disposition.FLAG),
        (
            "ins-16",
            "policy_number",
            "protected-policy-number-conflict",
            "TB-INS-POLICY-TAIL",
            Disposition.PRESERVE,
        ),
    ):
        _assert_cell(insurance, row, column, family, value, disposition)


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_physical_cell_has_exactly_one_label(domain: str) -> None:
    fixture = build_fixture(domain, seed=1729)
    expected = {(str(row), str(col)) for row in fixture.frame.index for col in fixture.frame}
    actual = {(cell.row_id, cell.column) for cell in fixture.cells}
    assert actual == expected
    assert len(fixture.cells) == len(expected)
    fixture.validate()


def test_injected_case_replaces_default_preserve_label(minimal_fixture) -> None:
    cell = next(
        cell for cell in minimal_fixture.cells if cell.row_id == "r2" and cell.column == "amount"
    )
    assert cell.disposition is Disposition.REPAIR
    assert cell.expected_output is not None
    assert cell.expected_output.type_label == "python.float"
    assert minimal_fixture.frame.at["r2", "amount"] == "2.50"


def test_sensitive_cells_have_canary_ids_and_no_raw_canary_in_hash(minimal_fixture) -> None:
    frame = pd.DataFrame(
        {"notes": ["ordinary", "tb.person+7@example.invalid"]}, index=["r1", "r2"]
    )
    builder = FixtureBuilder("v1", "minimal", frame)
    builder.inject(
        "r2",
        "notes",
        "tb.person+7@example.invalid",
        "flag",
        family="email",
        sensitive=True,
    )
    fixture = builder.build()
    sensitive = fixture.cells[1]
    assert sensitive.sensitive and sensitive.canary_id
    assert sensitive.canary_id in fixture.pii_canaries
    assert "tb.person+7@example.invalid" not in json.dumps(fixture.to_dict())


@pytest.mark.parametrize(
    "value",
    [
        "alice@gmail.com TB-1",
        "TB-1 alice@gmail.com",
        "415-555-0107",
        "555-0107 9876543210",
        "alice@gmail.com",
        "123-45-6789",
    ],
)
def test_sensitive_canaries_must_be_whole_value_synthetic_forms(value: str) -> None:
    frame = pd.DataFrame({"notes": ["ordinary"]}, index=["r1"])
    builder = FixtureBuilder("v1", "minimal", frame)
    with pytest.raises(FixtureError, match="synthetic PII"):
        builder.inject("r1", "notes", value, "flag", family="pii", sensitive=True)


def test_builder_rejects_missing_rows_duplicate_rows_unknown_cells_and_bad_repairs() -> None:
    with pytest.raises(FixtureError, match="row IDs"):
        FixtureBuilder("v1", "minimal", pd.DataFrame({"a": [1, 2]}, index=["r", "r"]))

    builder = FixtureBuilder("v1", "minimal", pd.DataFrame({"a": [1]}, index=["r1"]))
    with pytest.raises(FixtureError, match="unknown cell"):
        builder.inject("r9", "a", 2, "flag", family="x")
    with pytest.raises(FixtureError, match="unknown cell"):
        builder.inject("r1", "missing", 2, "flag", family="x")
    with pytest.raises(FixtureError, match="expected"):
        builder.inject("r1", "a", "2", "repair", family="x")


def test_repair_expected_output_cannot_be_contradictory() -> None:
    builder = FixtureBuilder("v1", "minimal", pd.DataFrame({"a": [1]}, index=["r1"]))
    builder.inject("r1", "a", "2", "repair", expected=2, family="x")
    with pytest.raises(FixtureError, match="contradictory"):
        builder.inject("r1", "a", "3", "repair", expected=3, family="x")
    assert builder.frame.at["r1", "a"] == "2"


def test_fixture_hash_is_stable_and_includes_metadata() -> None:
    first = build_fixture("finance", seed=1729)
    second = build_fixture("finance", seed=1729)
    assert first.fixture_hash == second.fixture_hash
    altered = FixtureBuilder(
        "v1", "finance", first.pristine.copy(), schema={"different": True}
    ).build()
    assert altered.fixture_hash != first.fixture_hash
