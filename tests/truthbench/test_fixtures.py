from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.truthbench import Disposition
from benchmarks.truthbench.fixtures import build_fixture
from benchmarks.truthbench.fixtures.base import FixtureBuilder, FixtureError

DOMAINS = ("minimal", "finance", "healthcare", "retail", "crm")


def _cell(fixture, row_id: str, column: str):
    return next(cell for cell in fixture.cells if cell.row_id == row_id and cell.column == column)


def _assert_cell(fixture, row_id: str, column: str, family: str, value, disposition: Disposition):
    cell = _cell(fixture, row_id, column)
    assert cell.family == family
    assert cell.disposition is disposition
    assert fixture.frame.at[row_id, column] == value
    return cell


@pytest.mark.parametrize("domain", DOMAINS[1:])
def test_domain_fixture_has_complete_oracle_and_required_dispositions(domain: str) -> None:
    fixture = build_fixture(domain, seed=1729)
    assert fixture.frame.shape[0] == 16
    assert {cell.disposition for cell in fixture.cells} == set(Disposition)
    assert len(fixture.row_cases) >= 2
    assert len(fixture.schema_cases) >= 4
    assert fixture.policy["reference_date"] == "2026-01-15"
    assert fixture.policy["timezone"] == "UTC"


@pytest.mark.parametrize("domain", DOMAINS[1:])
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
    _assert_cell(
        fixture, "fin-04", "ticker", "protected-ticker-policy-conflict", "AAPL", Disposition.REVIEW
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
    _assert_cell(
        fixture, "hc-04", "dose", "mg-mcg-unit-conversion", "5000 mcg", Disposition.REPAIR
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
        Disposition.REVIEW,
    )
    assert protected.sensitive
    assert sum(cell.family != "background" for cell in fixture.cells) >= 12


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
    first = build_fixture("minimal", seed=1729)
    second = build_fixture("minimal", seed=1729)
    assert first.fixture_hash == second.fixture_hash
    altered = FixtureBuilder(
        "v1", "minimal", first.pristine.copy(), schema={"different": True}
    ).build()
    assert altered.fixture_hash != first.fixture_hash
