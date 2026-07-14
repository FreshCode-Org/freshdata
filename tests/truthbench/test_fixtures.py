from __future__ import annotations

import json

import pandas as pd
import pytest
from benchmarks.truthbench import Disposition
from benchmarks.truthbench.fixtures import build_fixture
from benchmarks.truthbench.fixtures.base import FixtureBuilder, FixtureError

DOMAINS = ("minimal", "finance", "healthcare", "retail", "crm")


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


@pytest.mark.parametrize(
    ("domain", "needles"),
    [
        ("finance", ("apple", "Apple", "AAPL", "₹1,23,456.70", "01/02/2025")),
        ("healthcare", ("98.6", "5000 mcg", "MRN", "FHIR")),
        ("retail", ("SKU", "GTIN", "&amp;", "mojibake")),
        ("crm", ("apple", "Apple", ".invalid", "SSN")),
    ],
)
def test_domain_fixture_contains_adversarial_family_markers(
    domain: str, needles: tuple[str, ...]
) -> None:
    fixture = build_fixture(domain, seed=1729)
    serialized = json.dumps(fixture.to_dict(), ensure_ascii=False).casefold()
    for needle in needles:
        assert needle.casefold() in serialized


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
