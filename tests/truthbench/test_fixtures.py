from __future__ import annotations

import json

import pandas as pd
import pytest
from benchmarks.truthbench import Disposition
from benchmarks.truthbench.fixtures import build_fixture
from benchmarks.truthbench.fixtures.base import FixtureBuilder, FixtureError

DOMAINS = ("minimal",)


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
    assert cell.expected_output.type_label == "python.float[float64]"
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
