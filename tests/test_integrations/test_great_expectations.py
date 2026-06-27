"""Tests for the Great Expectations suite exporter (``export_gx_suite``)."""

from __future__ import annotations

import json

import freshdata as fd
from freshdata import QualityFinding


def _findings() -> list[QualityFinding]:
    return [
        QualityFinding.create(severity="error", step="domain", column="email",
                              rule_name="email_not_null", message="nulls",
                              expected_condition="not_null"),
        QualityFinding.create(severity="warning", step="domain", column="id",
                              rule_name="id_unique", message="dups",
                              expected_condition="unique"),
        QualityFinding.create(severity="warning", step="domain", column="status",
                              rule_name="status_set", message="bad",
                              extra={"value_set": ["open", "closed"]}),
        QualityFinding.create(severity="warning", step="domain", column="code",
                              rule_name="code_re", message="format",
                              extra={"regex": "^[A-Z]{3}$"}),
        QualityFinding.create(severity="warning", step="domain", column="qty",
                              rule_name="qty_range", message="range",
                              extra={"min_value": 0, "max_value": 100}),
        # advanced / table-level -> not expressible as a column expectation
        QualityFinding.create(severity="error", step="clean", column=None,
                              rule_name="row_count", message="few rows"),
    ]


def test_export_writes_valid_suite(tmp_path):
    path = tmp_path / "suite.json"
    suite = fd.export_gx_suite(_findings(), "orders_suite", str(path))

    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk == suite
    assert suite["expectation_suite_name"] == "orders_suite"

    types = {e["expectation_type"] for e in suite["expectations"]}
    assert types == {
        "expect_column_values_to_not_be_null",
        "expect_column_values_to_be_unique",
        "expect_column_values_to_be_in_set",
        "expect_column_values_to_match_regex",
        "expect_column_values_to_be_between",
    }
    # the table-level finding is counted as skipped, not silently dropped
    assert suite["meta"]["freshdata"]["n_skipped"] == 1
    assert suite["meta"]["freshdata"]["n_expectations"] == 5


def test_kwargs_and_meta_are_populated(tmp_path):
    suite = fd.export_gx_suite(_findings(), "s", str(tmp_path / "s.json"))
    by_type = {e["expectation_type"]: e for e in suite["expectations"]}
    in_set = by_type["expect_column_values_to_be_in_set"]["kwargs"]
    assert in_set["value_set"] == ["open", "closed"]
    assert by_type["expect_column_values_to_match_regex"]["kwargs"]["regex"] == "^[A-Z]{3}$"
    between = by_type["expect_column_values_to_be_between"]["kwargs"]
    assert (between["min_value"], between["max_value"]) == (0, 100)
    not_null_meta = by_type["expect_column_values_to_not_be_null"]["meta"]["freshdata"]
    assert not_null_meta["severity"] == "error"


def test_duplicate_findings_are_deduplicated(tmp_path):
    f = QualityFinding.create(severity="error", step="domain", column="email",
                              rule_name="x", message="m", expected_condition="not_null")
    g = QualityFinding.create(severity="warning", step="domain", column="email",
                              rule_name="y", message="m2", expected_condition="not_null")
    suite = fd.export_gx_suite([f, g], "s", str(tmp_path / "s.json"))
    assert len(suite["expectations"]) == 1
