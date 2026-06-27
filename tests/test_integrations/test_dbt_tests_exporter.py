"""Tests for the dbt generic-tests exporter (``export_dbt_tests``)."""

from __future__ import annotations

import pytest

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
                              rule_name="status_set", message="bad value",
                              extra={"value_set": ["open", "closed"]}),
        QualityFinding.create(severity="error", step="privacy", column="ssn",
                              rule_name="US_SSN", message="pii", sensitive=True),
        QualityFinding.create(severity="error", step="clean", column=None,
                              rule_name="row_count", message="too few rows"),
    ]


def test_export_writes_valid_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "models" / "schema.yml"
    text = fd.export_dbt_tests(_findings(), "orders", str(path))

    assert text.startswith("version: 2")
    assert path.exists()
    parsed = yaml.safe_load(text)
    assert parsed["version"] == 2
    model = parsed["models"][0]
    assert model["name"] == "orders"

    cols = {c["name"]: c["tests"] for c in model["columns"]}
    assert any("not_null" in t and t["not_null"]["config"]["severity"] == "error"
               for t in cols["email"])
    assert any("unique" in t and t["unique"]["config"]["severity"] == "warn"
               for t in cols["id"])
    accepted = next(t["accepted_values"] for t in cols["status"] if "accepted_values" in t)
    assert accepted["values"] == ["open", "closed"]
    # advanced finding -> custom freshdata_expectation generic test
    assert any("freshdata_expectation" in t for t in cols["ssn"])
    # table-level finding (column is None) -> model-level test
    assert any("freshdata_expectation" in t for t in model["tests"])


def test_severity_map_override(tmp_path):
    yaml = pytest.importorskip("yaml")
    text = fd.export_dbt_tests(_findings(), "orders", str(tmp_path / "s.yml"),
                               severity_map={"warning": "error"})
    parsed = yaml.safe_load(text)
    cols = {c["name"]: c["tests"] for c in parsed["models"][0]["columns"]}
    unique = next(t["unique"] for t in cols["id"] if "unique" in t)
    assert unique["config"]["severity"] == "error"


def test_accepts_a_report_object(tmp_path):
    yaml = pytest.importorskip("yaml")
    import pandas as pd
    _, report = fd.clean(pd.DataFrame({"a": [1, 2]}), return_report=True)
    text = fd.export_dbt_tests(report, "m", str(tmp_path / "s.yml"))
    parsed = yaml.safe_load(text)
    assert parsed["models"][0]["name"] == "m"


def test_strings_with_special_chars_are_quoted(tmp_path):
    yaml = pytest.importorskip("yaml")
    f = QualityFinding.create(severity="error", step="domain", column="weird:col",
                              rule_name="r", message="has: colon, and #hash")
    text = fd.export_dbt_tests([f], "m", str(tmp_path / "s.yml"))
    parsed = yaml.safe_load(text)  # must still be valid YAML
    assert parsed["models"][0]["columns"][0]["name"] == "weird:col"
