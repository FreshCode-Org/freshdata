"""``allowed_values`` compares by column dtype for numeric and boolean columns (#255)."""

from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

import freshdata as fd


def _allowed(findings):
    return {f.column: f for f in findings if f.rule_name == "context.allowed_values"}


def _validate(frame, context):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fd.validate(frame, context=context)


def test_issue_repro_float_and_bool_columns_pass():
    frame = pd.DataFrame({"rating": [1.0, 2.0, 3.0, None], "active": [True, False, True, False]})
    findings = _validate(
        frame, "Allowed rating values are 1, 2, 3. Allowed active values are true, false."
    )
    assert _allowed(findings) == {}


def test_float_column_with_decimal_allowed_values():
    frame = pd.DataFrame({"score": [1.5, 2.5, 1.5, None]})
    assert _allowed(_validate(frame, "Allowed score values are 1.5, 2.5.")) == {}


def test_float_column_violation_reports_typed_values():
    frame = pd.DataFrame({"score": [1.5, 2.5, 3.5, None]})
    finding = _allowed(_validate(frame, "Allowed score values are 1.5, 2.5."))["score"]
    assert finding.extra["value_set"] == [1.5, 2.5]
    assert finding.extra["n_violations"] == 1
    assert finding.observed_value == [3.5]


def test_int_column_exact_numeric_equality():
    frame = pd.DataFrame({"grade": [1, 2, 3, 4]})
    finding = _allowed(_validate(frame, "Allowed grade values are 1, 2, 3."))["grade"]
    assert finding.extra["value_set"] == [1, 2, 3]
    assert all(type(v) is int for v in finding.extra["value_set"])
    assert finding.extra["n_violations"] == 1
    assert finding.observed_value == [4]


def test_nullable_int_column_ignores_missing():
    frame = pd.DataFrame({"grade": pd.array([1, None, 2], dtype="Int64")})
    assert _allowed(_validate(frame, "Allowed grade values are 1, 2.")) == {}


def test_numeric_column_non_numeric_entries_cannot_match():
    frame = pd.DataFrame({"grade": [1, 2]})
    finding = _allowed(_validate(frame, "Allowed grade values are 1, low."))["grade"]
    assert finding.extra["value_set"] == [1, "low"]
    assert finding.extra["n_violations"] == 1
    assert finding.observed_value == [2]


@pytest.mark.parametrize("words", ["true, false", "yes, no", "TRUE, False", "1, 0", "Y, n"])
def test_bool_column_word_spellings(words):
    frame = pd.DataFrame({"active": [True, False, True]})
    findings = _validate(frame, f"Allowed active values are {words}.")
    assert _allowed(findings) == {}


def test_bool_column_violation_reports_typed_values():
    frame = pd.DataFrame({"active": [True, False, True]})
    finding = _allowed(_validate(frame, "Allowed active values are yes."))["active"]
    assert finding.extra["value_set"] == [True]
    assert finding.extra["n_violations"] == 1
    assert finding.observed_value == [False]


def test_nullable_boolean_column():
    frame = pd.DataFrame({"active": pd.array([True, None, False], dtype="boolean")})
    assert _allowed(_validate(frame, "Allowed active values are true, false.")) == {}
    finding = _allowed(_validate(frame, "Allowed active values are true."))["active"]
    assert finding.extra["value_set"] == [True]
    assert finding.extra["n_violations"] == 1


def test_string_column_behaviour_unchanged():
    frame = pd.DataFrame({"status": ["active", "zombie", "True", "1.0", None]})
    finding = _allowed(_validate(frame, "Allowed status values are active, true, 1."))["status"]
    assert finding.extra["value_set"] == ["active", "true", "1"]
    assert finding.extra["n_violations"] == 3
    assert finding.observed_value == ["1.0", "True", "zombie"]


def test_gx_export_uses_typed_value_set(tmp_path):
    frame = pd.DataFrame({"rating": [1.0, 2.0, 9.0], "active": [True, False, True]})
    findings = _validate(
        frame, "Allowed rating values are 1, 2, 3. Allowed active values are true."
    )
    path = tmp_path / "suite.json"
    suite = fd.export_gx_suite(findings, "s", str(path))
    value_sets = {
        e["kwargs"]["column"]: e["kwargs"]["value_set"]
        for e in suite["expectations"]
        if "value_set" in e["kwargs"]
    }
    assert value_sets == {"rating": [1, 2, 3], "active": [True]}
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == suite


def test_dbt_export_emits_typed_scalars(tmp_path):
    frame = pd.DataFrame({"rating": [1.0, 9.0], "active": [True, False]})
    findings = _validate(
        frame, "Allowed rating values are 1, 2.5. Allowed active values are true."
    )
    text = fd.export_dbt_tests(findings, "m", str(tmp_path / "schema.yml"))
    assert "- 1\n" in text and "- 2.5\n" in text
    assert "- true\n" in text
    assert '"1"' not in text and '"true"' not in text
