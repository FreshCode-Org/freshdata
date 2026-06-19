"""Tests for the domain base interface and the shared config-driven engine."""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.domains import (
    ConfigDrivenValidator,
    DomainError,
    DomainValidator,
    Rule,
)
from freshdata.domains.base import _load_yaml_rules


class DummyPack(ConfigDrivenValidator):
    """Synthetic pack exercising every check/repair path via injected rules."""

    domain_name = "dummy"
    version = "0.1.0"
    schema_version = "t"
    canonical_fields = ("id", "status", "score", "name")
    required_fields = ("id", "status")
    id_fields = ("id",)
    aliases = {"status": (r"state",)}

    _RULES = [
        {"id": "D1", "name": "id non-null", "layer": "schema", "severity": "error",
         "field": "id", "check": "not_null"},
        {"id": "D2", "name": "required", "layer": "schema", "severity": "error",
         "fields": ["id", "status"], "check": "required"},
        {"id": "D3", "name": "score range", "layer": "format", "severity": "warning",
         "field": "score", "check": "range", "params": {"min": 0, "max": 100}},
        {"id": "D4", "name": "status enum", "layer": "reference", "severity": "warning",
         "field": "status", "check": "enum",
         "params": {"values": ["active", "inactive"], "case_insensitive": True}},
        {"id": "D5", "name": "name fill", "layer": "business", "severity": "info",
         "field": "name", "check": "custom", "params": {"func": "missing_name"},
         "repair": "fill_default", "repair_params": {"value": "unknown"}},
        {"id": "D6", "name": "neg reject", "layer": "business", "severity": "error",
         "field": "score", "check": "custom", "params": {"func": "negative_score"},
         "repair": "reject"},
    ]

    def register_extensions(self):
        self.register_check(
            "missing_name",
            lambda df, m, r: df.index[df[m.actual("name")].isna()].tolist(),
        )
        self.register_check(
            "negative_score",
            lambda df, m, r: df.index[
                pd.to_numeric(df[m.actual("score")], errors="coerce").fillna(0) < 0
            ].tolist(),
        )

    @property
    def rules(self):
        if self._rules is None:
            self._rules = [Rule.from_dict(d) for d in self._RULES]
        return self._rules


@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "id": [1, 2, 3, None],
        "state": ["active", "INACTIVE", "bogus", "active"],   # alias -> status
        "score": [50, 150, -5, 20],
        "name": ["a", None, "c", "d"],
    })


def test_domain_validator_is_abstract():
    with pytest.raises(TypeError):
        DomainValidator()  # abstract methods are unimplemented


def test_config_driven_requires_rules_path():
    class Bare(ConfigDrivenValidator):
        domain_name = "bare"

    with pytest.raises(DomainError, match="rules_path"):
        _ = Bare().rules


def test_column_detection_methods(dummy_df):
    mapping = DummyPack().detect_columns(dummy_df)
    assert mapping.actual("id") == "id"                # exact
    assert mapping.actual("status") == "state"         # regex alias
    methods = {entry["canonical"]: entry["method"] for entry in mapping.log}
    assert methods["status"] == "regex"
    assert methods["id"] == "exact"


def test_case_insensitive_and_override_detection():
    df = pd.DataFrame({"ID": [1], "state": ["active"], "sc": [1], "name": ["x"]})
    mapping = DummyPack(column_map={"sc": "score"}).detect_columns(df)
    assert mapping.actual("id") == "ID"                # case-insensitive
    assert mapping.actual("score") == "sc"             # user override
    methods = {e["canonical"]: e["method"] for e in mapping.log}
    assert methods["id"] == "case_insensitive"
    assert methods["score"] == "override"


def test_validate_runs_layers_and_scores(dummy_df):
    report = DummyPack().validate(dummy_df)
    fired = {r.rule_id for r in report.results if r.violated}
    assert {"D1", "D3", "D4", "D5", "D6"} <= fired   # D2 passes (cols present)
    assert 0.0 <= report.domain_trust_score <= 1.0
    assert report.domain_trust_score < 1.0
    # layer ordering is enforced
    layers = [r.layer for r in report.results]
    assert layers == sorted(
        layers,
        key=("schema", "format", "reference", "business", "semantic").index
    )


def test_validate_does_not_mutate(dummy_df):
    before = dummy_df.copy()
    DummyPack().validate(dummy_df)
    pd.testing.assert_frame_equal(dummy_df, before)


def test_repair_fill_and_reject_and_id_safety(dummy_df):
    pack = DummyPack()
    report = pack.validate(dummy_df)
    out, log = pack.repair(dummy_df, report)
    assert 2 not in out.index                          # D6 rejected the negative-score row
    assert out.loc[1, "name"] == "unknown"             # D5 filled the missing name (row 1)
    # id column (protected) is never mutated, even though id row 3 is null
    assert pd.isna(out.loc[3, "id"])
    assert all(a.column != "id" or a.status != "applied" for a in log)
    assert len(log) >= 1 and log.applied


def test_trust_score_is_one_for_clean_data():
    clean = pd.DataFrame({
        "id": [1, 2], "state": ["active", "inactive"], "score": [10, 90], "name": ["a", "b"],
    })
    report = DummyPack().validate(clean)
    assert report.domain_trust_score == 1.0
    assert report.passed and not report.errors


def test_report_and_log_serialization(dummy_df):
    pack = DummyPack()
    report = pack.validate(dummy_df)
    payload = report.to_dict()
    assert payload["domain"] == "dummy" and "results" in payload
    assert set(report.severity_counts) == {"info", "warning", "error"}
    assert "D1" in report.violation_index
    assert "trust=" in report.summary()
    _, log = pack.repair(dummy_df, report)
    assert "actions" in log.to_dict()


def test_describe_includes_versioning():
    desc = DummyPack().describe()
    assert desc["version"] == "0.1.0" and desc["schema_version"] == "t"
    assert desc["id_fields"] == ["id"]


@pytest.mark.parametrize("bad", [
    {"id": "X", "layer": "nope", "severity": "error", "field": "a", "check": "regex"},
    {"id": "X", "layer": "schema", "severity": "loud", "field": "a", "check": "regex"},
    {"id": "X", "layer": "schema", "severity": "error", "field": "a", "check": "regex",
     "repair": "explode"},
    {"id": "X", "layer": "schema", "severity": "error", "field": "a", "fields": ["b"],
     "check": "regex"},
])
def test_rule_validation_errors(bad):
    with pytest.raises(DomainError):
        Rule.from_dict(bad)


def test_rule_requires_schema_keys_and_list_fields():
    with pytest.raises(DomainError, match="missing required key"):
        Rule.from_dict({"id": "X"})
    with pytest.raises(DomainError, match="fields.*list"):
        Rule.from_dict({
            "id": "X",
            "layer": "schema",
            "severity": "error",
            "fields": "identifier",
            "check": "required",
        })


def test_unknown_check_and_reference_errors():
    df = pd.DataFrame({"id": [1], "status": ["active"], "score": [1], "name": ["x"]})
    pack = DummyPack()
    mapping = pack.detect_columns(df)
    with pytest.raises(DomainError, match="unknown check"):
        pack._dispatch_check(df, mapping, Rule.from_dict(
            {"id": "Z", "name": "z", "layer": "format", "severity": "info",
             "field": "score", "check": "bogus"}))
    with pytest.raises(DomainError, match="reference"):
        pack.load_reference_values("nonexistent")


def test_yaml_loader_accepts_list_root(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        "- id: X\n  layer: schema\n  severity: error\n  field: id\n  check: not_null\n",
        encoding="utf-8",
    )
    assert _load_yaml_rules(str(path))[0]["id"] == "X"


def test_yaml_loader_rejects_invalid_root(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("not-a-rule-list\n", encoding="utf-8")
    with pytest.raises(DomainError, match="mapping or a list"):
        _load_yaml_rules(str(path))


def test_duplicate_rule_ids_are_rejected():
    class DuplicateRules(DummyPack):
        _RULES = [DummyPack._RULES[0], DummyPack._RULES[0]]

        @property
        def rules(self):
            return super().rules

    # Exercise the shared config validation through a temporary YAML-backed pack.
    pack = DuplicateRules()
    pack._rules = [Rule.from_dict(raw) for raw in pack._RULES]
    with pytest.raises(DomainError, match="duplicate rule id"):
        pack._validate_rules(pack._rules)
