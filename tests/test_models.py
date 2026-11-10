import pytest
from freshdata._core.models import (
    ActionRecord,
    CleanConfig,
    CleanPlan,
    CleanReport,
    ColumnProfile,
    ColumnRule,
    ProfileReport,
)

def test_clean_config_defaults():
    cfg = CleanConfig()
    assert cfg.strategy == "balanced"
    assert cfg.drop_duplicates is False
    assert cfg.default_outliers == "preserve"
    assert cfg.strict_backend is True
    assert cfg.iqr_factor == 1.5

def test_column_rule_immutability():
    rule = ColumnRule(role="numeric", impute="median", outliers="clip", protected=False)
    with pytest.raises(Exception):
        rule.role = "categorical"  # type: ignore

def test_action_record_serialization():
    rec = ActionRecord(
        stage="missing",
        column="age",
        action="impute_median",
        parameters={"median": 29.0},
        rows_affected=5,
    )
    data = rec.to_dict()
    assert data["stage"] == "missing"
    assert data["column"] == "age"
    assert data["rows_affected"] == 5

def test_clean_plan_diff():
    plan_a = CleanPlan(actions=(
        ActionRecord("missing", "age", "impute_median", {"median": 29.0}, 5),
        ActionRecord("strings", "name", "strip", {}, 10),
    ))
    plan_b = CleanPlan(actions=(
        ActionRecord("missing", "age", "impute_median", {"median": 30.0}, 5),
        ActionRecord("outliers", "salary", "clip", {"lower": 1000}, 2),
    ))
    diff = plan_a.diff(plan_b)
    assert len(diff["added"]) == 1
    assert diff["added"][0].column == "salary"
    assert len(diff["removed"]) == 1
    assert diff["removed"][0].column == "name"
    assert len(diff["modified"]) == 1
    assert diff["modified"][0]["column"] == "age"

def test_clean_plan_json_roundtrip():
    import json
    plan = CleanPlan(actions=(
        ActionRecord("missing", "age", "impute_median", {"median": 25.0}, 4),
    ))
    json_str = json.dumps(plan.to_dict())
    loaded = json.loads(json_str)
    assert len(loaded["actions"]) == 1
    assert loaded["actions"][0]["stage"] == "missing"

