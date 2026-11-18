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



@pytest.mark.parametrize("make", [
    lambda: ColumnRule(impute="medain"),
    lambda: ColumnRule(outliers="cap"),
    lambda: ColumnRule(role="key"),
    lambda: CleanConfig(default_impute="knn"),
    lambda: CleanConfig(strategy="fast"),
    lambda: CleanConfig(id_columns="customer_id"),
    lambda: CleanConfig(iqr_factor=-1),
])
def test_invalid_options_are_rejected(make):
    with pytest.raises(ValueError):
        make()


def test_rules_must_be_column_rules_and_id_columns_become_a_tuple():
    with pytest.raises(TypeError):
        CleanConfig(rules={"a": {"impute": "mean"}})
    assert CleanConfig(id_columns=["a", "b"]).id_columns == ("a", "b")


def test_options_without_effect_warn():
    with pytest.warns(UserWarning, match="no effect"):
        CleanConfig(strategy="aggressive")
    with pytest.warns(UserWarning, match="no effect"):
        ColumnRule(role="numeric")
    with pytest.warns(UserWarning, match="like 'preserve'"):
        CleanConfig(default_outliers="auto")


def test_role_id_and_target_protect_the_column():
    import pandas as pd

    import freshdata as fd

    df = pd.DataFrame({"k": [" a ", None, " a "], "v": [1.0, None, 3.0]})
    cfg = CleanConfig(rules={"k": ColumnRule(role="id"), "v": ColumnRule(role="target")}, default_impute="mode")
    out = fd.clean(df, cfg)
    assert out["k"].tolist()[0] == " a " and out["v"].isna().sum() == 1
