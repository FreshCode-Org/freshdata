"""Tests for the pure-Python plan generator (native vs fallback split)."""

from __future__ import annotations

from freshdata.config import CleanConfig
from freshdata.execution import PlanGenerator


def _plan(config, columns=("a", "b")):
    return PlanGenerator(config).plan(list(columns))


def test_native_config_has_no_fallback():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False)
    assert _plan(cfg).fallback_reason is None
    assert _plan(cfg).needs_fallback is False


def test_decision_engine_forces_fallback():
    assert _plan(CleanConfig(strategy="balanced")).needs_fallback


def test_fix_dtypes_forces_fallback():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=True)
    assert _plan(cfg).needs_fallback


def test_impute_runs_natively():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False, impute="median")
    plan = _plan(cfg)
    assert not plan.needs_fallback
    assert "impute" in plan.stages


def test_iqr_outliers_run_natively():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False,
                      outliers="clip", outlier_method="iqr")
    plan = _plan(cfg)
    assert not plan.needs_fallback
    assert "outliers" in plan.stages


def test_isolation_forest_outliers_force_fallback():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False,
                      outliers="clip", outlier_method="isolation_forest")
    assert _plan(cfg).needs_fallback


def test_subset_dedup_forces_fallback():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False,
                      duplicate_subset=("a",))
    assert _plan(cfg).needs_fallback


def test_rename_map_uses_snake_case():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False)
    plan = _plan(cfg, columns=["Customer ID", "amount"])
    assert plan.rename_map == {"Customer ID": "customer_id"}


def test_native_stage_order():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False)
    stages = _plan(cfg).stages
    # representation repair + structural reduction, in pipeline order
    assert stages == [
        "column_names",
        "clean_strings",
        "drop_empty_columns",
        "drop_empty_rows",
        "drop_duplicates",
    ]


def test_disabled_stages_excluded():
    cfg = CleanConfig(
        strategy="conservative", fix_dtypes=False,
        column_names=False, drop_duplicates=False,
    )
    stages = _plan(cfg).stages
    assert "column_names" not in stages
    assert "drop_duplicates" not in stages


def test_logical_plan_nodes_carry_contract():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False,
                      impute="median", outliers="clip", outlier_method="iqr")
    nodes = PlanGenerator(cfg).logical_plan(["Customer ID", "amount"])
    by_name = {n.name: n for n in nodes}

    # rename node reflects input -> output columns and parameters
    rename = by_name["column_names"]
    assert rename.input_columns == ("Customer ID", "amount")
    assert rename.output_columns == ("customer_id", "amount")
    assert rename.parameters["rename_map"] == {"Customer ID": "customer_id"}

    for node in nodes:
        assert node.audit_schema["step"] == node.name or node.name in ("column_names",)
        assert node.fallback_policy in ("native", "pandas")

    assert by_name["impute"].fallback_policy == "native"
    assert by_name["outliers"].parameters["method"] == "iqr"


def test_logical_plan_marks_global_fallback():
    cfg = CleanConfig(strategy="balanced")  # decision engine -> whole plan delegated
    nodes = PlanGenerator(cfg).logical_plan(["a", "b"])
    assert all(n.fallback_policy == "pandas" for n in nodes)


def test_logical_plan_splits_string_steps():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False,
                      strip_whitespace=True, normalize_sentinels=True)
    names = [n.name for n in PlanGenerator(cfg).logical_plan(["a"])]
    assert "strip_whitespace" in names
    assert "normalize_sentinels" in names


def test_logical_plan_zscore_outliers_native():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False,
                      outliers="flag", outlier_method="zscore")
    nodes = {n.name: n for n in PlanGenerator(cfg).logical_plan(["a"])}
    assert nodes["outliers"].fallback_policy == "native"
