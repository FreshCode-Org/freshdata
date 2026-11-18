import polars as pl
import pandas as pd
import pytest
from freshdata._core.backends import resolve_backend
from freshdata._core.models import CleanConfig, ActionRecord, CleanPlan

def test_lazyframe_preserved_without_collect(monkeypatch):
    called = []
    original_collect = pl.LazyFrame.collect
    def guarded_collect(self, *args, **kwargs):
        called.append(True)
        return original_collect(self, *args, **kwargs)
    monkeypatch.setattr(pl.LazyFrame, "collect", guarded_collect)

    lazy_df = pl.LazyFrame({"a": [" hello ", "world"], "b": [1, 2]})
    backend = resolve_backend(lazy_df)
    plan = CleanPlan(actions=(
        ActionRecord("strings", "a", "strip_strings", {}),
    ))
    res = backend.apply_plan(lazy_df, plan, CleanConfig())
    assert isinstance(res, pl.LazyFrame)
    assert len(called) == 0, "LazyFrame must not be collected eagerly during apply_plan"

def test_to_pandas_never_called(monkeypatch):
    called = []
    def guarded_to_pandas(self, *args, **kwargs):
        called.append(True)
        raise RuntimeError("Forbidden to_pandas call!")
    monkeypatch.setattr(pl.DataFrame, "to_pandas", guarded_to_pandas)

    pl_df = pl.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    backend = resolve_backend(pl_df)
    report = backend.profile(pl_df)
    assert report.row_count == 3
    assert len(called) == 0

def test_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported frame type"):
        resolve_backend({"a": [1, 2, 3]})

def test_import_and_clean_without_polars():
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys; sys.modules['polars'] = None\n"
        "import pandas as pd, freshdata\n"
        "out = freshdata.clean(pd.DataFrame({'A b': [1.0, None, 3.0]}))\n"
        "assert out['a_b'].tolist() == [1.0, 2.0, 3.0]\n"
    )
    src = Path(__file__).resolve().parents[1] / "src"
    subprocess.run([sys.executable, "-c", code], check=True, env={"PYTHONPATH": str(src)})

def test_polars_profile_counts_match_pandas():
    data = {"a": [1, 2, None, 2], "b": ["x", None, None, "y"]}
    pd_rep = resolve_backend(pd.DataFrame(data)).profile(pd.DataFrame(data))
    pl_rep = resolve_backend(pl.DataFrame(data)).profile(pl.DataFrame(data))
    for col in data:
        p, q = pd_rep.columns[col], pl_rep.columns[col]
        assert (q.null_count, q.unique_count, q.null_ratio) == (p.null_count, p.unique_count, p.null_ratio)

def test_polars_plan_matches_pandas_plan():
    import freshdata as fd

    data = {"Customer ID": ["1", "2", "3", "4", "5"], "Age": [25.0, 30.0, None, 35.0, 40.0],
            "Spend": [100.0, 120.0, 110.0, 115.0, 9999.0], "Churn": [0.0, 1.0, None, 0.0, 1.0]}
    cfg = CleanConfig(id_columns=("customer_id",), target_column="churn", default_outliers="clip",
                      strip_strings=False, fix_dtypes=False)
    def key(plan):
        return {(a.stage, a.column, a.action): a.parameters for a in plan.actions
                if a.stage in ("column_names", "outliers")}
    assert key(fd.suggest_plan(pl.LazyFrame(data), cfg)) == key(fd.suggest_plan(pd.DataFrame(data), cfg))

def test_lazy_plan_applies_without_materializing():
    import freshdata as fd

    lf = pl.DataFrame({
        "Customer ID": ["001", "002", "003", "004", "005"],
        "Age": [25, 30, None, 35, 40],
        "Spend": [100.0, 120.0, 110.0, 115.0, 9999.0],
        "Segment": ["  Gold ", "Silver", None, "Gold", "Gold"],
        "Churn": [0, 1, None, 0, 1],
    }).lazy()
    config = CleanConfig(id_columns=("customer_id",), target_column="churn",
                         default_impute="auto", default_outliers="clip", drop_duplicates=True)
    plan = fd.suggest_plan(lf, config)

    actions = {(a.column, a.stage): a for a in plan.actions}
    assert ("churn", "missing") not in actions
    assert actions[("age", "missing")].action == "impute_median"
    assert actions[("spend", "outliers")].action == "clip_iqr"

    poisoned_lf = lf.with_columns(
        pl.col("Age").map_batches(
            lambda _: (_ for _ in ()).throw(RuntimeError("Eager collect triggered!")),
            return_dtype=pl.Int64,
        )
    )
    assert isinstance(fd.apply_plan(poisoned_lf, plan, config), pl.LazyFrame)

    cleaned_lf, report = fd.apply_plan(lf, plan, config, return_report=True)
    assert isinstance(cleaned_lf, pl.LazyFrame)
    assert "PYTHON_UDF" not in cleaned_lf.explain()
    result = cleaned_lf.collect()
    assert result.columns == ["customer_id", "age", "spend", "segment", "churn"]
    assert result["age"].null_count() == 0
    assert result["churn"].null_count() == 1
    assert result["spend"].max() < 9999.0
    assert report.total_actions == len(plan.actions)

def test_lazy_plan_diff_detects_rule_changes():
    import freshdata as fd

    lf = pl.LazyFrame({"score": [1.0, 2.0, None, 4.0, 100.0]})
    conservative = fd.suggest_plan(lf, CleanConfig(strategy="conservative", default_outliers="preserve"))
    aggressive = fd.suggest_plan(lf, CleanConfig(
        strategy="aggressive", rules={"score": fd.ColumnRule(impute="median", outliers="clip")}))
    diff = conservative.diff(aggressive)
    assert [(a.column, a.action) for a in diff["added"]] == [("score", "clip_iqr")]
    assert diff["removed"] == [] and diff["modified"] == []


@pytest.mark.parametrize("rule, default", [
    ({"outliers": "clip"}, {"default_impute": "median"}),
    ({"impute": "mean"}, {"default_outliers": "clip"}),
])
def test_rules_fall_back_to_defaults_per_field_on_both_backends(rule, default):
    import freshdata as fd

    data = {"v": [1.0, None, 2.0, 3.0, 2.0, 100.0]}
    cfg = CleanConfig(rules={"v": fd.ColumnRule(**rule)}, **default)
    pd_actions = {a.action for a in fd.suggest_plan(pd.DataFrame(data), cfg).actions}
    pl_actions = {a.action for a in fd.suggest_plan(pl.DataFrame(data), cfg).actions}
    assert pd_actions == pl_actions and len(pl_actions) == 2


def _rows(frame):
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if isinstance(frame, pl.DataFrame):
        frame = frame.to_pandas()
    return sorted(map(tuple, frame.astype(object).where(frame.notna(), None).values.tolist()), key=repr)


def test_every_pandas_action_runs_identically_on_polars():
    import freshdata as fd

    data = {"price": ["$1", "$2", "$3", None, "$5", "$6"], "tag": ["a", "a", None, "b", "b", "a"],
            "v": [1.0, None, 2.0, 3.0, 2.0, 2.5], "w": [1.0, 2.0, 2.0, 3.0, 2.0, 100.0]}
    cfg = CleanConfig(rules={"tag": fd.ColumnRule(impute="mode"), "v": fd.ColumnRule(impute="drop"),
                             "w": fd.ColumnRule(outliers="drop")}, strip_strings=False)
    plan = fd.suggest_plan(pd.DataFrame(data), cfg)
    assert {a.action for a in plan.actions} >= {"parse_currency", "impute_mode", "drop_nulls", "drop_outliers"}
    expected = _rows(fd.apply_plan(pd.DataFrame(data), plan, cfg))
    assert _rows(fd.apply_plan(pl.DataFrame(data), plan, cfg)) == expected
    lazy = fd.apply_plan(pl.LazyFrame(data), plan, cfg)
    assert "PYTHON_UDF" not in lazy.explain()
    assert _rows(lazy) == expected


def test_polars_planner_honors_mode_drop_and_outlier_drop_rules():
    import freshdata as fd

    data = {"tag": ["a", "a", None, "b"], "v": [1.0, None, 2.0, 3.0], "w": [1.0, 2.0, 2.0, 3.0]}
    cfg = CleanConfig(rules={"tag": fd.ColumnRule(impute="mode"), "v": fd.ColumnRule(impute="drop"),
                             "w": fd.ColumnRule(outliers="drop")})
    actions = {(a.column, a.action) for a in fd.suggest_plan(pl.LazyFrame(data), cfg).actions}
    assert {("tag", "impute_mode"), ("v", "drop_nulls"), ("w", "drop_outliers")} <= actions


def test_polars_plan_without_values_imputes_on_pandas():
    import freshdata as fd

    data = {"v": [1.0, None, 3.0], "t": ["x", None, "x"]}
    cfg = CleanConfig(rules={"t": fd.ColumnRule(impute="mode")})
    plan = fd.suggest_plan(pl.DataFrame(data), cfg)
    out = fd.apply_plan(pd.DataFrame(data), plan, cfg)
    assert out["v"].tolist() == [1.0, 2.0, 3.0] and out["t"].tolist() == ["x", "x", "x"]


@pytest.mark.parametrize("frame", [pd.DataFrame({"v": [1.0]}), pl.DataFrame({"v": [1.0]})])
def test_strict_backend_rejects_actions_it_cannot_run(frame):
    import freshdata as fd

    for bad in (ActionRecord("missing", "v", "impute_magic"), ActionRecord("outliers", "v", "clip_iqr", {})):
        with pytest.raises(ValueError, match="cannot run"):
            fd.apply_plan(frame, CleanPlan((bad,)))
        with pytest.warns(UserWarning, match="Skipped"):
            out = fd.apply_plan(frame, CleanPlan((bad,)), CleanConfig(strict_backend=False))
        assert _rows(out) == _rows(frame)
