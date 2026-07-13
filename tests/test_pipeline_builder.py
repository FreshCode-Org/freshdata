"""fd.pipeline() fluent builder and the fd.plan / fd.apply front doors."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import freshdata as fd
from freshdata.pipeline import Pipeline


@pytest.fixture
def messy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            " Name ": [" al ", "bo", None, "al "],
            "Age": ["34", "", "52", "34"],
            "email": ["a@x", "b@y", "c@z", "a@x"],
        }
    )


# -- immutability + construction --------------------------------------------


def test_each_step_returns_new_pipeline():
    base = fd.pipeline()
    one = base.normalize_columns()
    two = one.strip_whitespace()
    assert base.steps == ()
    assert len(one.steps) == 1
    assert len(two.steps) == 2
    assert one is not two


def test_duplicate_step_rejected():
    with pytest.raises(ValueError, match="twice"):
        fd.pipeline().impute(strategy="mean").impute(strategy="median")


def test_unknown_step_and_params_rejected():
    with pytest.raises(ValueError, match="unknown pipeline step"):
        Pipeline(steps=(("frobnicate", {}),))
    with pytest.raises(ValueError, match="unknown parameter"):
        Pipeline(steps=(("impute", {"strategy": "mean", "turbo": True}),))


def test_empty_pipeline_cannot_run(messy):
    with pytest.raises(ValueError, match="empty pipeline"):
        fd.pipeline().run(messy)


def test_pipelines_compare_by_value():
    a = fd.pipeline().normalize_columns().impute(strategy="mean")
    b = fd.pipeline().normalize_columns().impute(strategy="mean")
    assert a == b


# -- compile + run ------------------------------------------------------------


def test_only_chained_steps_are_enabled():
    cfg = fd.pipeline().normalize_columns().compile()
    assert cfg.column_names is True
    assert cfg.strip_whitespace is False
    assert cfg.normalize_sentinels is False
    assert cfg.fix_dtypes is False
    assert cfg.drop_duplicates is False
    assert cfg.strategy == "conservative"


def test_run_equals_clean_with_compiled_config(messy):
    pipe = (
        fd.pipeline()
        .normalize_columns()
        .strip_whitespace()
        .normalize_missing()
        .validate_types()
        .deduplicate(subset=["email"])
    )
    via_pipe = pipe.run(messy)
    via_clean = fd.clean(messy, config=pipe.compile())
    pd.testing.assert_frame_equal(via_pipe, via_clean)


def test_run_does_not_mutate_input(messy):
    before = messy.copy(deep=True)
    fd.pipeline().normalize_columns().strip_whitespace().run(messy)
    pd.testing.assert_frame_equal(messy, before)


def test_impute_with_columns_maps_to_per_column_strategy():
    cfg = fd.pipeline().impute(strategy="median", columns=["age", "height"]).compile()
    assert cfg.impute_strategy == {"age": "median", "height": "median"}
    assert cfg.impute is None
    cfg = fd.pipeline().impute(strategy="mean").compile()
    assert cfg.impute == "mean"


def test_deduplicate_subset_and_keep():
    cfg = fd.pipeline().deduplicate(subset=["email"], keep="last").compile()
    assert cfg.duplicate_subset == ("email",)
    assert cfg.duplicate_keep == "last"


def test_outliers_step(messy):
    cfg = fd.pipeline().outliers(method="zscore", action="flag").compile()
    assert cfg.outliers == "flag"
    assert cfg.outlier_method == "zscore"


def test_run_with_report_and_engine(messy):
    cleaned, report = (
        fd.pipeline().normalize_columns().strip_whitespace().run(messy, return_report=True)
    )
    assert report.rows_before == len(messy)
    assert isinstance(cleaned, pd.DataFrame)


def test_run_on_polars_engine_native():
    pytest.importorskip("polars")
    df = pd.DataFrame({"A B": ["x ", " y", "x ", "z"], "v": [1.0, 2.0, 1.0, 3.0]})
    pipe = fd.pipeline().normalize_columns().strip_whitespace().deduplicate()
    cleaned, report = pipe.run(
        df, engine="polars", fallback_policy="error", return_report=True
    )
    assert report.backend == "polars"
    ref = pipe.run(df)
    # streaming full-row dedup does not guarantee row order (disclosed on the
    # report) — compare content, not order
    key = list(ref.columns)
    pd.testing.assert_frame_equal(
        ref.sort_values(key).reset_index(drop=True),
        cleaned.sort_values(key).reset_index(drop=True),
    )


# -- serialization ------------------------------------------------------------


def test_json_roundtrip():
    pipe = (
        fd.pipeline()
        .normalize_columns()
        .normalize_missing(extra_sentinels=("MISSING",))
        .impute(strategy="median", columns=["age"])
        .deduplicate(subset=["email"], keep="last")
    )
    restored = Pipeline.from_json(pipe.to_json())
    assert restored == pipe
    assert restored.compile() == pipe.compile()


def test_json_schema_version_enforced():
    payload = json.dumps({"schema_version": "v999", "steps": []})
    with pytest.raises(ValueError, match="schema_version"):
        Pipeline.from_json(payload)


def test_describe_lists_steps_in_order():
    text = fd.pipeline().normalize_columns().impute(strategy="mean").describe()
    lines = text.splitlines()
    assert "normalize_columns" in lines[1]
    assert "impute" in lines[2]


# -- fd.plan / fd.apply --------------------------------------------------------


def test_plan_reports_native_backend(messy):
    p = fd.plan(messy, strategy="conservative", engine="polars", fix_dtypes=False)
    assert p.backend == "polars"
    assert p.fallback_reason is None
    assert "runs natively" in p.summary()


def test_plan_reports_fallback_before_execution(messy):
    p = fd.plan(messy, strategy="balanced", engine="duckdb")
    assert p.backend == "duckdb"
    assert p.fallback_reason is not None
    assert "decision engine" in p.fallback_reason
    assert "fall back" in p.summary()
    payload = p.to_dict()
    assert payload["backend"] == "duckdb"
    assert payload["fallback_reason"]


def test_plan_never_mutates(messy):
    before = messy.copy(deep=True)
    fd.plan(messy, strategy="balanced")
    pd.testing.assert_frame_equal(messy, before)


def test_plan_defaults_to_pandas_backend(messy):
    p = fd.plan(messy)
    assert p.backend == "pandas"
    assert p.fallback_reason is None


def test_apply_is_apply_plan_alias():
    assert fd.apply is fd.apply_plan
