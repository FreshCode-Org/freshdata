import polars as pl

from freshdata.polars_engine import (
    CleanConfig,
    ColumnRule,
    apply_plan,
    suggest_plan,
)


def test_apply_plan_preserves_lazyframe_without_materializing() -> None:
    raw_df = pl.DataFrame(
        {
            "Customer ID": ["001", "002", "003", "004", "005"],
            "Age": [25, 30, None, 35, 40],
            "Spend": [100.0, 120.0, 110.0, 115.0, 9999.0],
            "Segment": ["  Gold ", "Silver", None, "Gold", "Gold"],
            "Churn": [0, 1, None, 0, 1],
        }
    )
    lf = raw_df.lazy()

    config = CleanConfig(
        id_columns=("customer_id",),
        target_column="churn",
        default_impute="auto",
        default_outliers="clip",
        drop_duplicates=True,
    )

    plan = suggest_plan(lf, config=config)

    # Verify protected columns are never imputed or clipped
    actions_by_col = {(a.column, a.stage): a for a in plan.actions}
    assert actions_by_col[("churn", "missing")].operation == "preserve"
    assert actions_by_col[("age", "missing")].operation == "fill_null"
    assert actions_by_col[("age", "missing")].params["value"] == 32.5
    assert actions_by_col[("spend", "outliers")].operation == "clip_iqr"

    # Poison the LazyFrame after planning to prove apply_plan() never calls .collect()
    poisoned_lf = lf.with_columns(
        pl.col("Age").map_batches(
            lambda _: (_ for _ in ()).throw(RuntimeError("Eager collect triggered!")),
            return_dtype=pl.Int64,
        )
    )
    lazy_repaired = apply_plan(poisoned_lf, plan)
    assert isinstance(lazy_repaired, pl.LazyFrame)

    # Now execute on the clean LazyFrame and verify native Polars output
    cleaned_lf, report = apply_plan(lf, plan, return_report=True)
    assert isinstance(cleaned_lf, pl.LazyFrame)
    assert "PYTHON_UDF" not in cleaned_lf.explain()

    result = cleaned_lf.collect()
    assert result.columns == ["customer_id", "age", "spend", "segment", "churn"]
    assert result["age"].null_count() == 0
    assert result["churn"].null_count() == 1  # Protected target preserved
    assert result["spend"].max() < 9999.0     # Outlier clipped via precomputed IQR
    assert report.total_actions == len(plan.actions)


def test_plan_diff_detects_strategy_changes() -> None:
    lf = pl.LazyFrame({"score": [1.0, 2.0, None, 4.0, 100.0]})

    conservative_plan = suggest_plan(
        lf, CleanConfig(strategy="conservative", default_outliers="preserve")
    )
    aggressive_plan = suggest_plan(
        lf,
        CleanConfig(
            strategy="aggressive",
            rules={"score": ColumnRule(impute="median", outliers="clip")},
        ),
    )

    diffs = conservative_plan.diff(aggressive_plan)
    assert ("score", "outliers") in diffs
    before_act, after_act = diffs[("score", "outliers")]
    assert before_act is None
    assert after_act is not None and after_act.operation == "clip_iqr"
