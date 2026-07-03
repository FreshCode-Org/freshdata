"""Phase-2 per-column impute-confidence threshold overrides (IMPUTE_IF)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import freshdata as fd

CONTEXT = "Missing Age should be estimated only if confidence >95%."


def _frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "age": [25.0 + (i % 30) for i in range(n)],
            "income": rng.normal(50_000, 5_000, n).round(2),
            "score": rng.normal(100, 10, n).round(2),
        }
    )
    df.loc[[3, 17, 41], "age"] = None
    df.loc[[5, 23], "income"] = None
    return df


def test_age_threshold_holds_back_imputation_and_only_age():
    df = _frame()
    out, report = fd.clean(df, context=CONTEXT, return_report=True, verbose=False)
    # age: every engine fill confidence (<= 0.9) is below 0.95 -> preserved.
    assert out["age"].isna().sum() == 3
    # other columns keep the normal behaviour (low band -> filled).
    assert out["income"].isna().sum() == 0
    gated = [
        a for a in report
        if a.column == "age" and a.metadata.get("impute_min_confidence") == 0.95
    ]
    assert gated and gated[0].status == "suggested" and gated[0].count == 0
    assert any("held back by the context confidence" in r
               for r in report.recommendations)


def test_below_threshold_does_not_mutate_frame():
    df = _frame()
    out = fd.clean(df, context=CONTEXT, verbose=False)
    pd.testing.assert_series_equal(
        out["age"].reset_index(drop=True), df["age"].reset_index(drop=True),
        check_names=False,
    )


def test_above_threshold_allows_existing_behaviour():
    df = _frame()
    # An explicit 0.85 threshold sits below the 0.9 low-band confidence.
    out, report = fd.clean(
        df,
        semantic_context={"columns": {"age": {"impute_min_confidence": 0.85}}},
        return_report=True,
        verbose=False,
    )
    assert out["age"].isna().sum() == 0
    filled = [a for a in report if a.column == "age" and a.step == "missing"]
    assert filled and filled[0].status == "automatic"


def test_global_thresholds_unchanged_for_other_columns():
    df = _frame()
    baseline = fd.clean(df, verbose=False)
    with_context = fd.clean(df, context=CONTEXT, verbose=False)
    pd.testing.assert_frame_equal(
        baseline.drop(columns=["age"]), with_context.drop(columns=["age"])
    )


def test_policy_thresholds_helper_scoping():
    policy = fd.compile_context(CONTEXT, columns=["age", "income"])
    cfg = fd.CleanConfig()
    age = policy.thresholds("age", "impute", cfg)
    assert age.auto == 0.95 and age.from_policy
    income = policy.thresholds("income", "impute", cfg)
    assert income.auto == cfg.semantic_auto_threshold and not income.from_policy
    other_kind = policy.thresholds("age", "outlier", cfg)
    assert not other_kind.from_policy
