from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import merge_options

pytest.importorskip("sklearn")

ISOLATE = {
    "drop_duplicates": False,
    "drop_empty_rows": False,
    "drop_empty_columns": False,
    "fix_dtypes": False,
    "verbose": False,
}


def _mixed_frame(n: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(42, 8, n).round(1)
    income = 1200 + age * 950 + rng.normal(0, 900, n)
    segment = np.where(income > np.nanmedian(income), "enterprise", "starter")
    active = income > np.nanmedian(income)
    return pd.DataFrame(
        {
            "row_id": np.arange(n),
            "age": age,
            "income": income.round(2),
            "segment": pd.Series(segment, dtype=object),
            "active": pd.Series(active, dtype="boolean"),
            "notes": [f"long free text account note number {i} with context" for i in range(n)],
            "target": rng.integers(0, 2, n).astype(float),
        }
    )


def _missforest_actions(report: fd.CleanReport, column: str):
    return [
        action
        for action in report
        if action.column == column and action.model_id.startswith("missforest")
    ]


def test_impute_method_alias_runs_missforest_on_mixed_frame() -> None:
    df = _mixed_frame()
    original = df.copy(deep=True)
    df.loc[5:14, "age"] = np.nan
    df.loc[20:29, "segment"] = None
    df.loc[35:44, "active"] = pd.NA
    mutated = df.copy(deep=True)

    out, report = fd.clean(df, impute_method="missforest", return_report=True, **ISOLATE)

    assert out["age"].isna().sum() == 0
    assert out["segment"].isna().sum() == 0
    assert out["active"].isna().sum() == 0
    assert set(out["segment"]).issubset({"enterprise", "starter"})
    assert out is not df
    assert not df.equals(original)
    pd.testing.assert_frame_equal(df, mutated)

    age_action = _missforest_actions(report, "age")[-1]
    assert "random-forest regressor" in age_action.description
    assert age_action.metadata["selected_model_type"] == "regressor"
    assert age_action.metadata["missing_count_before"] == 10
    assert age_action.metadata["imputed_count"] == 10
    assert "iterations" in age_action.metadata
    assert "converged" in age_action.metadata

    segment_action = _missforest_actions(report, "segment")[-1]
    assert "random-forest classifier" in segment_action.description
    assert segment_action.metadata["selected_model_type"] == "classifier"


def test_column_level_strategy_mixes_missforest_and_legacy_imputers() -> None:
    df = _mixed_frame()
    df.loc[0:8, "age"] = np.nan
    df.loc[10:18, "income"] = np.nan
    df.loc[20:28, "segment"] = None

    out, report = fd.clean(
        df,
        impute_strategy={"age": "missforest", "income": "median", "segment": "mode"},
        return_report=True,
        **ISOLATE,
    )

    assert out[["age", "income", "segment"]].isna().sum().sum() == 0
    assert _missforest_actions(report, "age")
    assert any(
        a.step == "impute" and a.column == "income" and "median" in a.description
        for a in report
    )
    assert any(
        a.step == "impute" and a.column == "segment" and "mode" in a.description
        for a in report
    )


def test_missforest_respects_target_id_text_and_high_missingness_gates() -> None:
    df = _mixed_frame()
    df.loc[0:9, "target"] = np.nan
    df.loc[10:19, "row_id"] = np.nan
    df.loc[20:29, "notes"] = None
    df["sparse"] = np.arange(len(df), dtype=float)
    df.loc[:70, "sparse"] = np.nan

    out, report = fd.clean(
        df,
        impute_method="missforest",
        target_column="target",
        id_columns=("row_id",),
        return_report=True,
        **ISOLATE,
    )

    assert out["target"].isna().sum() == 10
    assert out["row_id"].isna().sum() == 10
    assert out["notes"].isna().sum() == 10
    assert out["sparse"].isna().sum() == 71
    for column in ("target", "row_id", "notes", "sparse"):
        [action] = _missforest_actions(report, column)
        assert "fallback_reason" in action.metadata
        assert action.count == 0


def test_tiny_dataset_falls_back_with_audited_reason() -> None:
    df = pd.DataFrame({"age": [20.0, None, 40.0], "income": [100.0, 200.0, 300.0]})

    out, report = fd.clean(
        df,
        impute_method="missforest",
        return_report=True,
        missforest_min_rows_for_model=50,
        **ISOLATE,
    )

    assert out["age"].isna().sum() == 0
    [action] = _missforest_actions(report, "age")
    assert "fallback" in action.description
    assert (
        action.metadata["fallback_reason"]
        == "dataset has fewer rows than missforest_min_rows_for_model"
    )


def test_missforest_output_is_deterministic_with_fixed_random_state() -> None:
    df = _mixed_frame(seed=11)
    df.loc[::7, "age"] = np.nan
    df.loc[::9, "segment"] = None

    out1 = fd.clean(df, impute_method="missforest", missforest_random_state=123, **ISOLATE)
    out2 = fd.clean(df, impute_method="missforest", missforest_random_state=123, **ISOLATE)

    pd.testing.assert_frame_equal(out1, out2)


def test_optional_dependency_error_names_ml_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("blocked sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    df = _mixed_frame()
    df.loc[0:4, "age"] = np.nan

    with pytest.raises(ImportError, match=r'pip install "freshdata-cleaner\[ml\]"'):
        fd.clean(df, impute_method="missforest", **ISOLATE)


def test_impute_method_alias_conflict_is_rejected() -> None:
    with pytest.raises(TypeError, match="impute_method"):
        merge_options(None, impute="median", impute_method="missforest")
