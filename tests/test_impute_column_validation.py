"""Regression tests for impute column validation (#310) and the MissForest
predictor-less fallback (#324)."""

from __future__ import annotations

import builtins
import re

import numpy as np
import pandas as pd
import pytest

import freshdata as fd


def _age_frame() -> pd.DataFrame:
    return pd.DataFrame({"Age": [30.0, np.nan, 40.0, 50.0], "id": [1, 2, 3, 4]})


# -- #310: unknown impute_strategy / Pipeline.impute(columns=) keys -----------


def test_pipeline_impute_pre_rename_name_raises_with_hint() -> None:
    pipe = fd.pipeline().normalize_columns().impute(strategy="median", columns=["Age"])

    with pytest.raises(ValueError) as excinfo:
        pipe.run(_age_frame(), return_report=True)

    message = str(excinfo.value)
    assert "impute_strategy column(s) not found: ['Age']" in message
    assert "Available columns: ['age', 'id']" in message
    assert "*after* renaming" in message
    assert "'Age' -> 'age'" in message


def test_clean_impute_strategy_typo_raises() -> None:
    with pytest.raises(ValueError) as excinfo:
        fd.clean(_age_frame(), impute_strategy={"agee": "median"}, verbose=False)

    message = str(excinfo.value)
    assert "impute_strategy column(s) not found: ['agee']" in message
    assert "Available columns: ['age', 'id']" in message
    assert "normalized name" not in message  # 'agee' was never a real column


def test_clean_default_rename_rejects_original_name() -> None:
    with pytest.raises(ValueError, match=re.escape("'Age' -> 'age'")):
        fd.clean(_age_frame(), impute_strategy={"Age": "median"}, verbose=False)


def test_unknown_key_without_renaming_has_no_rename_note() -> None:
    with pytest.raises(ValueError) as excinfo:
        fd.clean(
            _age_frame(),
            impute_strategy={"age": "median"},
            column_names=False,
            verbose=False,
        )

    message = str(excinfo.value)
    assert "impute_strategy column(s) not found: ['age']" in message
    assert "renaming" not in message


def test_pipeline_impute_normalized_name_imputes() -> None:
    pipe = fd.pipeline().normalize_columns().impute(strategy="median", columns=["age"])

    out, rep = pipe.run(_age_frame(), return_report=True)

    assert out["age"].tolist() == [30.0, 40.0, 40.0, 50.0]
    assert [a.column for a in rep.actions if a.step == "impute"] == ["age"]


def test_pipeline_impute_original_name_without_renaming_imputes() -> None:
    out = fd.pipeline().impute(strategy="median", columns=["Age"]).run(_age_frame())

    assert out["Age"].tolist() == [30.0, 40.0, 40.0, 50.0]


def test_key_for_column_dropped_by_later_step_is_still_accepted() -> None:
    df = _age_frame()
    df["Empty"] = np.nan

    out = fd.clean(
        df,
        impute_strategy={"age": "median", "empty": "median"},
        drop_empty_columns=True,
        verbose=False,
    )

    assert "empty" not in out.columns
    assert out["age"].isna().sum() == 0


def test_non_string_labels_match_by_string_key() -> None:
    df = pd.DataFrame({0: [1.0, np.nan, 3.0], 1: [4.0, 5.0, 6.0]})

    out = fd.clean(df, impute_strategy={"0": "median"}, verbose=False)

    assert out[0].isna().sum() == 0


def test_error_is_raised_before_input_is_touched() -> None:
    df = _age_frame()
    original = df.copy(deep=True)

    with pytest.raises(ValueError, match="impute_strategy"):
        fd.clean(df, impute_strategy={"agee": "median"}, verbose=False)

    pd.testing.assert_frame_equal(df, original)


# -- #324: MissForest with no predictor columns -------------------------------


def _single_column_frame() -> pd.DataFrame:
    df = pd.DataFrame({"x": [float(i) for i in range(60)]})
    df.loc[[3, 7], "x"] = np.nan
    return df


def _block_sklearn(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("blocked sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def _assert_single_fallback(out: pd.DataFrame, rep: fd.CleanReport) -> None:
    impute_actions = [a for a in rep.actions if a.step == "impute"]
    assert [(a.model_id, a.count) for a in impute_actions] == [("missforest_fallback", 2)]
    assert rep.columns_imputed == ["x"]
    action = impute_actions[0]
    assert "random-forest" not in action.description
    assert "no predictor columns" in action.rationale
    assert action.metadata["fallback_reason"] == "no predictor columns available for MissForest"
    assert action.metadata["selected_model_type"] is None
    assert action.metadata["iterations"] == 0
    assert out["x"].isna().sum() == 0
    assert out.loc[3, "x"] == out.loc[7, "x"] == pd.Series(range(60)).drop([3, 7]).median()


def test_missforest_single_column_records_fallback_once() -> None:
    out, rep = fd.clean(
        _single_column_frame(),
        impute="missforest",
        drop_empty_rows=False,
        return_report=True,
        verbose=False,
    )

    _assert_single_fallback(out, rep)


def test_missforest_single_column_does_not_require_sklearn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_sklearn(monkeypatch)

    out, rep = fd.clean(
        _single_column_frame(),
        impute="missforest",
        drop_empty_rows=False,
        return_report=True,
        verbose=False,
    )

    _assert_single_fallback(out, rep)


def test_missforest_single_column_via_impute_strategy_without_sklearn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_sklearn(monkeypatch)

    out, rep = fd.clean(
        _single_column_frame(),
        impute_strategy={"x": "missforest"},
        drop_empty_rows=False,
        return_report=True,
        verbose=False,
    )

    _assert_single_fallback(out, rep)


def test_missforest_with_predictors_still_fits_model_once() -> None:
    pytest.importorskip("sklearn")
    df = pd.DataFrame({"x": [float(i) for i in range(60)], "y": [2.0 * i + 1 for i in range(60)]})
    df.loc[[3, 7], "x"] = np.nan

    out, rep = fd.clean(
        df,
        impute="missforest",
        drop_empty_rows=False,
        return_report=True,
        verbose=False,
    )

    impute_actions = [a for a in rep.actions if a.step == "impute" and a.column == "x"]
    assert [(a.model_id, a.count) for a in impute_actions] == [("missforest_regressor", 2)]
    assert rep.columns_imputed == ["x"]
    assert out["x"].isna().sum() == 0
