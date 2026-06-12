"""API surface and safety guarantees of the auto strategy."""

import json

import numpy as np
import pandas as pd
import pytest

import freshdata as fd


def sample_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.normal(50, 5, n),
        "y2": rng.normal(0, 1, n),
        "c": [f"cat_{i % 3}" for i in range(n)],
    })
    df.loc[:2, "x"] = None
    return df


def test_return_report_tuple():
    out, report = fd.clean(sample_df(), return_report=True, verbose=False)
    assert isinstance(out, pd.DataFrame)
    assert isinstance(report, fd.CleanReport)
    assert report.missing_before == 3
    assert report.missing_after == 0


def test_report_alias_still_works():
    out, report = fd.clean(sample_df(), report=True, verbose=False)
    assert isinstance(report, fd.CleanReport)


def test_preserve_original_default_never_mutates():
    df = sample_df()
    snapshot = df.copy(deep=True)
    fd.clean(df, verbose=False)
    pd.testing.assert_frame_equal(df, snapshot)


def test_preserve_original_false_returns_valid_frame():
    df = sample_df()
    out = fd.clean(df, preserve_original=False, verbose=False)
    assert isinstance(out, pd.DataFrame)
    assert out["x"].isna().sum() == 0


def test_verbose_prints_one_line_summary(capsys):
    fd.clean(sample_df(), verbose=True)
    captured = capsys.readouterr().out
    assert captured.startswith("freshdata:")
    fd.clean(sample_df(), verbose=False)
    assert capsys.readouterr().out == ""


def test_threshold_parameters_are_respected():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
    df.loc[:39, "b"] = None  # 40% missing
    # Aggressive + default thresholds: 40% is "high" -> dropped.
    dropped = fd.clean(df, strategy="aggressive", verbose=False)
    assert "b" not in dropped.columns
    # Raising the medium threshold makes 40% "medium" -> imputed instead.
    kept = fd.clean(df, strategy="aggressive", missing_threshold_medium=0.5, verbose=False)
    assert "b" in kept.columns
    assert kept["b"].isna().sum() == 0
    # Balanced preserves high-missing columns.
    balanced = fd.clean(df, verbose=False)
    assert "b" in balanced.columns


def test_invalid_engine_options_fail_fast():
    with pytest.raises(ValueError, match="strategy"):
        fd.CleanConfig(strategy="bogus")
    with pytest.raises(ValueError, match="outlier_action"):
        fd.CleanConfig(outlier_action="zap")
    with pytest.raises(ValueError, match="duplicate_keep"):
        fd.CleanConfig(duplicate_keep="random")
    with pytest.raises(ValueError, match="ordered"):
        fd.CleanConfig(missing_threshold_low=0.5, missing_threshold_medium=0.1)
    with pytest.raises(ValueError, match="missing_threshold_low"):
        fd.CleanConfig(missing_threshold_low=0.0)


def test_every_action_carries_confidence_and_risk():
    out, report = fd.clean(sample_df(), return_report=True, verbose=False)
    for action in report:
        assert 0.0 <= action.confidence <= 1.0
        assert action.risk in ("low", "medium", "high")


def test_report_to_dict_includes_engine_fields():
    _, report = fd.clean(sample_df(), return_report=True, verbose=False)
    payload = report.to_dict()
    for key in ("missing_before", "missing_after", "duplicates_removed",
                "outliers_handled", "columns_dropped", "columns_imputed",
                "columns_preserved", "warnings", "recommendations"):
        assert key in payload
    assert json.dumps(payload)


def test_no_nan_left_unexplained():
    # Whatever the engine decides, every column that still has NaNs after
    # cleaning must have a "missing" action explaining why.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "num": rng.normal(0, 1, 60),
        "order_id": range(60),
        "note": [f"free text entry number {i} with several words" for i in range(60)],
    })
    df.loc[:5, "num"] = None
    df.loc[:5, "order_id"] = None
    df.loc[:5, "note"] = None
    out, report = fd.clean(df, return_report=True, verbose=False)
    explained = {a.column for a in report if a.step == "missing"}
    for col in out.columns:
        if out[col].isna().any():
            assert col in explained


def test_conservative_strategy_disables_engine():
    df = sample_df()
    out, report = fd.clean(df, strategy="conservative", return_report=True,
                           verbose=False)
    assert out["x"].isna().sum() == 3  # untouched
    assert not [a for a in report if a.step in ("missing", "outliers")]


def test_auto_strategy_deprecated_alias_for_aggressive():
    with pytest.warns(DeprecationWarning, match='strategy="auto" is deprecated'):
        cfg = fd.CleanConfig(strategy="auto")
    assert cfg.engine_mode == "aggressive"
