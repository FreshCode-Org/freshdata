"""FreshCore fallbacks and report parity for what the native v1 kernels miss.

- #322: ``impute="missforest"`` / per-column ``impute_strategy`` imputed nothing.
- #334: ``±inf`` made the native outlier fences unbounded, so nothing was flagged.
- #335: missing values in nullable boolean columns were never imputed.
- #323: duplicate detection was never reported, so ``duplicate_ratio_action``
  could not escalate.

The Rust extension is optional, so a fake native module stands in for it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.execution.backends._freshcore import FreshCoreEngine
from freshdata.steps.duplicates import DuplicateRatioError

KW = {"strategy": "conservative", "fix_dtypes": False, "verbose": False}


class _ForbiddenNative:
    @staticmethod
    def execute_plan(payload):  # pragma: no cover - must not be called
        raise AssertionError("unexpected native call")


class _EchoNative:
    """Returns the input columns unchanged, plus whatever extra keys are set."""

    calls = 0
    extra: dict = {}

    @classmethod
    def execute_plan(cls, payload):
        cls.calls += 1
        return {"columns": payload["columns"], "actions": [], **cls.extra}


def _use_native(monkeypatch, module) -> None:
    monkeypatch.setattr(FreshCoreEngine, "_load_native", staticmethod(lambda: module))


def _echo(monkeypatch, **extra) -> type[_EchoNative]:
    native = type("EchoNative", (_EchoNative,), {"calls": 0, "extra": extra})
    _use_native(monkeypatch, native)
    return native


def _inf_frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"x": values, "k": np.arange(len(values), dtype=float)})


FALLBACK_CASES = [
    pytest.param(
        pd.DataFrame({"x": [1.0, None, 3.0] * 7, "y": np.arange(21.0)}),
        {"impute": "missforest"},
        "missforest imputation uses scikit-learn",
        None,
        id="322-missforest",
    ),
    pytest.param(
        pd.DataFrame({"x": [1.0, None, 3.0] * 20, "y": np.arange(60.0)}),
        {"impute_strategy": {"x": "mean"}},
        "impute_strategy per-column overrides",
        None,
        id="322-impute-strategy",
    ),
    pytest.param(
        _inf_frame([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100, np.inf]),
        {"outliers": "flag", "outlier_method": "zscore", "outlier_factor": 2.0},
        "non-finite values in numeric column(s) 'x'",
        2,
        id="334-zscore",
    ),
    pytest.param(
        _inf_frame([1.0, 2, 3, 4, np.inf, np.inf, np.inf, np.inf]),
        {"outliers": "flag", "outlier_method": "iqr", "outlier_factor": 1.5},
        "non-finite values in numeric column(s) 'x'",
        4,
        id="334-iqr",
    ),
    pytest.param(
        pd.DataFrame(
            {"b": pd.array([True, True, None, False], dtype="boolean"), "k": [1.0, 2, 3, 4]}
        ),
        {"impute": "mode"},
        "missing values in boolean column(s) 'b'",
        None,
        id="335-mode",
    ),
    pytest.param(
        pd.DataFrame(
            {"b": pd.array([True, True, None, False], dtype="boolean"), "k": [1.0, 2, 3, 4]}
        ),
        {"impute": "auto"},
        "missing values in boolean column(s) 'b'",
        None,
        id="335-auto",
    ),
]


@pytest.mark.parametrize(("df", "options", "reason", "outliers"), FALLBACK_CASES)
def test_falls_back_to_pandas_with_identical_output(monkeypatch, df, options, reason, outliers):
    _use_native(monkeypatch, _ForbiddenNative)
    expected, expected_report = fd.clean(
        df.copy(), engine="pandas", return_report=True, **KW, **options
    )
    out, report = fd.clean(df.copy(), engine="freshcore", return_report=True, **KW, **options)

    assert report.backend == "pandas"
    (event,) = report.fallback_events
    assert event["backend"] == "freshcore"
    assert event["fallback_step"] == "pipeline"
    assert reason in event["fallback_reason"]
    # engine="pandas" wraps its frame in CleanResult (a DataFrame subclass).
    pd.testing.assert_frame_equal(out, pd.DataFrame(expected))
    assert [(a.step, a.column, a.count) for a in report.actions] == [
        (a.step, a.column, a.count) for a in expected_report.actions
    ]
    if outliers is not None:
        assert report.outliers_handled == expected_report.outliers_handled == outliers


@pytest.mark.parametrize(("df", "options", "reason", "outliers"), FALLBACK_CASES)
def test_fallback_policy_error_refuses(monkeypatch, df, options, reason, outliers):
    _use_native(monkeypatch, _ForbiddenNative)
    with pytest.raises(fd.FallbackError, match=re.escape(reason)):
        fd.clean(df, engine="freshcore", fallback_policy="error", **KW, **options)


def test_imputation_fills_what_pandas_fills(monkeypatch):
    _use_native(monkeypatch, _ForbiddenNative)
    df = pd.DataFrame({"x": [1.0, None, 3.0] * 20, "y": np.arange(60.0)})
    out = fd.clean(df, engine="freshcore", impute_strategy={"x": "mean"}, **KW)
    assert int(out["x"].isna().sum()) == 0

    bools = pd.DataFrame(
        {"b": pd.array([True, True, None, False], dtype="boolean"), "k": [1.0, 2, 3, 4]}
    )
    out = fd.clean(bools, engine="freshcore", impute="mode", **KW)
    assert out["b"].tolist() == [True, True, True, False]


@pytest.mark.parametrize(
    "options",
    [{"impute": "missforest"}, {"impute_strategy": {"x": "median"}}],
    ids=["missforest", "impute-strategy"],
)
def test_plan_preview_matches_recorded_fallback(monkeypatch, options):
    _use_native(monkeypatch, _ForbiddenNative)
    df = pd.DataFrame({"x": [1.0, None, 3.0, 4.0]})
    preview = fd.plan(df, engine="freshcore", **KW, **options)
    _, report = fd.clean(df, engine="freshcore", return_report=True, **KW, **options)
    assert preview.fallback_reason == report.fallback_events[0]["fallback_reason"]


@pytest.mark.parametrize(
    ("df", "options"),
    [
        pytest.param(
            _inf_frame([1.0, 2, 3, 4, 100.0]), {"outliers": "flag"}, id="finite-outliers"
        ),
        pytest.param(
            _inf_frame([1.0, np.inf, 3.0]), {"impute": "median"}, id="inf-without-outliers"
        ),
        pytest.param(
            pd.DataFrame({"b": pd.array([True, None, False], dtype="boolean")}),
            {"impute": "median"},
            id="boolean-median-is-a-skip-on-pandas-too",
        ),
        pytest.param(
            pd.DataFrame({"b": pd.array([True, False, True], dtype="boolean")}),
            {"impute": "mode"},
            id="boolean-without-missing",
        ),
        pytest.param(
            pd.DataFrame({"x": [1.0, None, 3.0]}),
            {"impute": "mean", "impute_strategy": {}},
            id="empty-impute-strategy",
        ),
    ],
)
def test_supported_shapes_stay_native(monkeypatch, df, options):
    native = _echo(monkeypatch)
    _, report = fd.clean(df, engine="freshcore", return_report=True, **KW, **options)
    assert native.calls == 1
    assert report.backend == "freshcore"
    assert report.fallback_events == []


# -- #323: duplicate detection and duplicate_ratio_action --------------------


def _dup_frame() -> pd.DataFrame:
    """12 rows: 2 all-missing, then 2 duplicates among the remaining 10 (20%)."""
    return pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0, 2, 3, 4, 5, 6, 7, 8, None, None],
            "b": ["x", "x", "x", "y", "y", "y", "y", "y", "y", "y", None, None],
        }
    )


def _dup_native_result(**extra) -> dict:
    return {
        "rows_before": 12,
        "rows_after": 10,
        "actions": [
            {
                "step": "drop_empty_rows",
                "column": None,
                "description": "dropped 2 all-missing row(s)",
                "count": 2,
            },
            {
                "step": "impute",
                "column": "a",
                "description": "filled 0 missing value(s) with mean (1.0)",
                "count": 0,
            },
        ],
        **extra,
    }


def _dup_actions(report) -> list[str]:
    return [a.description for a in report.actions if a.step == "drop_duplicates"]


def test_native_duplicate_count_is_reported_like_pandas(monkeypatch):
    _, expected = fd.clean(_dup_frame(), engine="pandas", return_report=True, **KW)
    _echo(monkeypatch, **_dup_native_result(duplicates_detected=2))
    _, report = fd.clean(_dup_frame(), engine="freshcore", return_report=True, **KW)

    assert report.backend == "freshcore"
    assert _dup_actions(report) == _dup_actions(expected)
    assert _dup_actions(report) == ["detected 2 duplicate row(s) (20.0%), none removed"]
    assert report.warnings == expected.warnings
    assert report.recommendations == expected.recommendations
    steps = [a.step for a in report.actions]
    assert steps.index("drop_duplicates") < steps.index("impute")


def test_native_duplicate_count_escalates_under_error(monkeypatch):
    with pytest.raises(DuplicateRatioError):
        fd.clean(_dup_frame(), engine="pandas", duplicate_ratio_action="error", **KW)
    _echo(monkeypatch, **_dup_native_result(duplicates_detected=2))
    with pytest.raises(DuplicateRatioError):
        fd.clean(_dup_frame(), engine="freshcore", duplicate_ratio_action="error", **KW)


def test_missing_native_duplicate_count_skips_detection(monkeypatch):
    native = _echo(monkeypatch, **_dup_native_result())
    _, report = fd.clean(_dup_frame(), engine="freshcore", return_report=True, **KW)

    assert native.calls == 1
    assert report.backend == "freshcore"
    assert _dup_actions(report) == []
    assert report.warnings == []


def test_missing_native_duplicate_count_falls_back_under_error(monkeypatch):
    native = _echo(monkeypatch, **_dup_native_result())
    with pytest.raises(DuplicateRatioError):
        fd.clean(_dup_frame(), engine="freshcore", duplicate_ratio_action="error", **KW)
    assert native.calls == 1

    below = fd.CleanConfig(**KW, duplicate_ratio_action="error", duplicate_threshold=0.5)
    _, report = fd.clean(_dup_frame(), config=below, engine="freshcore", return_report=True)
    assert report.backend == "pandas"
    (event,) = report.fallback_events
    assert event["fallback_step"] == "drop_duplicates"
    assert 'duplicate_ratio_action="error"' in event["fallback_reason"]

    with pytest.raises(fd.FallbackError, match="duplicate-row count"):
        fd.clean(
            _dup_frame(),
            config=below,
            engine="freshcore",
            fallback_policy="error",
        )


def _dropped_native_result() -> dict:
    result = _dup_native_result()
    result["actions"].insert(
        1,
        {
            "step": "drop_duplicates",
            "column": None,
            "description": 'dropped 2 duplicate row(s) (20.0% of rows, keep="first")',
            "count": 2,
        },
    )
    result["duplicates_removed"] = 2
    return result


def test_drop_duplicates_ratio_escalates_under_error(monkeypatch):
    options = {"drop_duplicates": True, "duplicate_ratio_action": "error"}
    with pytest.raises(DuplicateRatioError):
        fd.clean(_dup_frame(), engine="pandas", **KW, **options)
    _echo(monkeypatch, **_dropped_native_result())
    with pytest.raises(DuplicateRatioError):
        fd.clean(_dup_frame(), engine="freshcore", **KW, **options)


def test_drop_duplicates_high_ratio_warns_like_pandas(monkeypatch):
    _, expected = fd.clean(
        _dup_frame(), engine="pandas", drop_duplicates=True, return_report=True, **KW
    )
    _echo(monkeypatch, **_dropped_native_result())
    _, report = fd.clean(
        _dup_frame(), engine="freshcore", drop_duplicates=True, return_report=True, **KW
    )

    assert report.backend == "freshcore"
    (action,) = [a for a in report.actions if a.step == "drop_duplicates"]
    (pandas_action,) = [a for a in expected.actions if a.step == "drop_duplicates"]
    assert action.risk == pandas_action.risk == "medium"
    assert report.warnings == expected.warnings
    assert report.recommendations == expected.recommendations
