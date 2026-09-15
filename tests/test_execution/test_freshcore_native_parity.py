"""Native FreshCore duplicate detection matches pandas (#323).

These tests run the real ``freshdata_freshcore`` extension and are skipped when
it is not built::

    maturin develop --manifest-path crates/freshcore/Cargo.toml --features extension-module

``test_freshcore_fallbacks.py`` covers the adapter with a fake module, including
native modules built before the duplicate count existed.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.execution.backends._freshcore import FreshCoreEngine
from freshdata.steps.duplicates import DuplicateRatioError

native_module = pytest.importorskip("freshdata_freshcore")

KW = {"strategy": "conservative", "verbose": False}


class _CountingNative:
    """Delegates to the real extension and counts the calls."""

    calls = 0

    @classmethod
    def execute_plan(cls, payload):
        cls.calls += 1
        return native_module.execute_plan(payload)


@pytest.fixture
def native(monkeypatch):
    spy = type("CountingNative", (_CountingNative,), {"calls": 0})
    monkeypatch.setattr(FreshCoreEngine, "_load_native", staticmethod(lambda: spy))
    return spy


def _clean_both(native, df: pd.DataFrame, **options):
    """Clean on pandas and natively; the native run may not fall back."""
    _, expected = fd.clean(df.copy(), engine="pandas", return_report=True, **KW, **options)
    _, report = fd.clean(
        df.copy(),
        engine="freshcore",
        return_report=True,
        fallback_policy="error",
        **KW,
        **options,
    )
    assert native.calls == 1
    assert report.backend == "freshcore"
    assert report.fallback_events == []
    return expected, report


def _duplicate_view(report, descriptions: bool = True) -> dict:
    return {
        "actions": [
            (a.description if descriptions else None, a.count, a.risk)
            for a in report.actions
            if a.step == "drop_duplicates"
        ],
        "warnings": [w for w in report.warnings if "duplicate" in w],
        "recommendations": [r for r in report.recommendations if "duplicated" in r],
    }


def _detections(report) -> list[str]:
    return [a.description for a in report.actions if a.step == "drop_duplicates"]


# -- the #323 reproduction ---------------------------------------------------


def _repro_frame() -> pd.DataFrame:
    """10 rows, 9 of them duplicates."""
    return pd.DataFrame({"a": [1.0] * 10, "b": ["x"] * 10})


REPRO = {"fix_dtypes": False}


def test_issue_323_repro_raises_natively_under_error(native):
    with pytest.raises(DuplicateRatioError) as expected:
        fd.clean(_repro_frame(), engine="pandas", duplicate_ratio_action="error", **KW, **REPRO)
    # fallback_policy="error" turns any pandas fallback into FallbackError, so
    # DuplicateRatioError here comes from the native duplicate count.
    with pytest.raises(DuplicateRatioError) as raised:
        fd.clean(
            _repro_frame(),
            engine="freshcore",
            duplicate_ratio_action="error",
            fallback_policy="error",
            **KW,
            **REPRO,
        )
    assert native.calls == 1
    assert str(raised.value) == str(expected.value)


def test_issue_323_repro_warns_like_pandas_under_warn(native):
    expected, report = _clean_both(native, _repro_frame(), duplicate_ratio_action="warn", **REPRO)

    assert _duplicate_view(report) == _duplicate_view(expected)
    assert _detections(report) == ["detected 9 duplicate row(s) (90.0%), none removed"]
    assert len(_duplicate_view(report)["warnings"]) == 1
    assert len(_duplicate_view(report)["recommendations"]) == 1


def test_ignore_is_rejected_on_both_engines(native):
    # duplicate_ratio_action accepts only "warn" and "error"; "ignore" fails
    # config validation before either engine runs.
    for engine in ("pandas", "freshcore"):
        with pytest.raises(ValueError, match="duplicate_ratio_action must be one of"):
            fd.clean(
                _repro_frame(), engine=engine, duplicate_ratio_action="ignore", **KW, **REPRO
            )
    assert native.calls == 0


@pytest.mark.parametrize("action", ["warn", "error"])
@pytest.mark.parametrize(
    ("rows", "description"),
    [
        pytest.param(20, "detected 1 duplicate row(s) (5.0%), none removed", id="below"),
        pytest.param(10, "detected 1 duplicate row(s) (10.0%), none removed", id="at-threshold"),
    ],
)
def test_ratio_within_threshold_reports_without_warning(native, action, rows, description):
    df = pd.DataFrame({"a": [float(i) for i in range(rows - 1)] + [0.0], "b": ["x"] * rows})
    expected, report = _clean_both(native, df, duplicate_ratio_action=action, **REPRO)

    assert _duplicate_view(report) == _duplicate_view(expected)
    assert _detections(report) == [description]
    assert _duplicate_view(report)["warnings"] == []
    assert _duplicate_view(report)["recommendations"] == []


def test_no_duplicates_records_no_detection(native):
    df = pd.DataFrame({"a": [" x", "y", "z"], "v": [1.0, 2.0, 3.0]})
    expected, report = _clean_both(native, df, **REPRO)
    assert _detections(report) == _detections(expected) == []


# -- counts taken after cleaning, like the pandas step ----------------------


CLEANING_CASES = [
    pytest.param(
        pd.DataFrame(
            {"name": [" alice", "alice ", "alice", "bob", "carol"], "v": [1.0, 1.0, 1.0, 2, 3]}
        ),
        {"fix_dtypes": False},
        2,
        id="whitespace",
    ),
    pytest.param(
        pd.DataFrame({"code": ["N/A", "", "-", "x", "y"], "v": [1.0, 1.0, 1.0, 2, 3]}),
        {"fix_dtypes": False},
        2,
        id="sentinels",
    ),
    pytest.param(
        pd.DataFrame({"name": ["Alice", "ALICE", "alice", "Bob"], "v": [1.0, 1.0, 1.0, 1.0]}),
        {"fix_dtypes": False, "string_case": "lower"},
        2,
        id="string-case",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "a": [1.0, 1.0, 1.0, 2, 3, 4, 5, 6, 7, 8, None, None],
                "b": ["x", "x", "x", "y", "y", "y", "y", "y", "y", "y", None, None],
            }
        ),
        {"fix_dtypes": False},
        2,
        id="empty-rows-leave-the-denominator",
    ),
    pytest.param(
        pd.DataFrame({"a": [" x", "x", None, None], "b": ["N/A", None, "", None]}),
        {"fix_dtypes": False},
        1,
        id="sentinels-make-empty-rows",
    ),
    pytest.param(
        pd.DataFrame(
            {"n": ["1", "1.0", "2", "2.00", "3", "4"], "k": ["a", "a", "b", "b", "c", "d"]}
        ),
        {"fix_dtypes": True},
        2,
        id="numeric-casts",
    ),
]


@pytest.mark.parametrize(("df", "options", "n_dup"), CLEANING_CASES)
def test_detected_count_matches_pandas_after_cleaning(native, df, options, n_dup):
    # The raw frame has a different count: only cleaning makes these rows equal.
    assert int(df.duplicated().sum()) != n_dup
    expected, report = _clean_both(native, df, **options)

    assert _duplicate_view(report) == _duplicate_view(expected)
    (detection,) = _detections(report)
    assert detection.startswith(f"detected {n_dup} duplicate row(s) (")


@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize(("df", "options", "n_dup"), CLEANING_CASES)
def test_dropped_count_matches_pandas_after_cleaning(native, df, options, n_dup, keep):
    expected, report = _clean_both(
        native, df, drop_duplicates=True, duplicate_keep=keep, **options
    )

    assert report.duplicates_removed == expected.duplicates_removed == n_dup
    # Drop descriptions spell the keep policy differently (keep="first" vs
    # keep='first'); counts, risk, warnings and recommendations must match.
    assert _duplicate_view(report, descriptions=False) == _duplicate_view(
        expected, descriptions=False
    )


# -- the native result contract ---------------------------------------------


def _native_result(df: pd.DataFrame, **options) -> dict:
    config = fd.CleanConfig(**KW, **options)
    return native_module.execute_plan(FreshCoreEngine()._payload(df, config))


def test_native_result_reports_detection_count_and_stage():
    df = pd.DataFrame({"a": [" x", "x", "y"], "v": [1.0, 1.0, None]})

    detected = _native_result(df, fix_dtypes=False, impute="mean")
    assert detected["duplicates_detected"] == 1
    assert detected["duplicates_removed"] == 0
    assert detected["rows_after"] == 3
    stages = [stage for stage, _ in detected["stage_timings"]]
    assert "drop_duplicates" not in stages
    # Counted after casts and before imputation, where the pandas step runs.
    assert stages.index("fix_dtypes") < stages.index("detect_duplicates") < stages.index("impute")

    dropped = _native_result(df, fix_dtypes=False, impute="mean", drop_duplicates=True)
    assert dropped["duplicates_detected"] is None
    assert dropped["duplicates_removed"] == 1
    assert "detect_duplicates" not in [stage for stage, _ in dropped["stage_timings"]]


def test_detection_counts_before_imputation_like_pandas(native):
    # Imputation would make rows 1 and 2 equal; neither engine counts them.
    df = pd.DataFrame({"a": [1.0, 2.0, None, 2.0], "b": ["x", "y", "y", "z"]})
    expected, report = _clean_both(native, df, impute="mean", **REPRO)
    assert _detections(report) == _detections(expected) == []
