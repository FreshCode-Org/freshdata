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


# -- native casts that build columns the kernels mishandle -------------------
#
# ``fix_dtypes`` runs natively, so a text column can turn into a boolean column
# with missing values (never imputed natively) or a float column holding ±inf
# (which leaves the native outlier fences undefined). The input frame shows
# neither, so the adapter checks the native result and falls back.


def _yes_no_frame(missing: bool = True) -> pd.DataFrame:
    flags = ["yes", "no", None if missing else "no", "yes", "Yes"]
    return pd.DataFrame({"flag": flags, "k": [1.0, 2, 3, 4, 5]})


def _inf_text_frame(values: list[str] | None = None) -> pd.DataFrame:
    if values is None:
        values = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "100", "inf"]
    return pd.DataFrame({"x": values})


INF_ZSCORE = {"outliers": "flag", "outlier_method": "zscore", "outlier_factor": 2.0}
CAST_REPROS = [
    pytest.param(_yes_no_frame(), {"impute": "mode"}, "impute", "'flag'", id="bool-mode"),
    pytest.param(_yes_no_frame(), {"impute": "auto"}, "impute", "'flag'", id="bool-auto"),
    pytest.param(_inf_text_frame(), INF_ZSCORE, "outliers", "'x'", id="inf-zscore"),
    pytest.param(
        _inf_text_frame(["1", "2", "3", "4", "inf", "inf", "inf", "inf"]),
        {"outliers": "flag", "outlier_method": "iqr"},
        "outliers",
        "'x'",
        id="inf-iqr",
    ),
    pytest.param(
        _inf_text_frame(["1", "2", "3", "4", "5", "6", "7", "8", "9", "100", "-inf"]),
        INF_ZSCORE,
        "outliers",
        "'x'",
        id="negative-inf-zscore",
    ),
]


def test_native_casts_diverge_without_the_fallback():
    """Pins the kernel behaviour the adapter guards against."""
    booleans = _native_result(_yes_no_frame(), impute="mode")
    [flag] = [c for c in booleans["columns"] if c["name"] == "flag"]
    assert flag["dtype"] == "bool"
    assert None in flag["values"]

    infinite = _native_result(_inf_text_frame(), **INF_ZSCORE)
    assert [c["name"] for c in infinite["columns"]] == ["x"]  # no flag column
    assert infinite["outliers_handled"] == 0


@pytest.mark.parametrize(("df", "options", "step", "column"), CAST_REPROS)
def test_native_cast_repros_fall_back_to_the_pandas_result(native, df, options, step, column):
    expected, _ = fd.clean(df.copy(), engine="pandas", return_report=True, **KW, **options)
    out, report = fd.clean(df.copy(), engine="freshcore", return_report=True, **KW, **options)

    assert native.calls == 1
    assert report.backend == "pandas"
    [event] = report.fallback_events
    assert event["fallback_step"] == step
    assert column in event["fallback_reason"]
    pd.testing.assert_frame_equal(pd.DataFrame(out), pd.DataFrame(expected))


def test_native_cast_repro_values_match_pandas(native):
    out = fd.clean(_yes_no_frame(), engine="freshcore", impute="mode", **KW)
    assert out["flag"].tolist() == [True, False, True, True, True]

    flagged = fd.clean(_inf_text_frame(), engine="freshcore", **INF_ZSCORE, **KW)
    assert flagged["x_outlier"].tolist() == [False] * 9 + [True, True]


@pytest.mark.parametrize(("df", "options", "step", "column"), CAST_REPROS)
def test_native_cast_repros_raise_under_error_policy(native, df, options, step, column):
    with pytest.raises(fd.FallbackError, match=f"step {step!r}"):
        fd.clean(df.copy(), engine="freshcore", fallback_policy="error", **KW, **options)
    assert native.calls == 1


@pytest.mark.parametrize(
    ("df", "options"),
    [
        pytest.param(_yes_no_frame(missing=False), {"impute": "mode"}, id="bool-cast-no-missing"),
        pytest.param(_yes_no_frame(), {"outliers": "flag"}, id="bool-cast-without-impute"),
        pytest.param(_inf_text_frame(), {"impute": "median"}, id="inf-cast-without-outliers"),
        pytest.param(
            _inf_text_frame(["1", "2", "3", "4", "5", "6", "7", "8", "9", "100", None]),
            {**INF_ZSCORE, "impute": "median"},
            id="finite-float-cast",
        ),
        pytest.param(
            pd.DataFrame({"s": ["a", "b", None, "a"], "v": [1.0, 2.0, 3.0, 40.0]}),
            {"impute": "mode", "outliers": "flag"},
            id="no-cast",
        ),
    ],
)
def test_frames_without_mishandled_casts_stay_native(native, df, options):
    expected, _ = fd.clean(df.copy(), engine="pandas", return_report=True, **KW, **options)
    out, report = fd.clean(
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
    # Values match; dtypes may not (e.g. native boolean vs pandas bool).
    pd.testing.assert_frame_equal(pd.DataFrame(out), pd.DataFrame(expected), check_dtype=False)


# -- zero-IQR outlier fallback ------------------------------------------------


def _outlier_frame() -> pd.DataFrame:
    """A zero-IQR column with spikes, a zero-IQR second mode, a constant and
    an ordinary column."""
    n = 100
    return pd.DataFrame({
        "spiky": [0.0] * 95 + [1000.0, 5000.0, 2.0, 3.0, -800.0],
        "second_mode": [0.0] * 70 + [float(s * i) for i in range(1, 16) for s in (-1, 1)],
        "constant": [4.0] * n,
        "normal": [(((i * 37) % 23) - 11) / 7.0 for i in range(n - 1)] + [90.0],
    })


def _outlier_counts(report) -> dict:
    return {a.column: a.count for a in report.actions if a.step == "outliers"}


@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_outlier_flags_match_pandas_with_zero_iqr(native, method):
    df = _outlier_frame()
    options = {"outliers": "flag", "outlier_method": method, "fix_dtypes": False}
    expected_frame = pd.DataFrame(fd.clean(df.copy(), engine="pandas", **KW, **options))
    expected, report = _clean_both(native, df, **options)
    native_frame = pd.DataFrame(fd.clean(df.copy(), engine="freshcore", **KW, **options))

    assert _outlier_counts(report) == _outlier_counts(expected)
    assert list(native_frame.columns) == list(expected_frame.columns)
    for flag in [c for c in expected_frame.columns if str(c).endswith("_outlier")]:
        assert native_frame[flag].astype(bool).tolist() == expected_frame[flag].tolist()
    if method == "iqr":
        assert _outlier_counts(expected) == {"spiky": 3, "normal": 1}


def test_outlier_clips_match_pandas_with_zero_iqr():
    # The adapter sends clip to pandas (skew-aware capping), so drive the
    # kernel directly. These columns hold non-positive values, so pandas never
    # widens its fences in log space and both engines clip to detection fences.
    df = _outlier_frame()
    options = {"outliers": "clip", "outlier_method": "iqr", "fix_dtypes": False}
    expected, report = fd.clean(df.copy(), engine="pandas", return_report=True, **KW, **options)
    result = _native_result(df, **options)

    assert result["outliers_handled"] == report.outliers_handled == 4
    columns = {c["name"]: c["values"] for c in result["columns"]}
    for name in df.columns:
        # MeanAD is summed in a different order, so allow round-off.
        assert columns[name] == pytest.approx(expected[name].tolist(), rel=1e-12, abs=0.0)
    assert columns["spiky"][95] < 1000.0 and columns["spiky"][96] == columns["spiky"][95]
    assert columns["spiky"][97:99] == [2.0, 3.0]
    assert columns["second_mode"] == df["second_mode"].tolist()
    assert columns["constant"] == df["constant"].tolist()
