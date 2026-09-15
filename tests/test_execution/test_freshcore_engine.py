"""FreshCore adapter tests.

The Rust extension is optional, so these tests cover the always-available
fallback path and exercise native-result mapping with a tiny fake module.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.execution import EngineConfig, EngineSelector
from freshdata.execution.backends._freshcore import FreshCoreEngine


def _cfg(**kwargs) -> fd.CleanConfig:
    return fd.CleanConfig(strategy="conservative", fix_dtypes=False, verbose=False, **kwargs)


def test_engine_config_accepts_freshcore():
    cfg = EngineConfig(engine="freshcore")
    assert cfg.engine == "freshcore"
    assert EngineSelector.get_engine("freshcore", cfg).name == "freshcore"


def test_missing_native_module_falls_back_to_pandas(monkeypatch):
    # Simulate a missing extension so the test holds where it is built.
    monkeypatch.setattr(FreshCoreEngine, "_load_native", staticmethod(lambda: None))
    df = pd.DataFrame({"name": [" Alice ", "Bob"], "empty": [None, None]})
    out, report = fd.clean(df, config=_cfg(), engine="freshcore", return_report=True)

    assert isinstance(out, pd.DataFrame)
    assert report.backend == "pandas"
    assert report.fallback_events
    assert report.fallback_events[0]["backend"] == "freshcore"
    assert "freshdata_freshcore" in report.fallback_events[0]["fallback_reason"]


def test_unsupported_balanced_strategy_falls_back_even_with_native_module(monkeypatch):
    class FakeNative:
        @staticmethod
        def execute_plan(payload):  # pragma: no cover - must not be called
            raise AssertionError("unexpected native call")

    monkeypatch.setattr(FreshCoreEngine, "_load_native", staticmethod(lambda: FakeNative))
    df = pd.DataFrame({"x": [" 1 ", "2"]})
    _, report = fd.clean(df, engine="freshcore", return_report=True, verbose=False)

    assert report.backend == "pandas"
    assert report.fallback_events[0]["fallback_step"] == "pipeline"
    assert "accuracy-first decision engine" in report.fallback_events[0]["fallback_reason"]


def test_native_result_mapping_with_fake_module(monkeypatch):
    class FakeNative:
        @staticmethod
        def execute_plan(payload):
            assert payload["config"]["string_case"] == "lower"
            return {
                "rows_before": 2,
                "rows_after": 2,
                "cols_before": 1,
                "cols_after": 1,
                "missing_before": 0,
                "missing_after": 0,
                "duplicates_removed": 0,
                "outliers_handled": 0,
                "columns_dropped": [],
                "columns_imputed": [],
                "actions": [
                    {
                        "step": "normalize_case",
                        "column": "name",
                        "description": "converted text to lower",
                        "count": 2,
                    }
                ],
                "stage_timings": [("clean_strings", 0.001)],
                "columns": [
                    {"name": "name", "dtype": "string", "values": ["alice", "bob"]},
                ],
            }

    monkeypatch.setattr(FreshCoreEngine, "_load_native", staticmethod(lambda: FakeNative))
    df = pd.DataFrame({"name": ["Alice", "BOB"]})
    out, report = fd.clean(
        df,
        config=_cfg(string_case="lower"),
        engine="freshcore",
        return_report=True,
    )

    assert out["name"].tolist() == ["alice", "bob"]
    assert report.backend == "freshcore"
    assert report.actions[0].step == "normalize_case"
    assert report.stage_timings == [
        {"backend": "freshcore", "stage": "clean_strings", "seconds": 0.001}
    ]
    assert report.to_dict()["stage_timings"][0]["backend"] == "freshcore"


class _CastingNative:
    """Echoes the input but reports *cast* columns as a native fix_dtypes would."""

    calls = 0
    cast: dict = {}

    @classmethod
    def execute_plan(cls, payload):
        cls.calls += 1
        columns = [cls.cast.get(c["name"], c) for c in payload["columns"]]
        return {"columns": columns, "actions": []}


def _casting(monkeypatch, **cast) -> type[_CastingNative]:
    native = type("CastingNative", (_CastingNative,), {"calls": 0, "cast": cast})
    monkeypatch.setattr(FreshCoreEngine, "_load_native", staticmethod(lambda: native))
    return native


def _cast_cfg(**kwargs) -> fd.CleanConfig:
    return fd.CleanConfig(strategy="conservative", verbose=False, **kwargs)


_YES_NO = pd.DataFrame({"flag": ["yes", "no", None, "yes"], "k": [1.0, 2, 3, 4]})
_BOOL_CAST = {"name": "flag", "dtype": "bool", "values": [True, False, None, True]}
_INF_TEXT = pd.DataFrame({"x": ["1", "2", "3", "100", "inf"]})
_INF_CAST = {"name": "x", "dtype": "float", "values": [1.0, 2.0, 3.0, 100.0, float("inf")]}


@pytest.mark.parametrize("impute", ["mode", "auto"])
def test_native_bool_cast_with_missing_values_falls_back_for_imputation(monkeypatch, impute):
    native = _casting(monkeypatch, flag=_BOOL_CAST)
    out, report = fd.clean(
        _YES_NO.copy(), config=_cast_cfg(impute=impute), engine="freshcore", return_report=True
    )

    assert native.calls == 1
    assert report.backend == "pandas"
    [event] = report.fallback_events
    assert event["fallback_step"] == "impute"
    assert "'flag'" in event["fallback_reason"]
    assert "cast to boolean" in event["fallback_reason"]
    assert out["flag"].tolist() == [True, False, True, True]


@pytest.mark.parametrize("method", ["zscore", "iqr"])
def test_native_float_cast_holding_inf_falls_back_for_outliers(monkeypatch, method):
    native = _casting(monkeypatch, x=_INF_CAST)
    _, report = fd.clean(
        _INF_TEXT.copy(),
        config=_cast_cfg(outliers="flag", outlier_method=method),
        engine="freshcore",
        return_report=True,
    )

    assert native.calls == 1
    assert report.backend == "pandas"
    [event] = report.fallback_events
    assert event["fallback_step"] == "outliers"
    assert "'x'" in event["fallback_reason"]
    assert "±inf" in event["fallback_reason"]


@pytest.mark.parametrize(
    ("frame", "cast", "options"),
    [
        pytest.param(_YES_NO, {"flag": _BOOL_CAST}, {"impute": "mode"}, id="bool"),
        pytest.param(_INF_TEXT, {"x": _INF_CAST}, {"outliers": "flag"}, id="inf"),
    ],
)
def test_native_cast_fallback_honours_error_policy(monkeypatch, frame, cast, options):
    _casting(monkeypatch, **cast)
    with pytest.raises(fd.FallbackError):
        fd.clean(
            frame.copy(),
            config=_cast_cfg(**options),
            engine="freshcore",
            fallback_policy="error",
        )


@pytest.mark.parametrize(
    ("frame", "cast", "options"),
    [
        pytest.param(
            _YES_NO, {"flag": _BOOL_CAST}, {"outliers": "flag"}, id="bool-cast-without-impute"
        ),
        pytest.param(
            _YES_NO, {"flag": _BOOL_CAST}, {"impute": "median"}, id="bool-cast-median-skips"
        ),
        pytest.param(
            _YES_NO,
            {"flag": {**_BOOL_CAST, "values": [True, False, False, True]}},
            {"impute": "mode"},
            id="bool-cast-without-missing",
        ),
        pytest.param(
            _INF_TEXT, {"x": _INF_CAST}, {"impute": "median"}, id="inf-cast-without-outliers"
        ),
        pytest.param(
            _INF_TEXT,
            {"x": {**_INF_CAST, "values": [1.0, 2.0, 3.0, 100.0, None]}},
            {"outliers": "flag"},
            id="finite-float-cast",
        ),
        pytest.param(
            pd.DataFrame({"x": ["a", "b", None], "k": [1.0, 2, 3]}),
            {},
            {"impute": "mode", "outliers": "flag"},
            id="no-cast",
        ),
    ],
)
def test_frames_without_mishandled_casts_stay_native(monkeypatch, frame, cast, options):
    native = _casting(monkeypatch, **cast)
    _, report = fd.clean(
        frame.copy(),
        config=_cast_cfg(**options),
        engine="freshcore",
        fallback_policy="error",
        return_report=True,
    )

    assert native.calls == 1
    assert report.backend == "freshcore"
    assert report.fallback_events == []


def test_string_case_available_on_reference_pipeline():
    df = pd.DataFrame({"name": ["Alice", "BOB"]})
    out, report = fd.clean(df, config=_cfg(string_case="lower"), return_report=True)

    assert out["name"].tolist() == ["alice", "bob"]
    assert any(a.step == "normalize_case" for a in report.actions)
