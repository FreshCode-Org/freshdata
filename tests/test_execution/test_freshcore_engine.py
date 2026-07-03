"""FreshCore adapter tests.

The Rust extension is optional, so these tests cover the always-available
fallback path and exercise native-result mapping with a tiny fake module.
"""

from __future__ import annotations

import pandas as pd

import freshdata as fd
from freshdata.execution import EngineConfig, EngineSelector
from freshdata.execution.backends._freshcore import FreshCoreEngine


def _cfg(**kwargs) -> fd.CleanConfig:
    return fd.CleanConfig(strategy="conservative", fix_dtypes=False, verbose=False, **kwargs)


def test_engine_config_accepts_freshcore():
    cfg = EngineConfig(engine="freshcore")
    assert cfg.engine == "freshcore"
    assert EngineSelector.get_engine("freshcore", cfg).name == "freshcore"


def test_missing_native_module_falls_back_to_pandas():
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


def test_string_case_available_on_reference_pipeline():
    df = pd.DataFrame({"name": ["Alice", "BOB"]})
    out, report = fd.clean(df, config=_cfg(string_case="lower"), return_report=True)

    assert out["name"].tolist() == ["alice", "bob"]
    assert any(a.step == "normalize_case" for a in report.actions)
