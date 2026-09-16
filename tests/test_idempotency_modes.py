"""Cleaning twice must equal cleaning once -- in every mode, not just one.

Idempotency was asserted for the ``balanced`` strategy only:
``tests/expectations.py:174`` drives ``{"balanced": {"idempotent": true}}`` for
the fixtures, and ``tests/test_properties.py:17`` covers defaults. Nothing
checked ``conservative`` or ``aggressive``, any ``semantic_mode``, a compiled
context policy, lossy text options, imputation, outlier clipping, the native
engines, or memory replay -- even though ``test_compare_matrix.py`` exercises
all three strategies.

That matters because a second pass sees its own output: a repair that is not a
fixed point (case folding that re-triggers, a sentinel that re-matches, a
dominant-variant vote that flips once the variants change) drifts on every
rerun, and a pipeline that cleans on ingest and again on export would keep
moving the data.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd


def _frame() -> pd.DataFrame:
    """Dirty enough to engage several steps at once."""
    return pd.DataFrame(
        {
            "row_key": [f"r{i}" for i in range(10)],
            "customer_id": [f"{i:03d}" for i in range(1, 11)],
            "amount": [10.5, 20.0, "  30.5 ", 40.0, 50.0, "N/A", 70.0, "$1,200.50", 90.0, 100.0],
            "country": ["US", "GB", "FR", "DE", "JP", "CA", "AU", "BR", "IN", "US"],
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
            "note": ["a ", "b", "  c", "d", "e", "f", "g", "h", "i", "j"],
        }
    )


def _native_frame() -> pd.DataFrame:
    """Uniformly typed, so polars and duckdb stay on their native path.

    A mixed-type object column makes both engines disclose a fallback to
    pandas, which would make an "engine" test silently test pandas twice.
    """
    return pd.DataFrame(
        {
            "row_key": [f"r{i}" for i in range(10)],
            "code": [f"{i:03d}" for i in range(1, 11)],
            "note": ["a ", "b", "  c", "d", "N/A", "f", "g", "h", "i", "j"],
            "qty": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )


def _assert_fixed_point(frame: pd.DataFrame, **kw) -> None:
    once = pd.DataFrame(fd.clean(frame, verbose=False, **kw))
    twice = pd.DataFrame(fd.clean(once.copy(), verbose=False, **kw))
    pd.testing.assert_frame_equal(twice, once)


def test_defaults_are_a_fixed_point():
    _assert_fixed_point(_frame())


@pytest.mark.parametrize("strategy", ["conservative", "balanced", "aggressive"])
def test_every_strategy_is_a_fixed_point(strategy):
    """Only `balanced` was covered before."""
    _assert_fixed_point(_frame(), strategy=strategy)


@pytest.mark.parametrize("mode", ["assist", "review", "auto"])
def test_every_semantic_mode_is_a_fixed_point(mode):
    """`auto` matters most: it is the only mode that applies repairs itself."""
    _assert_fixed_point(_frame(), semantic_mode=mode)


def test_a_compiled_context_policy_is_a_fixed_point():
    _assert_fixed_point(_frame(), context="Never modify customer_id values.")


def test_lossy_text_options_are_a_fixed_point():
    """Case folding must not re-trigger on its own output."""
    _assert_fixed_point(_frame(), string_case="lower")


def test_imputation_is_a_fixed_point():
    """The second pass sees no missing values, so it must do nothing."""
    _assert_fixed_point(_frame(), impute="mean")


def test_outlier_clipping_is_a_fixed_point():
    """Clipping must not keep pulling the fences inward."""
    df = _frame()
    df.loc[9, "amount"] = 9999.0
    _assert_fixed_point(df, outliers="clip")


@pytest.mark.parametrize("engine", ["pandas", "polars", "duckdb"])
def test_each_engine_is_a_fixed_point_on_its_native_path(engine):
    if engine != "pandas":
        pytest.importorskip(engine)
    kw = {"engine": engine, "strategy": "conservative", "fix_dtypes": False}
    once, report = fd.clean(_native_frame(), verbose=False, return_report=True, **kw)
    twice = fd.clean(pd.DataFrame(once).copy(), verbose=False, **kw)
    pd.testing.assert_frame_equal(pd.DataFrame(twice), pd.DataFrame(once))
    if engine != "pandas":
        # Guard the guard: if this fell back, the test would be testing pandas.
        assert report.backend == engine
        assert not report.fallback_events


def test_memory_replay_is_a_fixed_point():
    """Replaying learned decisions must not keep changing the frame."""
    memory = fd.CleaningMemory(dataset_id="idempotency")
    once = pd.DataFrame(
        fd.clean(_frame(), verbose=False, semantic_mode="auto", memory=memory)
    )
    twice = pd.DataFrame(
        fd.clean(once.copy(), verbose=False, semantic_mode="auto", memory=memory)
    )
    pd.testing.assert_frame_equal(twice, once)
