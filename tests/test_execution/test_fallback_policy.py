"""fallback_policy (allow/warn/error), execution-honesty report fields, and
the Polars-native subset dedup."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.execution import (
    EngineConfig,
    FallbackError,
    FallbackWarning,
    PlanGenerator,
    enforce_fallback_policy,
)

pl = pytest.importorskip("polars")


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "email": ["a@x", "b@y", "a@x", "c@z", "b@y", "a@x"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "name": [" al ", "bo", "cy", "di", "ed", "fi"],
        }
    )


NATIVE = {"strategy": "conservative", "fix_dtypes": False, "verbose": False}


# -- EngineConfig validation ----------------------------------------------


def test_engine_config_rejects_bad_policy():
    with pytest.raises(ValueError, match="fallback_policy"):
        EngineConfig(fallback_policy="explode")


def test_enforce_helper_allow_is_noop():
    enforce_fallback_policy(EngineConfig(fallback_policy="allow"), "polars", "pipeline", "x")


# -- allow / warn / error through fd.clean ---------------------------------


def test_allow_records_fallback_silently(messy_df, recwarn):
    # default balanced strategy forces the whole-pipeline pandas fallback
    cleaned, report = fd.clean(messy_df, engine="polars", return_report=True)
    assert report.backend == "pandas"
    assert report.requested_backend == "polars"
    assert len(report.fallback_events) == 1
    assert not [w for w in recwarn if issubclass(w.category, FallbackWarning)]


def test_warn_emits_fallback_warning(messy_df):
    with pytest.warns(FallbackWarning, match="decision engine"):
        fd.clean(messy_df, engine="polars", fallback_policy="warn")


def test_error_raises_before_pandas_runs(messy_df):
    with pytest.raises(FallbackError) as exc:
        fd.clean(messy_df, engine="polars", fallback_policy="error")
    # the message names the exact trigger and the native escape hatch
    assert "decision engine" in str(exc.value)
    assert 'strategy="conservative"' in str(exc.value)


def test_error_policy_lets_native_run_through(messy_df):
    cleaned, report = fd.clean(
        messy_df, engine="polars", fallback_policy="error", return_report=True, **NATIVE
    )
    assert report.backend == "polars"
    assert report.fallback_events == []


def test_pandas_engine_rejects_policy(messy_df):
    with pytest.raises(TypeError, match="native engines"):
        fd.clean(messy_df, fallback_policy="error")


def test_policy_via_engine_config(messy_df):
    cfg = EngineConfig(engine="polars", fallback_policy="error")
    with pytest.raises(FallbackError):
        fd.clean(messy_df, engine_config=cfg)


def test_clean_kwarg_overrides_engine_config(messy_df):
    cfg = EngineConfig(engine="polars", fallback_policy="allow")
    with pytest.raises(FallbackError):
        fd.clean(messy_df, engine_config=cfg, fallback_policy="error")


# -- execution-honesty report fields ---------------------------------------


def test_report_fields_on_native_run(messy_df):
    cleaned, report = fd.clean(messy_df, engine="polars", return_report=True, **NATIVE)
    assert report.requested_backend == "polars"
    assert report.backend == "polars"
    assert report.rows_materialized == len(cleaned)
    assert report.peak_memory is None or report.peak_memory > 0
    payload = report.to_dict()
    assert payload["requested_backend"] == "polars"
    assert payload["rows_materialized"] == len(cleaned)


def test_report_fields_on_fallback_run(messy_df):
    cleaned, report = fd.clean(messy_df, engine="polars", return_report=True)
    assert report.requested_backend == "polars"
    assert report.backend == "pandas"
    assert report.rows_materialized == len(cleaned)


def test_native_handle_leaves_rows_materialized_none(messy_df):
    result, report = fd.clean(
        messy_df,
        engine="polars",
        output_format="polars-lazy",
        return_report=True,
        **NATIVE,
    )
    assert isinstance(result, pl.LazyFrame)
    assert report.rows_materialized is None
    assert report.materialized is False


def test_requested_backend_auto(messy_df):
    cleaned, report = fd.clean(
        messy_df, engine="auto", return_report=True, **NATIVE
    )
    assert report.requested_backend == "auto"
    assert report.backend in ("pandas", "polars", "duckdb")


# -- Polars-native subset dedup ---------------------------------------------


@pytest.mark.parametrize("keep", ["first", "last"])
def test_subset_dedupe_polars_parity(messy_df, keep):
    kw = dict(duplicate_subset=("email",), duplicate_keep=keep, return_report=True, **NATIVE)
    ref, ref_rep = fd.clean(messy_df, **kw)
    nat, nat_rep = fd.clean(messy_df, engine="polars", fallback_policy="error", **kw)
    pd.testing.assert_frame_equal(
        ref.reset_index(drop=True), nat.reset_index(drop=True)
    )
    assert nat_rep.backend == "polars"
    assert nat_rep.fallback_events == []
    assert ref_rep.duplicates_removed == nat_rep.duplicates_removed


def test_subset_dedupe_with_nulls_parity():
    df = pd.DataFrame(
        {
            "k": ["a", None, "a", None, "b"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    kw = dict(duplicate_subset=("k",), return_report=True, **NATIVE)
    ref, _ = fd.clean(df, **kw)
    nat, rep = fd.clean(df, engine="polars", fallback_policy="error", **kw)
    pd.testing.assert_frame_equal(ref.reset_index(drop=True), nat.reset_index(drop=True))
    assert rep.backend == "polars"


def test_subset_dedupe_unicode_keys():
    df = pd.DataFrame({"k": ["café", "café", "日本", "naïve"], "v": [1.0, 2.0, 3.0, 4.0]})
    kw = dict(duplicate_subset=("k",), **NATIVE)
    ref = fd.clean(df, **kw)
    nat = fd.clean(df, engine="polars", fallback_policy="error", **kw)
    pd.testing.assert_frame_equal(ref.reset_index(drop=True), nat.reset_index(drop=True))


def test_subset_dedupe_missing_column_raises(messy_df):
    with pytest.raises(ValueError, match="duplicate_subset column"):
        fd.clean(
            messy_df,
            engine="polars",
            drop_duplicates=True,
            duplicate_subset=("nope",),
            **NATIVE,
        )


def test_subset_dedupe_multi_column_parity():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "region": rng.choice(["eu", "us", "apac"], 200),
            "sku": rng.choice(["a", "b", "c", "d"], 200),
            "qty": rng.integers(1, 9, 200).astype(float),
        }
    )
    kw = dict(duplicate_subset=("region", "sku"), **NATIVE)
    ref = fd.clean(df, **kw)
    nat = fd.clean(df, engine="polars", fallback_policy="error", **kw)
    pd.testing.assert_frame_equal(ref.reset_index(drop=True), nat.reset_index(drop=True))


def test_subset_dedupe_still_falls_back_on_duckdb(messy_df):
    pytest.importorskip("duckdb")
    cleaned, report = fd.clean(
        messy_df,
        engine="duckdb",
        drop_duplicates=True,
        duplicate_subset=("email",),
        return_report=True,
        **NATIVE,
    )
    assert report.backend == "pandas"
    assert any(
        "subset" in e["fallback_reason"] for e in report.fallback_events
    )


def test_plan_generator_backend_awareness():
    # Subset semantics only force a fallback when removal is actually on;
    # detection-only (the default) counts natively on every backend.
    cfg = CleanConfig(
        strategy="conservative", fix_dtypes=False, duplicate_subset=("a",),
        drop_duplicates=True, verbose=False
    )
    assert PlanGenerator(cfg).fallback_reason() is not None
    assert PlanGenerator(cfg, backend="duckdb").fallback_reason() is not None
    assert PlanGenerator(cfg, backend="polars").fallback_reason() is None


def test_subset_dedupe_keep_drop_still_falls_back(messy_df):
    # keep="drop"/"aggregate" stay on the pandas reference even on polars
    cleaned, report = fd.clean(
        messy_df,
        engine="polars",
        drop_duplicates=True,
        duplicate_subset=("email",),
        duplicate_keep="drop",
        return_report=True,
        **NATIVE,
    )
    assert report.backend == "pandas"
    assert report.fallback_events
