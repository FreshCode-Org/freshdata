"""Unit tests for the shared native-step math helpers and report metadata."""

from __future__ import annotations

import math

from freshdata.config import CleanConfig
from freshdata.execution._native_steps import (
    impute_defined_for,
    integer_safe_bounds,
    iqr_bounds,
    native_outlier_factor,
    native_outlier_method,
    outlier_label,
    resolve_impute_strategy,
    zscore_bounds,
)
from freshdata.report import CleanReport


def test_resolve_impute_strategy_auto():
    assert resolve_impute_strategy("auto", is_numeric=True) == "median"
    assert resolve_impute_strategy("auto", is_numeric=False) == "mode"
    assert resolve_impute_strategy("mean", is_numeric=True) == "mean"


def test_impute_defined_for():
    assert impute_defined_for("mean", is_numeric=True)
    assert not impute_defined_for("median", is_numeric=False)
    assert impute_defined_for("mode", is_numeric=False)


def test_native_outlier_method_and_factor():
    assert native_outlier_method(CleanConfig(outlier_method="iqr")) == "iqr"
    assert native_outlier_method(CleanConfig(outlier_method="zscore")) == "zscore"
    # non-native methods resolve to iqr for the native path
    assert native_outlier_method(CleanConfig(outlier_method="isolation_forest")) == "iqr"
    assert native_outlier_factor(CleanConfig(outlier_method="iqr"), "iqr") == 1.5
    assert native_outlier_factor(CleanConfig(outlier_factor=2.0), "iqr") == 2.0


def test_iqr_and_zscore_bounds():
    assert iqr_bounds(10.0, 20.0, 1.5) == (-5.0, 35.0)
    assert iqr_bounds(5.0, 5.0, 1.5) is None  # constant
    assert iqr_bounds(0.0, float("nan"), 1.5) is None
    lo, hi = zscore_bounds(0.0, 2.0, 3.0)
    assert (lo, hi) == (-6.0, 6.0)
    assert zscore_bounds(1.0, 0.0, 3.0) is None  # zero std


def test_integer_safe_bounds_and_label():
    assert integer_safe_bounds(1.2, 8.9, is_integer=True) == (1.0, 9.0)
    assert integer_safe_bounds(1.2, 8.9, is_integer=False) == (1.2, 8.9)
    assert outlier_label("iqr", 1.5) == "iqr, factor 1.5"


def test_report_records_fallback_and_differences():
    rep = CleanReport(backend="spark")
    rep.record_fallback("spark", "impute", "advanced imputation")
    rep.record_backend_difference("spark", "outliers", "approx quantile", column="x")
    payload = rep.to_dict()
    assert payload["backend"] == "spark"
    assert payload["fallback_events"][0] == {
        "backend": "spark", "fallback_step": "impute",
        "fallback_reason": "advanced imputation",
    }
    diff = payload["backend_differences"][0]
    assert diff["backend"] == "spark" and diff["column"] == "x"
    assert not math.isnan(0.0)  # sanity
