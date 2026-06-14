"""Performance benchmarks: balanced vs aggressive on fixture scale."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import freshdata as fd
from expectations import (
    ALL_FIXTURES,
    ALL_ONLINE_FIXTURES,
    clean_with_timing,
    load_expectations,
    load_fixture,
    load_online_fixture,
)
from freshdata.engine.cache import build_engine_cache

PERF_DIR = Path(__file__).parent / "fixtures" / "perf"
BASELINES = json.loads((PERF_DIR / "baselines.json").read_text())
REGRESSION_TOLERANCE = float(
    os.getenv("FRESHDATA_PERF_TOLERANCE", "3.0")
)  # fallback to 3.0 locally


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES + ALL_ONLINE_FIXTURES)
@pytest.mark.parametrize("strategy", ["balanced", "aggressive"])
def test_clean_duration_within_baselines(fixture_name, strategy):
    if fixture_name in ALL_ONLINE_FIXTURES:
        df = load_online_fixture(fixture_name)
    else:
        df = load_fixture(fixture_name)
    _, _, duration = clean_with_timing(df, strategy=strategy)
    exp = load_expectations(fixture_name).get(strategy, {})
    baseline = BASELINES.get(fixture_name, {}).get(strategy, {})
    limit = exp.get("max_duration_seconds") or baseline.get("max_seconds", 10.0)
    assert duration <= limit * REGRESSION_TOLERANCE, (
        f"{fixture_name}/{strategy} took {duration:.2f}s (limit {limit}s)"
    )
    min_rps = baseline.get("min_rows_per_sec")
    if min_rps and duration > 0 and len(df) >= 1000:
        rps = len(df) / duration
        assert rps >= min_rps / REGRESSION_TOLERANCE, (
            f"{fixture_name}/{strategy} {rps:.0f} rows/s < {min_rps}"
        )


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES + ALL_ONLINE_FIXTURES)
def test_balanced_not_slower_than_aggressive(fixture_name):
    if fixture_name in ALL_ONLINE_FIXTURES:
        df = load_online_fixture(fixture_name)
    else:
        df = load_fixture(fixture_name)
    _, _, t_balanced = clean_with_timing(df, strategy="balanced")
    _, _, t_aggressive = clean_with_timing(df, strategy="aggressive")
    assert t_balanced <= max(t_aggressive * 3.0, 1.0), (
        f"{fixture_name}: balanced {t_balanced:.2f}s vs aggressive {t_aggressive:.2f}s"
    )


def test_build_engine_cache_called_once_per_clean():
    df = load_fixture("aqi_sample")
    with patch("freshdata.cleaner.build_engine_cache", wraps=build_engine_cache) as mock:
        fd.clean(df, verbose=False)
    assert mock.call_count == 1


@pytest.mark.large
def test_aqi_full_file_benchmark():
    path = os.environ.get("FRESHDATA_AQI_PATH", "/Users/kevincostner/AQI.csv")
    if not Path(path).exists():
        pytest.skip(f"large AQI file not found at {path}")
    df = __import__("pandas").read_csv(path)
    _, _, duration = clean_with_timing(df, strategy="balanced")
    assert duration < 5.0, f"balanced clean on {len(df)} rows took {duration:.1f}s"
    rps = len(df) / duration
    assert rps > 5000, f"only {rps:.0f} rows/s on full AQI"
