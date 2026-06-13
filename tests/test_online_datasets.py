"""Online real-world dataset tests (cached slices + optional live fetch)."""

from __future__ import annotations

import pytest

import freshdata as fd
from expectations import (
    ALL_ONLINE_FIXTURES,
    ALL_ONLINE_TIER1,
    assert_expectations,
    clean_with_timing,
    load_online_fixture,
)
from golden_util import load_golden, normalize_report, write_golden


@pytest.mark.parametrize("name", ALL_ONLINE_FIXTURES)
@pytest.mark.parametrize("strategy", ["conservative", "balanced", "aggressive"])
def test_online_clean_valid(name, strategy):
    df = load_online_fixture(name)
    cleaned, report, duration = clean_with_timing(df, strategy=strategy)
    assert cleaned is not None
    assert report.rows_after > 0
    assert report.cols_after >= 1
    if name in ALL_ONLINE_TIER1:
        assert_expectations(name, strategy, df, cleaned, report, duration=duration)


@pytest.mark.parametrize("name", ALL_ONLINE_TIER1)
def test_online_compare_clean_matrix(name):
    df = load_online_fixture(name)
    table = fd.compare_clean(df)
    assert set(table["strategy"]) >= {"conservative", "balanced", "aggressive"}
    bal = table.loc[table["strategy"] == "balanced"].iloc[0]
    agg = table.loc[table["strategy"] == "aggressive"].iloc[0]
    assert bal["missing_after"] >= agg["missing_after"], table.to_string()
    assert bal["columns_dropped"] <= agg["columns_dropped"], table.to_string()


@pytest.mark.tier1
@pytest.mark.parametrize("name", ALL_ONLINE_TIER1)
def test_online_balanced_report_golden_snapshot(name, update_golden):
    df = load_online_fixture(name)
    _, report = fd.clean(df, return_report=True, verbose=False)
    actual = normalize_report(report)

    if update_golden:
        path = write_golden(name, report, strategy="balanced", online=True)
        pytest.skip(f"updated online golden snapshot: {path}")

    expected = load_golden(name, strategy="balanced", online=True)
    assert actual == expected, (
        f"online golden mismatch for {name!r}; "
        "run pytest tests/test_online_datasets.py --update-golden"
    )


@pytest.mark.online
@pytest.mark.tier1
@pytest.mark.parametrize("name", ALL_ONLINE_TIER1)
def test_live_fetch_matches_cache(name):
    cached = load_online_fixture(name)
    live = load_online_fixture(name, live=True)
    assert live.shape == cached.shape, f"{name}: shape {live.shape} != {cached.shape}"
    assert list(live.columns) == list(cached.columns), f"{name}: column mismatch"
