"""Parametrized real-world fixture tests."""

import pytest

import freshdata as fd
from expectations import (
    ALL_FIXTURES,
    assert_expectations,
    clean_with_timing,
    load_fixture,
)


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
@pytest.mark.parametrize("strategy", ["conservative", "balanced", "aggressive"])
def test_clean_produces_valid_output(fixture_name, strategy):
    df = load_fixture(fixture_name)
    out, report, duration = clean_with_timing(df, strategy=strategy)
    assert len(out) >= 0
    assert report.rows_before == len(df)
    assert_expectations(fixture_name, strategy, df, out, report, duration=duration)


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_balanced_preserves_column_count(fixture_name):
    df = load_fixture(fixture_name)
    out = fd.clean(df, strategy="balanced", verbose=False)
    assert out.shape[1] >= df.shape[1] - 1  # may drop all-empty cols in layer 1


def test_wide_sparse_balanced_never_drops_sparse_columns():
    df = load_fixture("wide_sparse")
    out = fd.clean(df, strategy="balanced", verbose=False)
    for col in df.columns:
        if col != "empty":
            assert col.lower().replace(" ", "_") in out.columns or col in out.columns
