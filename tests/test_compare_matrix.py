"""Cross-fixture strategy comparison matrix (quality + efficiency)."""

import json

import pytest

import freshdata as fd
from expectations import ALL_FIXTURES, load_fixture, resolve_column


def _matrix(df, fixture_name: str) -> str:
    table = fd.compare_clean(df)
    lines = [f"\n=== compare_clean: {fixture_name} ===", table.to_string(index=False)]
    return "\n".join(lines)


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_compare_clean_all_strategies(fixture_name):
    df = load_fixture(fixture_name)
    table = fd.compare_clean(df)
    assert set(table["strategy"]) >= {"conservative", "balanced", "aggressive"}
    assert (table["duration_seconds"] >= 0).all()
    conservative = table.loc[table["strategy"] == "conservative"].iloc[0]
    assert conservative["missing_after"] >= conservative["missing_before"] - 1


@pytest.mark.parametrize("fixture_name", ["aqi_sample", "large_panel", "wide_sparse"])
def test_balanced_preserves_at_least_as_much_as_aggressive(fixture_name):
    df = load_fixture(fixture_name)
    table = fd.compare_clean(df, strategies=("balanced", "aggressive"))
    bal = table.loc[table["strategy"] == "balanced"].iloc[0]
    agg = table.loc[table["strategy"] == "aggressive"].iloc[0]
    assert bal["missing_after"] >= agg["missing_after"], _matrix(df, fixture_name)
    assert bal["columns_dropped"] <= agg["columns_dropped"], _matrix(df, fixture_name)


def test_aqi_balanced_never_imputes_target():
    df = load_fixture("aqi_sample")
    out = fd.clean(df, strategy="balanced", verbose=False)
    raw = resolve_column(df, "aqi")
    assert out["aqi"].isna().sum() == df[raw].isna().sum()


def test_large_panel_balanced_preserves_aqi():
    df = load_fixture("large_panel")
    out = fd.clean(df, strategy="balanced", verbose=False)
    raw = resolve_column(df, "aqi")
    assert out["aqi"].isna().sum() == df[raw].isna().sum()


def test_duplicate_heavy_drops_duplicates():
    df = load_fixture("duplicate_heavy")
    out, report = fd.clean(df, return_report=True, verbose=False)
    assert len(out) < len(df)
    assert report.duplicates_removed > 0


def test_locale_numbers_not_force_converted():
    df = load_fixture("locale_numbers")
    out = fd.clean(df, strategy="balanced", verbose=False)
    assert str(out["euro_amount"].dtype) in ("object", "string", "str")


def test_mixed_roles_preserves_outcome_score():
    df = load_fixture("mixed_roles")
    out = fd.clean(df, strategy="balanced", verbose=False)
    assert out["outcome_score"].isna().sum() >= df["outcome_score"].isna().sum()


def test_compare_plans_include_metrics():
    df = load_fixture("aqi_sample")
    table = fd.compare_plans(df, include_metrics=True)
    assert "missing_after" in table.columns
    assert "duration_seconds" in table.columns


def test_compare_clean_primary_models_json():
    df = load_fixture("fraud_signals")
    row = fd.compare_clean(df, strategies=("balanced",)).iloc[0]
    models = json.loads(row["primary_models"])
    assert isinstance(models, dict)
