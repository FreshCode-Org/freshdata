"""Tests for the Dagster integration (using a fake ``dagster`` module)."""

from __future__ import annotations

from freshdata.integrations.dagster import freshdata_asset_check


def test_asset_check_passes_warn_severity(fake_dagster, fake_asset, sample_df):
    check = freshdata_asset_check(asset=fake_asset, trust_score_threshold=0.0)
    result = check(orders=sample_df)
    assert result.passed is True
    assert result.severity == "WARN"
    assert result.metadata["freshdata/passed"] == ("bool", True)


def test_asset_check_failed_warn_severity(fake_dagster, fake_asset, sample_df):
    check = freshdata_asset_check(
        asset=fake_asset, trust_score_threshold=999.0, on_low_score="warn"
    )
    result = check(orders=sample_df)
    assert result.passed is False
    assert result.severity == "WARN"


def test_asset_check_failed_error_severity(fake_dagster, fake_asset, sample_df):
    check = freshdata_asset_check(
        asset=fake_asset, trust_score_threshold=999.0, on_low_score="fail"
    )
    result = check(orders=sample_df)
    assert result.passed is False
    assert result.severity == "ERROR"


def test_asset_check_publishes_report_metadata(fake_dagster, fake_asset, sample_df):
    check = freshdata_asset_check(
        asset=fake_asset, trust_score_threshold=0.0, publish_full_report=True
    )
    result = check(orders=sample_df)
    assert result.metadata["freshdata/report"][0] == "json"


def test_resource_gate(fake_dagster, sample_df):
    from freshdata.integrations.dagster import FreshDataResource

    resource = FreshDataResource(trust_score_threshold=0.0)
    df, result = resource.gate(sample_df)
    assert result.passed is True
    assert len(df) == result.row_count_out
