"""Duplicate rules: keep strategies, ratio warnings, time-series protection."""

import pandas as pd

import freshdata as fd

QUIET = {"verbose": False}


def test_exact_duplicates_removed_and_percentage_reported():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    out, report = fd.clean(df, report=True, **QUIET)
    assert len(out) == 3
    assert report.duplicates_removed == 1
    [action] = [a for a in report if a.step == "drop_duplicates"]
    assert "%" in action.description  # percentage always reported


def test_duplicate_ratio_above_threshold_warns():
    df = pd.DataFrame({"a": [1, 2] * 10})  # 90% duplicates
    _, report = fd.clean(df, report=True, **QUIET)
    assert any("duplicate" in w for w in report.warnings)
    assert report.recommendations


def test_keep_last():
    df = pd.DataFrame({"id": [1, 1, 2], "v": ["old", "new", "z"]})
    out = fd.clean(df, duplicate_subset=("id",), duplicate_keep="last", **QUIET)
    assert out["v"].tolist() == ["new", "z"]


def test_keep_drop_removes_all_members():
    df = pd.DataFrame({"id": [1, 1, 2], "v": ["a", "b", "c"]})
    out = fd.clean(df, duplicate_subset=("id",), duplicate_keep="drop", **QUIET)
    assert out["v"].tolist() == ["c"]


def test_keep_aggregate_means_numerics_and_keeps_first_text():
    df = pd.DataFrame({"id": [1, 1, 2], "amount": [10.0, 20.0, 5.0],
                       "note": ["first", "second", "only"]})
    out = fd.clean(df, duplicate_subset=("id",), duplicate_keep="aggregate",
                   **QUIET)
    assert len(out) == 2
    row = out.loc[out["id"] == 1].iloc[0]
    assert row["amount"] == 15.0
    assert row["note"] == "first"


def test_timeseries_duplicates_preserved_by_default():
    idx = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"v": [1, 1, 2]}, index=idx)
    out, report = fd.clean(df, report=True, **QUIET)
    assert len(out) == 3  # nothing removed
    assert any("time-indexed" in w for w in report.warnings)


def test_timeseries_duplicates_removed_when_allowed():
    idx = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"v": [1, 1, 2]}, index=idx)
    out = fd.clean(df, allow_timeseries_duplicates=True, **QUIET)
    assert len(out) == 2
