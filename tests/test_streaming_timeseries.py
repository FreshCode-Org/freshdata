"""Behavioural tests for time-series / streaming-aware cleaning modes.

Covers short-gap interpolation vs. long-gap preservation, seasonal imputation,
watermark-based late-data handling, ordered dedupe, and windowed anomaly detection —
through both ``fd.clean_timeseries`` and a multi-batch ``StreamingCleaner``.
"""

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata import StreamingCleaner, TimeSeriesCleanConfig


# -- config validation ---------------------------------------------------------

def test_config_requires_timestamp_and_validates_choices():
    with pytest.raises(ValueError):
        TimeSeriesCleanConfig(timestamp_column="")
    with pytest.raises(ValueError):
        TimeSeriesCleanConfig(timestamp_column="t", interpolation_method="cubic")
    with pytest.raises(ValueError):
        TimeSeriesCleanConfig(timestamp_column="t", late_data_action="explode")
    with pytest.raises(ValueError):
        TimeSeriesCleanConfig(timestamp_column="t", ordered_dedupe_keys=("k",),
                              ordered_dedupe_keep="highest_quality")  # no quality_column


def test_resolved_event_time_column_falls_back_to_timestamp():
    assert TimeSeriesCleanConfig(timestamp_column="t").resolved_event_time_column == "t"
    cfg = TimeSeriesCleanConfig(timestamp_column="t", watermark_column="evt")
    assert cfg.resolved_event_time_column == "evt"


# -- interpolation -------------------------------------------------------------

def test_short_gap_filled_long_gap_preserved():
    t = pd.date_range("2024-01-01", periods=8, freq="h")
    v = [1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 8.0]
    df = pd.DataFrame({"t": t, "v": v})
    out, report = fd.clean_timeseries(
        df, timestamp_column="t", max_interpolation_gap=1, return_report=True)
    # The single-step gap at index 1 is filled; the 3-step gap (3..5) stays missing.
    assert out["v"].iloc[1] == pytest.approx(2.0)
    assert out["v"].iloc[3:6].isna().all()
    steps = [a.step for a in report]
    assert "timeseries_interpolation" in steps


def test_interpolation_per_entity_does_not_bleed_across_groups():
    df = pd.DataFrame({
        "id": ["a", "a", "a", "b", "b", "b"],
        "t": list(pd.date_range("2024-01-01", periods=3, freq="h")) * 2,
        "v": [1.0, np.nan, 3.0, 100.0, np.nan, 300.0],
    })
    out = fd.clean_timeseries(
        df, timestamp_column="t", entity_id_columns=("id",), max_interpolation_gap=1)
    a = out[out["id"] == "a"]["v"].tolist()
    b = out[out["id"] == "b"]["v"].tolist()
    assert a == pytest.approx([1.0, 2.0, 3.0])
    assert b == pytest.approx([100.0, 200.0, 300.0])


# -- seasonal imputation -------------------------------------------------------

def test_seasonal_imputation_uses_matching_season():
    idx = pd.date_range("2024-01-01", periods=96, freq="h")  # 4 days, hourly
    values = 10.0 + 5.0 * np.sin(2 * np.pi * idx.hour / 24)
    series = pd.Series(values, dtype="float64")
    # Knock out one midnight (hour == 0) sample; others at hour 0 ≈ 10.0.
    target = 72  # day-3 midnight
    assert idx[target].hour == 0
    series.iloc[target] = np.nan
    df = pd.DataFrame({"t": idx, "v": series.to_numpy()})

    out, report = fd.clean_timeseries(
        df, timestamp_column="t", max_interpolation_gap=0,
        seasonal_period="hour", seasonal_imputation_enabled=True, return_report=True)
    assert out["v"].iloc[target] == pytest.approx(10.0, abs=0.5)
    seasonal = [a for a in report if a.step == "seasonal_imputation"]
    assert seasonal and seasonal[0].count == 1
    assert "season" in seasonal[0].rationale


# -- ordered dedupe ------------------------------------------------------------

def test_ordered_dedupe_keeps_latest_event_time():
    df = pd.DataFrame({
        "id": [1, 1, 2],
        "t": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00", "2024-01-01 00:00"]),
        "val": [10.0, 99.0, 5.0],
    })
    out, report = fd.clean_timeseries(
        df, timestamp_column="t", ordered_dedupe_keys=("id",),
        ordered_dedupe_keep="latest_event_time", return_report=True)
    assert len(out) == 2
    assert out.loc[out["id"] == 1, "val"].tolist() == [99.0]
    assert any(a.step == "ordered_dedupe" for a in report)


def test_ordered_dedupe_is_deterministic():
    df = pd.DataFrame({
        "id": [1, 1, 1],
        "t": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 00:30"]),
        "val": [1.0, 2.0, 3.0],
    })
    runs = [
        fd.clean_timeseries(df, timestamp_column="t", ordered_dedupe_keys=("id",))
        ["val"].tolist()
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2] == [2.0]  # latest event time wins, every time


def test_ordered_dedupe_highest_quality():
    df = pd.DataFrame({
        "id": [1, 1],
        "t": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01"]),
        "q": [0.9, 0.2],
        "val": [10.0, 20.0],
    })
    out = fd.clean_timeseries(
        df, timestamp_column="t", ordered_dedupe_keys=("id",),
        ordered_dedupe_keep="highest_quality", quality_column="q")
    assert out["val"].tolist() == [10.0]


# -- watermark / late data -----------------------------------------------------

def _late_config(**kw):
    base = dict(timestamp_column="t", entity_id_columns=("id",), event_time_column="t",
                allowed_lateness="2m")
    base.update(kw)
    return TimeSeriesCleanConfig(**base)


def test_late_events_quarantined_after_watermark():
    df = pd.DataFrame({
        "id": [1, 1, 1, 1],
        "t": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05",
                             "2024-01-01 00:10", "2024-01-01 00:01"]),
        "v": [1.0, 2.0, 3.0, 9.0],
    })
    out, report, exceptions = fd.clean_timeseries(
        df, time_series_config=_late_config(late_data_action="quarantine"),
        return_report=True, return_exceptions=True)
    assert len(out) == 3
    assert len(exceptions) == 1
    assert exceptions["v"].tolist() == [9.0]
    assert exceptions["_quarantine_reason"].tolist() == ["late_data"]
    assert any(a.step == "late_data" for a in report)


def test_late_events_dropped_when_configured():
    df = pd.DataFrame({
        "id": [1, 1, 1],
        "t": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-01 00:01"]),
        "v": [1.0, 2.0, 9.0],
    })
    out, exceptions = fd.clean_timeseries(
        df, time_series_config=_late_config(late_data_action="drop"),
        return_exceptions=True)
    assert len(out) == 2
    assert len(exceptions) == 0  # dropped rows are not quarantined


def test_late_events_kept_with_warning():
    df = pd.DataFrame({
        "id": [1, 1, 1],
        "t": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-01 00:01"]),
        "v": [1.0, 2.0, 9.0],
    })
    out, report = fd.clean_timeseries(
        df, time_series_config=_late_config(late_data_action="keep_with_warning"),
        return_report=True)
    assert len(out) == 3
    assert any("late" in w for w in report.warnings)


def test_watermark_persists_across_streaming_batches():
    cfg = _late_config(late_data_action="quarantine")
    cleaner = StreamingCleaner(time_series_config=cfg, warmup_batches=0)
    b1 = pd.DataFrame({"id": [1, 1], "v": [1.0, 2.0],
                       "t": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:10"])})
    # b2's second row predates the watermark established by b1 → late across batches.
    b2 = pd.DataFrame({"id": [1, 1], "v": [3.0, 9.0],
                       "t": pd.to_datetime(["2024-01-01 00:11", "2024-01-01 00:00"])})
    cleaner.clean_batch(b1)
    cleaned2, report2 = cleaner.clean_batch(b2)
    assert len(cleaned2) == 1
    assert len(cleaner.exceptions_) == 1
    final = cleaner.finalize()
    assert final.streaming["time_series"]["late_quarantined_total"] == 1


# -- windowed anomaly ----------------------------------------------------------

def test_windowed_anomaly_flags_without_dropping():
    rng = np.random.default_rng(0)
    x = rng.normal(50.0, 1.0, 200)
    x[100] = 500.0
    df = pd.DataFrame({"t": pd.date_range("2024", periods=200, freq="min"), "x": x})
    out, report = fd.clean_timeseries(
        df, timestamp_column="t", anomaly_window_size=20,
        anomaly_method="rolling_zscore", anomaly_action="flag", return_report=True)
    assert len(out) == 200  # nothing dropped
    assert "x_anomaly" in out.columns
    assert bool(out["x_anomaly"].iloc[100]) is True
    assert int(out["x_anomaly"].sum()) >= 1
    assert any(a.step == "windowed_anomaly" for a in report)


@pytest.mark.parametrize("method", ["rolling_zscore", "mad", "iqr", "ewma"])
def test_windowed_anomaly_methods_detect_spike(method):
    x = np.full(100, 5.0)
    x[60] = 999.0
    df = pd.DataFrame({"t": pd.date_range("2024", periods=100, freq="min"), "x": x})
    out = fd.clean_timeseries(
        df, timestamp_column="t", anomaly_window_size=15, anomaly_method=method)
    assert bool(out["x_anomaly"].iloc[60]) is True


def test_windowed_anomaly_cap_clips_value():
    x = np.full(80, 5.0)
    x[40] = 999.0
    df = pd.DataFrame({"t": pd.date_range("2024", periods=80, freq="min"), "x": x})
    out = fd.clean_timeseries(
        df, timestamp_column="t", anomaly_window_size=15, anomaly_action="cap")
    assert out["x"].iloc[40] < 999.0
    assert bool(out["x_anomaly"].iloc[40]) is True


def test_windowed_anomaly_quarantine_removes_rows():
    x = np.full(80, 5.0)
    x[40] = 999.0
    df = pd.DataFrame({"t": pd.date_range("2024", periods=80, freq="min"), "x": x})
    out, exceptions = fd.clean_timeseries(
        df, timestamp_column="t", anomaly_window_size=15,
        anomaly_action="quarantine", return_exceptions=True)
    assert len(out) == 79
    assert len(exceptions) == 1
    assert exceptions["_quarantine_reason"].tolist() == ["windowed_anomaly"]


# -- protected columns / report contract --------------------------------------

def test_ids_and_targets_are_not_interpolated():
    df = pd.DataFrame({
        "t": pd.date_range("2024", periods=4, freq="h"),
        "user_id": [1.0, np.nan, 3.0, 4.0],   # name → id role, never interpolated
        "v": [1.0, np.nan, 3.0, 4.0],
    })
    out = fd.clean_timeseries(df, timestamp_column="t", max_interpolation_gap=1)
    assert pd.isna(out.loc[out.index[1], "user_id"])  # id gap preserved
    assert out["v"].iloc[1] == pytest.approx(2.0)     # numeric gap interpolated


def test_report_streaming_contract_preserved_in_timeseries_mode():
    df = pd.DataFrame({"t": pd.date_range("2024", periods=4, freq="h"),
                       "v": [1.0, np.nan, 3.0, 4.0]})
    _, report = fd.clean_timeseries(df, timestamp_column="t", return_report=True)
    # Existing streaming keys must still be present and unchanged in shape.
    for key in ("batch_id", "rows_in_batch", "batch_trust_score", "rolling_trust_score",
                "cumulative_trust_score", "warmup_phase", "trust_gate_passed"):
        assert key in report.streaming
    assert "time_series" in report.streaming


# -- additional config / option coverage --------------------------------------

def test_config_rejects_out_of_range_numbers():
    with pytest.raises(ValueError):
        TimeSeriesCleanConfig(timestamp_column="t", max_interpolation_gap=-1)
    with pytest.raises(ValueError):
        TimeSeriesCleanConfig(timestamp_column="t", anomaly_window_size=-1)
    with pytest.raises(ValueError):
        TimeSeriesCleanConfig(timestamp_column="t", anomaly_threshold=0.0)


def test_to_timedelta_accepts_number_and_timedelta():
    from freshdata.streaming._timeseries import to_timedelta

    assert to_timedelta(None) is None
    assert to_timedelta(90) == pd.Timedelta(seconds=90)
    assert to_timedelta(pd.Timedelta(minutes=5)) == pd.Timedelta(minutes=5)
    assert to_timedelta("10m") == pd.Timedelta(minutes=10)


@pytest.mark.parametrize("keep,expected", [("first", 10.0), ("last", 99.0)])
def test_ordered_dedupe_first_and_last(keep, expected):
    df = pd.DataFrame({
        "id": [1, 1],
        "t": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"]),
        "val": [10.0, 99.0],
    })
    out = fd.clean_timeseries(
        df, timestamp_column="t", ordered_dedupe_keys=("id",), ordered_dedupe_keep=keep)
    assert out["val"].tolist() == [expected]


@pytest.mark.parametrize("method,expected", [("ffill", 1.0), ("bfill", 3.0)])
def test_interpolation_ffill_bfill(method, expected):
    df = pd.DataFrame({
        "t": pd.date_range("2024-01-01", periods=3, freq="h"),
        "v": [1.0, np.nan, 3.0],
    })
    out = fd.clean_timeseries(
        df, timestamp_column="t", max_interpolation_gap=1, interpolation_method=method)
    assert out["v"].iloc[1] == pytest.approx(expected)


def test_seasonal_falls_back_to_rolling_median_when_season_sparse():
    # Only two same-hour observations exist, so the season bucket is too sparse
    # (needs >= 3) and imputation falls back to a local rolling median.
    t = pd.date_range("2024-01-01", periods=6, freq="h")
    df = pd.DataFrame({"t": t, "v": [10.0, 11.0, np.nan, 13.0, 14.0, 15.0]})
    out, report = fd.clean_timeseries(
        df, timestamp_column="t", max_interpolation_gap=0,
        seasonal_period="hour", seasonal_imputation_enabled=True, return_report=True)
    assert not out["v"].isna().any()
    assert any(a.step == "seasonal_imputation" for a in report)


def test_frequency_recorded_in_summary():
    df = pd.DataFrame({"t": pd.date_range("2024", periods=3, freq="h"),
                       "v": [1.0, 2.0, 3.0]})
    _, report = fd.clean_timeseries(
        df, time_series_config=TimeSeriesCleanConfig(timestamp_column="t", frequency="h"),
        return_report=True)
    assert report.streaming["time_series"]["frequency"] == "h"


def test_no_anomalies_still_emits_flag_column():
    df = pd.DataFrame({"t": pd.date_range("2024", periods=30, freq="min"),
                       "x": np.full(30, 5.0)})
    out = fd.clean_timeseries(
        df, timestamp_column="t", anomaly_window_size=10, anomaly_method="rolling_zscore")
    assert "x_anomaly" in out.columns
    assert not out["x_anomaly"].any()
