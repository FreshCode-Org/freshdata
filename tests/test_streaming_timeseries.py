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
from freshdata.report import CleanReport
from freshdata.streaming._timeseries import TimeSeriesProcessor, coerce_datetimes, to_timedelta

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
    base = {"timestamp_column": "t", "entity_id_columns": ("id",),
            "event_time_column": "t", "allowed_lateness": "2m"}
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


def test_integer_column_labels_in_timeseries_mode():
    df = pd.DataFrame({"t": pd.date_range("2024", periods=4, freq="h"),
                       7: ["a", None, "a", "a"],
                       8: [1.0, np.nan, 3.0, 4.0]})
    out, report = fd.clean_timeseries(df, timestamp_column="t", max_interpolation_gap=1,
                                      return_report=True)
    assert 7 in out.columns and 8 in out.columns
    assert out[8].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert any(a.step == "timeseries_interpolation" and a.column == "8" for a in report)


# -- timestamp parsing (#227, #250) ---------------------------------------------

def _ts_clean(df, **kw):
    cfg = TimeSeriesCleanConfig(timestamp_column="ts", **kw)
    return StreamingCleaner(time_series_config=cfg, verbose=False).clean_batch(df)


@pytest.mark.parametrize("scale,unit", [(1, "s"), (10**3, "ms"), (10**6, "us"), (10**9, "ns")])
def test_integer_epoch_timestamps_infer_unit(scale, unit):
    epoch = [(1704067200 + 3600 * i) * scale for i in range(4)]
    df = pd.DataFrame({"ts": epoch, "v": [1.0, 2.0, 3.0, 4.0]})
    out, report = _ts_clean(df)
    assert out["ts"].tolist() == list(pd.date_range("2024-01-01", periods=4, freq="h"))
    parse = [a for a in report.actions if a.step == "timeseries_timestamp_parse"]
    assert [a.column for a in parse] == ["ts"]
    assert f"'{unit}'" in parse[0].description
    assert not report.coerced_cells


def test_float_and_nullable_epoch_timestamps_keep_rows():
    for values in ([1704067200.0, np.nan, 1704074400.0],
                   pd.array([1704067200, None, 1704074400], dtype="Int64")):
        df = pd.DataFrame({"ts": values, "v": [1.0, 2.0, 3.0]})
        out, report = _ts_clean(df)
        assert len(out) == 3
        assert out["ts"].iloc[0] == pd.Timestamp("2024-01-01 00:00")
        assert not report.coerced_cells  # a missing epoch is not a lost value


def test_explicit_timestamp_unit_overrides_inference():
    df = pd.DataFrame({"ts": [1704067200, 1704067260], "v": [1.0, 2.0]})
    out, _ = _ts_clean(df, timestamp_unit="ms")
    assert out["ts"].iloc[0] == pd.Timestamp(1704067200, unit="ms")
    with pytest.raises(ValueError, match="timestamp_unit"):
        TimeSeriesCleanConfig(timestamp_column="ts", timestamp_unit="minutes")


def test_unparseable_timestamps_are_reported_and_rows_kept():
    df = pd.DataFrame({"ts": ["2024-01-01", "2024-13-45", "2024-01-03"],
                       "v": [1.0, 2.0, 3.0]})
    out, report = _ts_clean(df)
    assert len(out) == 3
    assert out["ts"].isna().sum() == 1
    assert report.coerced_cells == {"ts": {1: "2024-13-45"}}
    assert report.coerced_rows["ts"] == (1,)
    assert report.to_dict()["coerced_cells"] == {"ts": {"1": "2024-13-45"}}
    lost = [a for a in report.actions
            if a.step == "timeseries_timestamp_parse" and a.count == 1]
    assert lost and lost[0].risk == "medium"
    assert any("2024-13-45" in w for w in report.warnings)


def test_out_of_range_epoch_is_reported():
    df = pd.DataFrame({"ts": [1704067200, 99_999_999_999], "v": [1.0, 2.0]})
    out, report = _ts_clean(df, timestamp_unit="s")
    assert out["ts"].isna().sum() == 1
    assert list(report.coerced_cells["ts"].values()) == [99_999_999_999]


def test_unparseable_sensitive_timestamp_is_masked_in_report():
    proc =TimeSeriesProcessor(TimeSeriesCleanConfig(timestamp_column="ts"),
                               clean_config=fd.CleanConfig(sensitive_columns=("ts",)))
    report = CleanReport()
    df = pd.DataFrame({"ts": ["2024-01-01", "secret-value"], "v": [1.0, 2.0]})
    proc.process(df, report)
    (masked,) = report.coerced_cells["ts"].values()
    assert masked.startswith("[SENSITIVE:")
    assert not any("secret-value" in w for w in report.warnings)


def test_mixed_utc_offsets_across_dst_are_parsed_to_utc():
    df = pd.DataFrame({"ts": ["2024-03-10T01:00:00-05:00", "2024-03-10T01:30:00-05:00",
                              "2024-03-10T03:00:00-04:00", "2024-03-10T03:30:00-04:00"],
                       "v": [1.0, None, 3.0, 4.0]})
    out, report = _ts_clean(df)
    assert str(out["ts"].dt.tz) == "UTC"
    assert out["ts"].tolist() == list(
        pd.date_range("2024-03-10 06:00", periods=4, freq="30min", tz="UTC"))
    assert out["v"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert not report.coerced_cells


def test_naive_and_offset_aware_timestamps_mix_parses_to_utc():
    df = pd.DataFrame({"ts": ["2024-03-10T01:00:00", "2024-03-10T03:00:00-04:00"],
                       "v": [1.0, 2.0]})
    out, report = _ts_clean(df)
    assert out["ts"].tolist() == [pd.Timestamp("2024-03-10 01:00", tz="UTC"),
                                  pd.Timestamp("2024-03-10 07:00", tz="UTC")]
    assert not report.coerced_cells


def test_coerce_datetimes_keeps_clean_batches_in_their_own_zone():
    naive = pd.Series(["2024-01-01 00:00", "2024-01-02 00:00", None])
    assert coerce_datetimes(naive).dt.tz is None
    one_offset = coerce_datetimes(pd.Series(["2024-01-01T00:00:00+05:30"] * 2))
    assert one_offset.iloc[0] == pd.Timestamp("2023-12-31 18:30", tz="UTC")
    assert str(one_offset.dt.tz) != "UTC"  # a single fixed offset is kept
    garbage = coerce_datetimes(pd.Series(["2024-01-01", "not a date"]))
    assert garbage.dt.tz is None and garbage.isna().tolist() == [False, True]


def test_epoch_event_time_column_drives_late_data():
    df = pd.DataFrame({"ts": pd.date_range("2024", periods=3, freq="h"),
                       "ev": [1704067200, 1704070800, 1704067210], "v": [1.0, 2.0, 3.0]})
    out, report = _ts_clean(df, event_time_column="ev", allowed_lateness="1m")
    assert len(out) == 2
    assert any(a.step == "late_data" and a.count == 1 for a in report.actions)


# -- anomaly config and MAD (#290, #291) ----------------------------------------

def test_anomaly_window_size_one_is_rejected():
    with pytest.raises(ValueError, match="0 or >= 2"):
        TimeSeriesCleanConfig(timestamp_column="ts", anomaly_window_size=1)
    df = pd.DataFrame({"ts": pd.date_range("2024", periods=5, freq="h"),
                       "v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out, _ = _ts_clean(df, anomaly_window_size=2)
    assert "v_anomaly" in out.columns


def test_mad_does_not_flag_minority_values_of_two_valued_series():
    v = [0.0, 0, 0, 1, 0, 0, 1, 0, 0, 0] * 3
    df = pd.DataFrame({"ts": pd.date_range("2024", periods=30, freq="h"), "v": v})
    out, _ = _ts_clean(df, anomaly_window_size=10, anomaly_method="mad",
                       anomaly_action="cap")
    assert int(out["v_anomaly"].sum()) == 0
    assert out["v"].sum() == 6


def test_mad_still_flags_and_caps_spike_in_flat_window():
    x = np.full(40, 5.0)
    x[25] = 999.0
    df = pd.DataFrame({"ts": pd.date_range("2024", periods=40, freq="min"), "x": x})
    out, _ = _ts_clean(df, anomaly_window_size=10, anomaly_method="mad",
                       anomaly_action="cap")
    assert out["x_anomaly"].tolist() == [i == 25 for i in range(40)]
    assert out["x"].iloc[25] == 5.0


# -- anomaly columns when batch dtypes change (#248 part b) ----------------------

def test_anomaly_columns_stable_when_numeric_column_holds_a_string():
    cfg = TimeSeriesCleanConfig(timestamp_column="ts", anomaly_window_size=20,
                                anomaly_method="mad")
    cleaner = StreamingCleaner(time_series_config=cfg, verbose=False)
    b1 = pd.DataFrame({"ts": ["2024-01-01", "2024-01-02", "2024-01-03"],
                       "a": [0, 1, 2], "b": [0, 10, 20]})
    b2 = pd.DataFrame({"ts": ["2024-01-04", "2024-01-05", "2024-01-06"],
                       "a": ["oops", 4, 5], "b": [30, 40, 50]})
    out1, _ = cleaner.clean_batch(b1)
    out2, _ = cleaner.clean_batch(b2)
    assert list(out2.columns) == list(out1.columns) == ["ts", "a", "b", "a_anomaly", "b_anomaly"]
    assert out2["a"].tolist() == ["oops", 4, 5]


def test_non_numeric_cell_is_neither_scored_nor_capped():
    cfg = TimeSeriesCleanConfig(timestamp_column="ts", anomaly_window_size=5,
                                anomaly_method="mad", anomaly_action="cap")
    cleaner = StreamingCleaner(time_series_config=cfg, verbose=False)
    cleaner.clean_batch(pd.DataFrame({"ts": pd.date_range("2024-01", periods=8, freq="h"),
                                      "a": [5.0] * 8}))  # locks 'a' as numeric
    b2 = pd.DataFrame({"ts": pd.date_range("2024-02", periods=8, freq="h"),
                       "a": [5, 5, 5, 5, "oops", 5, 999, 5]})
    out, _ = cleaner.clean_batch(b2)
    assert out["a_anomaly"].tolist() == [i == 6 for i in range(8)]
    assert out["a"].tolist() == [5, 5, 5, 5, "oops", 5, 5.0, 5]
