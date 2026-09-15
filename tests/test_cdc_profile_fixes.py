"""Regression tests for ``fd.cdc_profile`` fixes (#325, #326, #327, #328)."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd


def _counts(rep: fd.CDCReport) -> dict[str, int]:
    return {d.kind: d.n_rows for d in rep.defects}


# -- #325: rows with a null key are still ordering-checked ----------------------


def test_null_key_rows_are_checked_for_lateness():
    ts = pd.to_datetime(
        ["2024-01-01 00:10", "2024-01-01 00:00", "2024-01-01 00:20", "2024-01-01 00:00"]
    )
    df = pd.DataFrame({"ts": ts, "k": [None, None, "b", "b"]})
    rep = fd.cdc_profile(df, event_time="ts", key="k", now="2024-01-02")
    counts = _counts(rep)
    assert counts["late"] == 2  # row 1 (null key) and row 3 (key b)
    assert counts["missing_key"] == 2
    missing = next(d for d in rep.defects if d.kind == "missing_key")
    assert missing.level == "warning"
    assert missing.sample_keys == ("0", "1")  # row labels, since the key is null
    # The trust penalty formula is unchanged: missing keys do not add to it.
    assert rep.trust_penalties["cdc"] == 0.0
    assert rep.trust_penalties["ordering"] == 0.5


def test_null_keys_form_their_own_group_when_interleaved():
    ts = pd.to_datetime(
        ["2024-01-01 00:10", "2024-01-01 09:00", "2024-01-01 00:09", "2024-01-01 09:01"]
    )
    # NaN and None are both null and share one group; key "a" is in order.
    df = pd.DataFrame({"ts": ts, "k": [None, "a", float("nan"), "a"]})
    rep = fd.cdc_profile(df, event_time="ts", key="k", lateness="5m", now="2024-01-02")
    counts = _counts(rep)
    assert counts.get("out_of_order") == 1
    assert "late" not in counts
    assert counts["missing_key"] == 2


def test_missing_key_alone_does_not_fail_the_gate():
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01"]), "k": ["a", None]}
    )
    rep = fd.cdc_profile(df, event_time="ts", key="k", now="2024-01-01 00:02")
    assert _counts(rep) == {"missing_key": 1}
    assert rep.passed


def test_missing_key_reported_with_explicit_watermark():
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01"]), "k": [None, "a"]}
    )
    rep = fd.cdc_profile(
        df, event_time="ts", key="k", watermark="2024-01-01 00:01", now="2024-01-01 00:02"
    )
    assert _counts(rep) == {"late": 1, "missing_key": 1}


def test_no_missing_key_defect_without_nulls_or_key():
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01 00:00"]), "k": ["a"]})
    assert "missing_key" not in _counts(
        fd.cdc_profile(df, event_time="ts", key="k", now="2024-01-02")
    )
    assert "missing_key" not in _counts(fd.cdc_profile(df, event_time="ts", now="2024-01-02"))


# -- #326: zero stale_after -----------------------------------------------------


@pytest.mark.parametrize("stale_after", ["0s", 0, pd.Timedelta(0)])
def test_zero_stale_after_marks_any_positive_age_stale(stale_after):
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"])})
    rep = fd.cdc_profile(df, event_time="ts", now="2024-01-02", stale_after=stale_after)
    assert "stale" in _counts(rep)
    assert rep.trust_penalties["freshness"] == 1.0


def test_zero_stale_after_with_zero_age_is_not_stale():
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-02"])})
    rep = fd.cdc_profile(df, event_time="ts", now="2024-01-02", stale_after="0s")
    assert rep.freshness_seconds == 0.0
    assert "stale" not in _counts(rep)
    assert rep.trust_penalties["freshness"] == 0.0


def test_negative_stale_after_raises():
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"])})
    with pytest.raises(ValueError, match="stale_after"):
        fd.cdc_profile(df, event_time="ts", now="2024-01-02", stale_after="-1s")


# -- #327: integer epoch event times --------------------------------------------


def test_epoch_milliseconds_are_inferred():
    df = pd.DataFrame({"ts_ms": [1704067200000, 1704067260000], "k": ["a", "b"]})
    rep = fd.cdc_profile(df, event_time="ts_ms", key="k", now="2024-01-01 00:05", stale_after="1h")
    assert rep.freshness_seconds == pytest.approx(240.0)
    assert rep.defects == []


@pytest.mark.parametrize(
    ("values", "unit"),
    [
        ([1704067200, 1704067260], "s"),
        ([1704067200000, 1704067260000], "ms"),
        ([1704067200000000, 1704067260000000], "us"),
        ([1704067200000000000, 1704067260000000000], "ns"),
    ],
)
def test_every_epoch_unit_inferred_and_explicit(values, unit):
    df = pd.DataFrame({"ts": values})
    for event_time_unit in (None, unit):
        rep = fd.cdc_profile(
            df, event_time="ts", now="2024-01-01 00:05", event_time_unit=event_time_unit
        )
        assert rep.freshness_seconds == pytest.approx(240.0)


def test_float_epoch_seconds_and_nullable_ints():
    df = pd.DataFrame({"ts": [1704067200.5, 1704067260.0]})
    rep = fd.cdc_profile(df, event_time="ts", now="2024-01-01 00:05")
    assert rep.freshness_seconds == pytest.approx(240.0)

    df = pd.DataFrame({"ts": pd.array([1704067200000, None], dtype="Int64")})
    rep = fd.cdc_profile(df, event_time="ts", now="2024-01-01 00:05")
    assert rep.freshness_seconds == pytest.approx(300.0)
    assert _counts(rep) == {"missing_event_time": 1}


def test_explicit_event_time_unit_overrides_inference():
    # Millisecond magnitudes, but the caller says they are microseconds.
    df = pd.DataFrame({"ts": [1704067200000]})
    rep = fd.cdc_profile(df, event_time="ts", now="1970-01-20 18:00", event_time_unit="us")
    assert rep.freshness_seconds == pytest.approx(
        (pd.Timestamp("1970-01-20 18:00") - pd.Timestamp(1704067200000, unit="us")).total_seconds()
    )


def test_event_time_unit_ignored_for_string_columns():
    df = pd.DataFrame({"ts": ["2024-01-01 00:01"]})
    rep = fd.cdc_profile(df, event_time="ts", now="2024-01-01 00:05", event_time_unit="ms")
    assert rep.freshness_seconds == pytest.approx(240.0)


def test_invalid_event_time_unit_raises():
    df = pd.DataFrame({"ts": [1704067200000]})
    with pytest.raises(ValueError, match="event_time_unit"):
        fd.cdc_profile(df, event_time="ts", event_time_unit="minutes")


# -- numbers that are not plausible epochs --------------------------------------


def test_row_numbers_are_not_read_as_epoch_seconds():
    df = pd.DataFrame({"id": range(1, 101), "row_no": range(1, 101), "v": 1.0})
    rep = fd.cdc_profile(df, event_time="row_no", now="2026-09-15", stale_after="1h")
    assert not rep.passed
    assert rep.freshness_seconds is None
    assert _counts(rep) == {"event_time_implausible": 100}
    (defect,) = rep.defects
    assert defect.level == "error"
    assert defect.details == {"inferred_unit": "s", "median": 50.5, "plausible_from": "1990-01-01"}
    assert "before 1990-01-01" in defect.rationale
    assert "event_time_unit=" in defect.rationale
    assert "freshness:" not in rep.summary()
    assert rep.to_dict()["defects"][0]["kind"] == "event_time_implausible"


def test_implausible_event_time_skips_time_checks_but_keeps_key_checks():
    # Out of order, with a null and a duplicate change, as nullable integers.
    df = pd.DataFrame(
        {"seq": pd.array([3, 1, 2, 2, None], dtype="Int64"), "k": ["a", "a", "b", "b", "c"]}
    )
    for watermark in (None, "2026-01-01"):
        rep = fd.cdc_profile(df, event_time="seq", key="k", watermark=watermark, now="2026-01-02")
        assert _counts(rep) == {
            "missing_event_time": 1,
            "event_time_implausible": 4,
            "duplicate_key": 2,
            "replay_risk": 2,
        }
        assert rep.freshness_seconds is None


@pytest.mark.parametrize("scale", [1, 10**3, 10**6, 10**9])
@pytest.mark.parametrize(("seconds", "implausible"), [(631152000, False), (631151999, True)])
def test_implausible_epoch_boundary_is_1990_in_every_unit(scale, seconds, implausible):
    df = pd.DataFrame({"ts": [seconds * scale]})
    rep = fd.cdc_profile(df, event_time="ts", now="1990-01-02")
    assert ("event_time_implausible" in _counts(rep)) is implausible
    assert (rep.freshness_seconds is None) is implausible


@pytest.mark.parametrize("values", [[0, 0], [-86400, -3600], [0.5, 1.5]])
def test_zero_negative_and_float_offsets_are_implausible(values):
    rep = fd.cdc_profile(pd.DataFrame({"ts": values}), event_time="ts", now="2026-01-01")
    assert _counts(rep) == {"event_time_implausible": 2}


def test_real_epoch_seconds_column_is_unchanged():
    df = pd.DataFrame({"ts": [1704067200 + 60 * i for i in range(100)]})
    rep = fd.cdc_profile(df, event_time="ts", now="2024-01-01 01:40", stale_after="1h")
    assert rep.passed
    assert rep.defects == []
    assert rep.freshness_seconds == pytest.approx(60.0)


def test_explicit_event_time_unit_trusts_small_numbers():
    df = pd.DataFrame({"row_no": range(1, 101)})
    rep = fd.cdc_profile(df, event_time="row_no", now="1970-01-01 00:02", event_time_unit="s")
    assert rep.passed
    assert rep.defects == []
    assert rep.freshness_seconds == pytest.approx(20.0)


# -- #328: replay_risk needs a duplicate key ------------------------------------


def _late_only_frame() -> pd.DataFrame:
    ts = ["2024-01-01 10:00"] + [f"2024-01-01 00:0{i}" for i in range(9)]
    return pd.DataFrame({"ts": pd.to_datetime(ts), "k": ["a"] * 10})


def test_late_only_batch_never_raises_replay_risk():
    rep = fd.cdc_profile(
        _late_only_frame(),
        event_time="ts",
        key="k",
        now="2024-01-02",
        lateness="1m",
        replay_threshold=0.2,
    )
    assert _counts(rep) == {"late": 9}


def test_replay_risk_fires_once_a_duplicate_key_is_present():
    df = _late_only_frame()
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)  # duplicate one late row
    rep = fd.cdc_profile(
        df, event_time="ts", key="k", now="2024-01-02", lateness="1m", replay_threshold=0.2
    )
    counts = _counts(rep)
    assert counts["duplicate_key"] == 2
    assert counts["replay_risk"] == counts["duplicate_key"] + counts["late"]
