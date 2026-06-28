"""Tests for the CDC / event-time quality gate (``fd.cdc_profile``)."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata import CDCReport

NOW = pd.Timestamp("2024-01-01 12:00:00")


def _events(**cols: object) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_clean_stream_passes():
    df = _events(
        entity_id=["a", "a", "b"],
        event_ts=pd.to_datetime(["2024-01-01 11:58", "2024-01-01 11:59", "2024-01-01 11:59"]),
        op=["I", "U", "I"],
    )
    rep = fd.cdc_profile(df, event_time="event_ts", key="entity_id",
                         operation_col="op", now=NOW, stale_after="1h")
    assert isinstance(rep, CDCReport)
    assert rep.passed
    assert rep.n_errors == 0


def test_missing_event_time_is_error():
    df = _events(entity_id=["a", "b"], event_ts=["2024-01-01 10:00", None])
    rep = fd.cdc_profile(df, event_time="event_ts", key="entity_id", now=NOW)
    kinds = {d.kind: d for d in rep.defects}
    assert "missing_event_time" in kinds
    assert kinds["missing_event_time"].level == "error"
    assert not rep.passed


def test_out_of_order_vs_late_split_by_lateness():
    # a@10:05 then a@10:04 (1 min behind -> within 2m tolerance = out_of_order)
    # then a@09:00 (far behind -> late)
    df = _events(
        entity_id=["a", "a", "a"],
        event_ts=pd.to_datetime(
            ["2024-01-01 10:05", "2024-01-01 10:04", "2024-01-01 09:00"]
        ),
    )
    rep = fd.cdc_profile(df, event_time="event_ts", key="entity_id", lateness="2m", now=NOW)
    kinds = {d.kind: d.n_rows for d in rep.defects}
    assert kinds.get("out_of_order") == 1
    assert kinds.get("late") == 1
    # late is an error, out_of_order is a warning
    assert not rep.passed


def test_per_key_ordering_is_independent():
    # interleaved keys; each key is individually ordered -> no ordering defect
    df = _events(
        entity_id=["a", "b", "a", "b"],
        event_ts=pd.to_datetime(
            ["2024-01-01 10:00", "2024-01-01 09:00", "2024-01-01 10:01", "2024-01-01 09:01"]
        ),
    )
    rep = fd.cdc_profile(df, event_time="event_ts", key="entity_id", lateness="0s", now=NOW)
    assert all(d.kind not in ("out_of_order", "late") for d in rep.defects)


def test_duplicate_cdc_key():
    df = _events(
        entity_id=["a", "a"],
        event_ts=pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:00"]),
        op=["U", "U"],
    )
    rep = fd.cdc_profile(df, event_time="event_ts", key="entity_id", now=NOW)
    dup = [d for d in rep.defects if d.kind == "duplicate_key"]
    assert dup and dup[0].n_rows == 2
    assert not rep.passed


def test_invalid_operation():
    df = _events(
        entity_id=["a", "b"],
        event_ts=pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:01"]),
        op=["U", "Z"],
    )
    rep = fd.cdc_profile(df, event_time="event_ts", key="entity_id", operation_col="op", now=NOW)
    bad = [d for d in rep.defects if d.kind == "invalid_operation"]
    assert bad and bad[0].n_rows == 1


def test_explicit_watermark_flags_late():
    df = _events(
        entity_id=["a", "b"],
        event_ts=pd.to_datetime(["2024-01-01 10:00", "2024-01-01 08:00"]),
    )
    rep = fd.cdc_profile(
        df, event_time="event_ts", key="entity_id",
        watermark=pd.Timestamp("2024-01-01 09:00"), now=NOW,
    )
    late = [d for d in rep.defects if d.kind == "late"]
    assert late and late[0].n_rows == 1


def test_stale_batch_warning_and_freshness():
    df = _events(entity_id=["a"], event_ts=pd.to_datetime(["2024-01-01 06:00"]))
    rep = fd.cdc_profile(df, event_time="event_ts", key="entity_id", now=NOW, stale_after="1h")
    assert rep.freshness_seconds == pytest.approx(6 * 3600)
    assert any(d.kind == "stale" for d in rep.defects)
    assert rep.trust_penalties["freshness"] == 1.0


def test_replay_risk_batch():
    df = _events(
        entity_id=["a", "a", "a", "a"],
        event_ts=pd.to_datetime(
            ["2024-01-01 10:00", "2024-01-01 10:00", "2024-01-01 09:00", "2024-01-01 11:00"]
        ),
    )
    rep = fd.cdc_profile(df, event_time="event_ts", key="entity_id", lateness="1m", now=NOW)
    assert any(d.kind == "replay_risk" for d in rep.defects)


def test_trust_penalties_present_and_bounded():
    df = _events(
        entity_id=["a", "a"],
        event_ts=pd.to_datetime(["2024-01-01 10:00", "2024-01-01 09:00"]),
    )
    rep = fd.cdc_profile(df, event_time="event_ts", key="entity_id", lateness="0s", now=NOW)
    for dim in ("freshness", "ordering", "cdc"):
        assert 0.0 <= rep.trust_penalties[dim] <= 1.0


def test_report_exports():
    df = _events(entity_id=["a"], event_ts=["bad-date"])
    rep = fd.cdc_profile(df, event_time="event_ts", key="entity_id", now=NOW)
    assert "cdc profile" in rep.summary()
    assert rep.to_dict()["event_time"] == "event_ts"
    assert list(rep.to_frame().columns)[:2] == ["kind", "level"]
    assert '"event_time"' in rep.to_json()


def test_missing_column_raises():
    df = _events(entity_id=["a"], event_ts=pd.to_datetime(["2024-01-01 10:00"]))
    with pytest.raises(KeyError, match="nope"):
        fd.cdc_profile(df, event_time="nope")


def test_does_not_mutate_input():
    df = _events(
        entity_id=["a", "a"],
        event_ts=pd.to_datetime(["2024-01-01 10:00", "2024-01-01 09:00"]),
    )
    before = df.copy(deep=True)
    fd.cdc_profile(df, event_time="event_ts", key="entity_id", now=NOW)
    pd.testing.assert_frame_equal(df, before)
