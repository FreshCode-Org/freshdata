"""Unit tests for the bounded running-statistic primitives and StreamingState."""

import numpy as np
import pandas as pd

from freshdata.streaming._state import ColumnState, StreamingState
from freshdata.streaming._stats import BoundedCounter, ReservoirSampler, Welford


def test_welford_matches_numpy_across_batches():
    w = Welford()
    a, b, c = np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 30]), np.array([-5.0, 7, 7, 8, 9])
    for batch in (a, b, c):
        w.update(batch)
    allv = np.concatenate([a, b, c])
    assert w.count == allv.size
    assert abs(w.mean - allv.mean()) < 1e-9
    assert abs(w.variance - allv.var(ddof=1)) < 1e-9


def test_welford_empty_and_single():
    w = Welford()
    w.update(np.array([]))
    assert w.count == 0 and w.variance == 0.0
    w.update(np.array([42.0]))
    assert w.mean == 42.0 and w.variance == 0.0


def test_reservoir_is_bounded_and_approximates_median():
    r = ReservoirSampler(1000, seed=1)
    data = np.arange(100_000, dtype=float)
    for i in range(0, data.size, 10_000):
        r.update(data[i : i + 10_000])
    assert r.n_seen == 100_000
    assert r.size == 1000  # never exceeds capacity regardless of stream length
    assert abs(r.median() - 49_999.5) < 3_000  # approximate but close


def test_reservoir_empty():
    assert ReservoirSampler(10).median() is None


def test_bounded_counter_caps_and_finds_mode():
    c = BoundedCounter(3)
    s = pd.Series(["a"] * 100 + ["b"] * 50 + ["c"] * 10 + ["d"] * 5 + ["e"])
    c.update_from_series(s)
    assert len(c.counts) <= 3
    assert c.mode() == "a"
    assert 0.0 < c.mode_ratio() <= 1.0
    assert c.saturated  # five distinct keys, capacity three


def test_column_state_missing_ratio_and_numeric_snapshot():
    cs = ColumnState("x", "numeric", "float64", 1, reservoir_size=1000, max_categories=8, seed=0)
    cs.update(pd.Series([1.0, 2.0, np.nan, 4.0]))
    cs.update(pd.Series([np.nan, 6.0]))
    assert cs.seen == 6
    assert cs.missing == 2
    assert abs(cs.missing_ratio - 2 / 6) < 1e-9
    snap = cs.numeric_snapshot()
    assert snap.count == 4
    assert snap.minimum == 1.0 and snap.maximum == 6.0


def test_column_state_datetime_order_signal():
    ordered = ColumnState("t", "datetime", "datetime64[ns]", 1,
                          reservoir_size=10, max_categories=8, seed=0)
    ordered.update(pd.to_datetime(pd.Series(["2020-01-01", "2020-01-02"])))
    ordered.update(pd.to_datetime(pd.Series(["2020-01-03", "2020-01-04"])))
    assert ordered.datetime_ordered

    jumbled = ColumnState("t", "datetime", "datetime64[ns]", 1,
                          reservoir_size=10, max_categories=8, seed=0)
    jumbled.update(pd.to_datetime(pd.Series(["2020-01-05", "2020-01-01"])))
    assert not jumbled.datetime_ordered


def test_streaming_state_tracks_rows_and_trust():
    state = StreamingState(rolling_trust_window=2)
    state.observe_batch(pd.DataFrame({"a": [1, 2, 3]}), roles={"a": "numeric"})
    state.observe_batch(pd.DataFrame({"a": [4, 5]}), roles={"a": "numeric"})
    assert state.batch_count == 2
    assert state.rows_seen == 5
    assert state.schema_baseline == ["a"]

    rolling, cumulative = state.record_trust(80.0, rows=3)
    assert rolling == 80.0 and cumulative == 80.0
    rolling, cumulative = state.record_trust(90.0, rows=1)
    assert rolling == 85.0  # window of 2
    assert abs(cumulative - (80 * 3 + 90 * 1) / 4) < 1e-9  # rows-weighted


def test_state_to_dict_is_json_friendly():
    state = StreamingState()
    state.observe_batch(pd.DataFrame({"n": [1.0, 2.0], "c": ["x", "y"]}),
                        roles={"n": "numeric", "c": "categorical"})
    d = state.to_dict()
    assert d["rows_seen"] == 2
    assert "n" in d["columns"] and "c" in d["columns"]
