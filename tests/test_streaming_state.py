"""Unit tests for the bounded running-statistic primitives and StreamingState."""

import numpy as np
import pandas as pd
import pytest

from freshdata.streaming import StreamingCleaner
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


# --- documented behaviour: the rolling score is NOT row-weighted -------------
# docs/streaming.md called rolling_trust_score "row-weighted trust over the
# recent window", but it is an unweighted mean of the per-batch scores in the
# window — a 1-row batch counts exactly as much as a 999-row one. Only
# cumulative_trust_score is row-weighted. The docs were corrected to match;
# these tests pin the arithmetic so the two cannot drift apart again.

def test_rolling_trust_is_unweighted_while_cumulative_is_row_weighted():
    state = StreamingState(rolling_trust_window=8)
    state.record_trust(0.0, rows=1)               # tiny, terrible batch
    rolling, cumulative = state.record_trust(100.0, rows=999)  # huge, clean batch

    assert rolling == 50.0                        # (0 + 100) / 2 — rows ignored
    assert abs(cumulative - (0 * 1 + 100 * 999) / 1000) < 1e-9  # 99.9, rows honoured
    assert rolling != pytest.approx(cumulative)   # the two must not be conflated


def test_rolling_trust_ignores_batch_size_end_to_end():
    """The issue's reproduction: a 1-row all-None batch, then 999 clean rows.

    ``drop_empty_rows=False`` matters here: the streaming representation pass
    clears ``drop_empty_columns`` but keeps the default row dropping, so the
    all-None row would otherwise be removed before the batch is scored. That
    leaves a 0-row batch, and a 0-vs-999 split cannot demonstrate anything
    about weighting — the tiny batch would carry no weight under either rule.
    """
    cleaner = StreamingCleaner(verbose=False, drop_empty_rows=False)

    _, tiny_rep = cleaner.clean_batch(pd.DataFrame({"a": [None], "b": [None]}))
    _, big_rep = cleaner.clean_batch(pd.DataFrame({
        "a": list(range(999)), "b": [float(i) for i in range(999)],
    }))

    tiny, big = tiny_rep.streaming, big_rep.streaming
    scores = [tiny["batch_trust_score"], big["batch_trust_score"]]
    rows = [tiny["rows_in_batch"], big["rows_in_batch"]]
    assert rows == [1, 999]          # the 1-vs-999 split the issue describes
    assert scores[0] < scores[1]     # and the batches really do score differently

    # Derived from the observed scores, so an unrelated change to how a batch is
    # scored will not break this — only a change to the *weighting* will.
    assert big["rolling_trust_score"] == pytest.approx(sum(scores) / 2)
    assert big["rolling_trust_score"] != pytest.approx(
        sum(s * r for s, r in zip(scores, rows)) / sum(rows)
    )


def test_state_to_dict_is_json_friendly():
    state = StreamingState()
    state.observe_batch(pd.DataFrame({"n": [1.0, 2.0], "c": ["x", "y"]}),
                        roles={"n": "numeric", "c": "categorical"})
    d = state.to_dict()
    assert d["rows_seen"] == 2
    assert "n" in d["columns"] and "c" in d["columns"]
