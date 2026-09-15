"""Regression tests for streaming cross-batch dedup (#292, #293) and drift (#294)."""

from __future__ import annotations

import pandas as pd

import freshdata as fd


def _keys_batch(keys):
    return pd.DataFrame({"k": keys, "s": [f"x{i}" for i in keys]})


def _run(cleaner, batches):
    return [cleaner.clean_batch(b)[0] for b in batches]


# -- #292: the window keeps the most recent rows, not the first ones -------------------


def test_duplicates_of_recent_rows_are_removed_after_window_fills():
    cleaner = fd.StreamingCleaner(global_duplicates=True, window_size=2, verbose=False)
    outs = _run(cleaner, [_keys_batch([1, 2]), _keys_batch([3, 4]), _keys_batch([3, 4])])
    assert outs[2]["k"].tolist() == []


def test_duplicates_after_more_than_window_size_distinct_rows():
    cleaner = fd.StreamingCleaner(global_duplicates=True, window_size=3, verbose=False)
    batches = [_keys_batch([1, 2, 3]), _keys_batch([4, 5, 6]), _keys_batch([7, 8]),
               _keys_batch([6, 7, 8])]
    outs = _run(cleaner, batches)
    assert outs[3]["k"].tolist() == []


def test_oldest_rows_are_evicted_and_window_stays_bounded():
    cleaner = fd.StreamingCleaner(global_duplicates=True, window_size=2, verbose=False)
    outs = _run(cleaner, [_keys_batch([1, 2]), _keys_batch([3, 4]), _keys_batch([1])])
    assert outs[2]["k"].tolist() == [1]  # 1 fell out of the 2-row window
    assert len(cleaner._seen_hashes) <= 2


def test_repeat_refreshes_recency():
    cleaner = fd.StreamingCleaner(global_duplicates=True, window_size=2, verbose=False)
    outs = _run(cleaner, [
        _keys_batch([1, 2]),
        _keys_batch([1]),  # duplicate: removed, and 1 becomes most recent
        _keys_batch([3]),  # evicts 2, the least recently seen
        _keys_batch([1]),
        _keys_batch([2]),
    ])
    assert [len(o) for o in outs] == [2, 0, 1, 0, 1]


def test_batch_larger_than_window_keeps_only_latest_rows():
    cleaner = fd.StreamingCleaner(global_duplicates=True, window_size=2, verbose=False)
    outs = _run(cleaner, [_keys_batch([1, 2, 3, 4]), _keys_batch([1, 3, 4])])
    assert outs[1]["k"].tolist() == [1]
    assert len(cleaner._seen_hashes) == 2


# -- #293: hashing survives int64 <-> float64 flips ------------------------------------


def test_int_row_repeats_in_nan_promoted_float_batch():
    cleaner = fd.StreamingCleaner(global_duplicates=True, verbose=False)
    cleaner.clean_batch(pd.DataFrame({"k": [1, 2], "v": [10, 20]}))
    out, report = cleaner.clean_batch(pd.DataFrame({"k": [1, 3], "v": [10, None]}))
    assert out["k"].tolist() == [3]
    assert report.duplicates_removed == 1


def test_float_row_repeats_in_later_int_batch():
    cleaner = fd.StreamingCleaner(global_duplicates=True, verbose=False)
    cleaner.clean_batch(pd.DataFrame({"k": [1, 2], "v": [10.0, None]}))
    out, _ = cleaner.clean_batch(pd.DataFrame({"k": [2, 5], "v": [20, 30]}))
    assert out["k"].tolist() == [2, 5]  # (2, NaN) != (2, 20)
    out, _ = cleaner.clean_batch(pd.DataFrame({"k": [1, 6], "v": [10, 60]}))
    assert out["k"].tolist() == [6]


def test_nullable_integer_matches_plain_float():
    cleaner = fd.StreamingCleaner(global_duplicates=True, verbose=False)
    cleaner.clean_batch(pd.DataFrame({"v": pd.array([1, None, 3], dtype="Int64")}))
    out, _ = cleaner.clean_batch(pd.DataFrame({"v": [3.0, 4.0]}))
    assert out["v"].tolist() == [4.0]


def test_negative_zero_matches_integer_zero():
    cleaner = fd.StreamingCleaner(global_duplicates=True, verbose=False)
    cleaner.clean_batch(pd.DataFrame({"k": ["a", "b"], "v": [0, 1]}))
    out, _ = cleaner.clean_batch(pd.DataFrame({"k": ["a", "c"], "v": [-0.0, 2.5]}))
    assert out["k"].tolist() == ["c"]


def test_distinct_numeric_values_are_not_merged():
    cleaner = fd.StreamingCleaner(global_duplicates=True, verbose=False)
    cleaner.clean_batch(pd.DataFrame({"k": ["a"], "v": [1]}))
    out, _ = cleaner.clean_batch(pd.DataFrame({"k": ["a"], "v": [1.5]}))
    assert len(out) == 1


def test_integers_beyond_float64_precision_stay_distinct():
    big = 2**53
    cleaner = fd.StreamingCleaner(global_duplicates=True, verbose=False)
    cleaner.clean_batch(pd.DataFrame({"id": [big + 1]}))
    out, _ = cleaner.clean_batch(pd.DataFrame({"id": [big]}))
    assert out["id"].tolist() == [big]


# -- #294: drift on a column that was constant ------------------------------------------


def _drift_actions(report):
    return [a for a in report.actions if a.step == "drift"]


def test_constant_column_jump_reports_distribution_drift():
    cleaner = fd.StreamingCleaner(verbose=False)
    for val in (5, 5, 5, 1_000_000):
        _, report = cleaner.clean_batch(pd.DataFrame({"x": [val] * 50, "y": list(range(50))}))
    assert report.streaming["schema_drift_detected"] is True
    actions = _drift_actions(report)
    assert [a.column for a in actions] == ["x"]
    assert actions[0].risk == "high"
    assert cleaner.state.drift_log[-1]["kind"] == "distribution"


def test_constant_column_unchanged_reports_no_drift():
    cleaner = fd.StreamingCleaner(verbose=False)
    for _ in range(4):
        _, report = cleaner.clean_batch(pd.DataFrame({"x": [5.0] * 50, "y": list(range(50))}))
        assert _drift_actions(report) == []
    assert cleaner.state.drift_log == []


def test_constant_column_within_float_tolerance_reports_no_drift():
    cleaner = fd.StreamingCleaner(verbose=False)
    cleaner.clean_batch(pd.DataFrame({"x": [0.1] * 10, "y": list(range(10))}))
    _, report = cleaner.clean_batch(
        pd.DataFrame({"x": [0.1 + 1e-15] * 10, "y": list(range(10))}))
    assert _drift_actions(report) == []


def test_single_prior_value_does_not_report_drift():
    cleaner = fd.StreamingCleaner(verbose=False)
    cleaner.clean_batch(pd.DataFrame({"x": [5], "y": ["a"]}))
    _, report = cleaner.clean_batch(pd.DataFrame({"x": [9], "y": ["a"]}))
    assert _drift_actions(report) == []
