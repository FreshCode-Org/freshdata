"""Phase 15: streaming batch-boundary invariance.

Feeds the *same* rows through :class:`~freshdata.StreamingCleaner` split into 1,
10, 100 and 1000 micro-batches and pins down exactly which outcomes are
batch-independent and which are not.

Batch-**independent** (asserted here):

* the emitted row count, column set and column order;
* every bounded running statistic in :class:`~freshdata.StreamingState` —
  Welford mean/std, the reservoir's median/quartiles (*even when the reservoir
  is smaller than the stream*), the Space-Saving mode/mode-ratio while the
  counter has not saturated, and the per-column missing counts;
* the cumulative ``finalize()`` missing accounting.

Batch-**dependent** (characterised here, not defects):

* which value lands in a missing cell — imputation reads the running state *as
  of that batch*, so the same cell fills differently at different batch sizes;
* how many cells get imputed at all (warmup deferral + the cumulative
  missing-band gate both move with batch size), and warmup-deferred cells are
  never back-filled;
* the Space-Saving mode once the counter saturates;
* recent-window cross-batch dedup (``global_duplicates=True``);
* the number of drift events.

Findings that fall out of the same harness are marked ``FINDING`` in the test
docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.streaming import StreamingCleanConfig
from freshdata.streaming._state import _TRUST_HISTORY_CAP
from freshdata.streaming._stats import BoundedCounter
from freshdata.streaming._timeseries import TimeSeriesCleanConfig

BATCH_COUNTS = (1, 10, 100, 1000)


def sample_frame(n: int = 1000, seed: int = 7) -> pd.DataFrame:
    """A deterministic mixed frame with missing cells in two columns."""
    rng = np.random.default_rng(seed)
    amount = rng.normal(100.0, 15.0, n)
    amount[rng.choice(n, 80, replace=False)] = np.nan
    segment = rng.choice(["alpha", "beta", "gamma"], n, p=[0.6, 0.25, 0.15]).astype(object)
    segment[rng.choice(n, 60, replace=False)] = None
    return pd.DataFrame({
        "customer_id": np.arange(n),
        "amount": amount,
        "segment": segment,
        "churn": rng.integers(0, 2, n),
    })


def split(df: pd.DataFrame, n_batches: int):
    for idx in np.array_split(np.arange(len(df)), n_batches):
        if len(idx):
            yield df.iloc[idx].reset_index(drop=True)


def run_split(df: pd.DataFrame, n_batches: int, **kwargs):
    """Stream *df* in *n_batches* batches; return ``(concat_output, cleaner)``."""
    kwargs.setdefault("warmup_batches", 0)
    cleaner = fd.StreamingCleaner(
        target_column="churn", id_columns=("customer_id",), **kwargs
    )
    parts = [out for out, _ in cleaner.clean_batches(split(df, n_batches))]
    return pd.concat(parts, ignore_index=True), cleaner


# --------------------------------------------------------------------------- #
# What IS batch-invariant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_batches", BATCH_COUNTS)
def test_row_count_and_schema_are_batch_invariant(n_batches: int) -> None:
    df = sample_frame()
    out, cleaner = run_split(df, n_batches)
    assert len(out) == len(df)
    assert list(out.columns) == list(df.columns)
    assert cleaner.n_rows_seen == len(df)
    assert cleaner.n_batches_seen == min(n_batches, len(df))


def test_running_statistics_are_batch_invariant() -> None:
    """Every bounded accumulator lands on the same answer at 1/10/100/1000
    batches. Reservoir quantiles, min/max, Space-Saving mode/ratio and the
    missing counters are **bit-identical**; Welford's mean/variance are exact
    only up to floating-point associativity, because Chan's parallel update
    sums the stream in a different order at each batch size."""
    df = sample_frame()
    exact = {}
    approx = {}
    for n_batches in BATCH_COUNTS:
        _, cleaner = run_split(df, n_batches)
        amount = cleaner.state.columns["amount"]
        segment = cleaner.state.columns["segment"]
        snap = amount.numeric_snapshot()
        exact[n_batches] = (
            snap.count, snap.median, snap.q1, snap.q3, snap.minimum, snap.maximum,
            amount.missing, amount.non_null,
            segment.mode(), segment.mode_ratio(), segment.missing, segment.non_null,
        )
        approx[n_batches] = (snap.mean, snap.std)

    reference = exact[1]
    for n_batches in BATCH_COUNTS[1:]:
        assert exact[n_batches] == reference, f"state drifted at {n_batches} batches"
    for n_batches in BATCH_COUNTS[1:]:
        assert approx[n_batches] == pytest.approx(approx[1], rel=1e-12)
    # The float drift is real but bounded at ~1 ULP, not a decision-changing one.
    assert approx[1000] != approx[1]


def test_reservoir_median_is_batch_invariant_even_under_capacity() -> None:
    """Vitter-R here draws exactly one variate per post-fill element, and the
    global 1-indexed position drives the decision — so splitting the stream
    into more batches cannot move the sample. (The brief listed approximate
    quantiles as batch-sensitive; for this implementation they are not.)"""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"v": rng.normal(0.0, 1.0, 5000), "g": ["a"] * 5000})
    cfg = {"quantile_reservoir_size": 100, "warmup_batches": 0, "seed": 0}
    medians = []
    for n_batches in BATCH_COUNTS:
        cleaner = fd.StreamingCleaner(streaming_config=StreamingCleanConfig(**cfg))
        for batch in split(df, n_batches):
            cleaner.clean_batch(batch)
        snap = cleaner.state.columns["v"].numeric_snapshot()
        medians.append((snap.median, snap.q1, snap.q3))
    assert len(set(medians)) == 1, medians
    # And the reservoir really was under capacity (approximation was in play).
    assert cleaner.state.columns["v"]._reservoir.n_seen == 5000
    assert cleaner.state.columns["v"]._reservoir.size == 100


@pytest.mark.parametrize("n_batches", BATCH_COUNTS)
def test_finalize_missing_accounting_is_batch_invariant(n_batches: int) -> None:
    df = sample_frame()
    _, cleaner = run_split(df, n_batches)
    report = cleaner.finalize()
    assert report.rows_before == report.rows_after == len(df)
    assert report.missing_before == int(df.isna().sum().sum())
    assert report.streaming["rows_seen_total"] == len(df)


# --------------------------------------------------------------------------- #
# What is NOT batch-invariant (characterisation, not defects)
# --------------------------------------------------------------------------- #


def test_imputed_values_are_not_batch_invariant() -> None:
    """Online imputation reads the running state *as of* the current batch, so
    the same missing cell fills with a different value at a different batch
    size. Only the *identity* of the missing cells is stable."""
    df = sample_frame()
    one, _ = run_split(df, 1)
    ten, _ = run_split(df, 10)
    hundred, _ = run_split(df, 100)

    missing = df["amount"].isna().to_numpy()
    # Every previously-missing cell is filled in each run ...
    for out in (one, ten, hundred):
        assert out["amount"].notna().all()
    # ... but not with the same value.
    assert not np.allclose(one["amount"], ten["amount"])
    assert not np.allclose(ten["amount"], hundred["amount"])
    # Non-missing cells are untouched and therefore identical everywhere.
    kept = ~missing
    assert np.allclose(one["amount"][kept], df["amount"][kept])
    assert np.allclose(hundred["amount"][kept], df["amount"][kept])


def test_categorical_sentinel_vs_mode_flips_with_batch_size() -> None:
    """The mode-vs-sentinel choice keys off the *cumulative* missing band and
    the running mode ratio, both of which move as batches arrive."""
    df = sample_frame()
    one, _ = run_split(df, 1)
    ten, _ = run_split(df, 10)
    filled_one = set(one["segment"][df["segment"].isna().to_numpy()])
    filled_ten = set(ten["segment"][df["segment"].isna().to_numpy()])
    assert filled_one != filled_ten
    assert filled_one <= {"alpha", "beta", "gamma", "Unknown", "Missing"}


def test_warmup_deferred_cells_are_never_backfilled() -> None:
    """Warmup batches are emitted with their gaps intact and never revisited,
    so the number of surviving NaNs is a direct function of batch size."""
    values = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, np.nan, 10.0] * 10
    df = pd.DataFrame({"v": values, "g": ["a"] * len(values)})
    leftovers = {}
    for n_batches in (1, 2, 4, 10):
        cleaner = fd.StreamingCleaner(warmup_batches=3)
        parts = [out for out, _ in cleaner.clean_batches(split(df, n_batches))]
        out = pd.concat(parts, ignore_index=True)
        leftovers[n_batches] = int(out["v"].isna().sum())
    # 1 and 2 batches are entirely inside warmup -> nothing is ever imputed.
    assert leftovers[1] == leftovers[2] == 30
    # More batches -> warmup covers less of the stream -> fewer surviving gaps.
    assert leftovers[10] < leftovers[4] < leftovers[2]
    assert leftovers[10] == 9  # exactly the 3 warmup batches' gaps


def test_single_row_batches_preserve_instead_of_impute() -> None:
    """Pathological batch size 1: the cumulative missing ratio starts at 100%
    for any column whose first observed cell is missing, so the band gate
    preserves it instead of filling."""
    df = pd.DataFrame({"v": [np.nan, 1.0, 2.0, 3.0], "g": ["a", "b", "c", "d"]})
    out, cleaner = run_split(df, 4, warmup_batches=0)
    assert np.isnan(out["v"].iloc[0])  # never filled
    assert cleaner.finalize().streaming["cells_imputed"] == 0
    # The same rows in one batch see a 25% missing ratio and *are* filled.
    out_one, _ = run_split(df, 1, warmup_batches=0)
    assert out_one["v"].notna().all()


def test_bounded_counter_mode_is_batch_sensitive_once_saturated() -> None:
    """Space-Saving is exact while it fits; past ``max_categories`` the mode it
    reports depends on the order (hence the batching) of the arrivals."""
    rng = np.random.default_rng(11)
    values = [f"c{rng.integers(0, 200)}" for _ in range(3000)] + ["A"] * 400 + ["B"] * 395
    rng.shuffle(values)
    series = pd.Series(values)
    modes = set()
    for n_batches in (1, 5, 50, 500):
        counter = BoundedCounter(8)
        for idx in np.array_split(np.arange(len(series)), n_batches):
            counter.update_from_series(series.iloc[idx])
        assert counter.saturated
        assert counter.total == len(series)  # the *total* stays exact
        modes.add(counter.mode())
    assert len(modes) > 1, "saturated Space-Saving was expected to be order-dependent"


def test_drift_event_count_is_batch_size_dependent() -> None:
    df = sample_frame()
    counts = {}
    for n_batches in BATCH_COUNTS:
        _, cleaner = run_split(df, n_batches)
        counts[n_batches] = cleaner.finalize().streaming["drift_events"]
    assert counts[1] == 0  # the first batch only establishes the baseline
    assert counts[1000] > counts[100] >= counts[10]


# --------------------------------------------------------------------------- #
# Duplicates across batch boundaries
# --------------------------------------------------------------------------- #


def duplicate_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3, 1, 2, 3] * 2, "b": ["x", "y", "z", "x", "y", "z"] * 2})


@pytest.mark.parametrize("n_batches", (1, 2, 4, 12))
def test_within_batch_duplicates_are_left_alone_by_default(n_batches: int) -> None:
    """With ``global_duplicates=False`` (the default) within-batch duplicates are
    left to the per-batch pipeline, which on this frame removes nothing at any
    strategy — so the row count is batch-invariant and equal to the input."""
    df = duplicate_frame()
    cleaner = fd.StreamingCleaner(warmup_batches=0)
    rows = sum(len(out) for out, _ in cleaner.clean_batches(split(df, n_batches)))
    assert rows == len(df)


def test_global_duplicates_is_a_recent_window_not_a_global_dedup() -> None:
    """FINDING (S3, spec gap): ``global_duplicates=True`` only compares a row
    against *earlier batches*. A single batch therefore de-duplicates nothing,
    and the surviving row count is a function of how the caller chunked the
    stream — not of the data."""
    df = duplicate_frame()
    rows = {}
    for n_batches in (1, 2, 4, 12):
        cleaner = fd.StreamingCleaner(
            streaming_config=StreamingCleanConfig(global_duplicates=True, warmup_batches=0)
        )
        rows[n_batches] = sum(len(out) for out, _ in cleaner.clean_batches(split(df, n_batches)))
    assert rows[1] == 12   # one batch: nothing removed at all
    assert rows[2] == 6
    assert rows[4] == 3
    assert rows[12] == 3   # fully de-duplicated only once every row is its own batch


def test_recent_window_eviction_lets_an_old_duplicate_through() -> None:
    """A repeat further back than ``window_size`` distinct rows is not caught."""
    cfg = StreamingCleanConfig(global_duplicates=True, window_size=2, warmup_batches=0)
    cleaner = fd.StreamingCleaner(streaming_config=cfg)
    first, _ = cleaner.clean_batch(pd.DataFrame({"a": [1], "b": ["x"]}))
    for value in (2, 3):
        cleaner.clean_batch(pd.DataFrame({"a": [value], "b": ["x"]}))
    again, report = cleaner.clean_batch(pd.DataFrame({"a": [1], "b": ["x"]}))
    assert len(first) == 1
    assert len(again) == 1  # evicted from the window, so re-emitted
    assert report.duplicates_removed == 0


def test_duplicate_inside_the_window_is_removed_and_audited() -> None:
    cfg = StreamingCleanConfig(global_duplicates=True, window_size=100, warmup_batches=0)
    cleaner = fd.StreamingCleaner(streaming_config=cfg)
    cleaner.clean_batch(pd.DataFrame({"a": [1], "b": ["x"]}))
    out, report = cleaner.clean_batch(pd.DataFrame({"a": [1], "b": ["x"]}))
    assert len(out) == 0
    assert report.duplicates_removed == 1
    assert any(a.step == "duplicates" for a in report.actions)


# --------------------------------------------------------------------------- #
# Schema changes across batches / pathological batches
# --------------------------------------------------------------------------- #


def test_column_missing_in_a_later_batch_is_drift_not_an_error() -> None:
    cleaner = fd.StreamingCleaner(warmup_batches=0)
    cleaner.clean_batch(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]}))
    out, report = cleaner.clean_batch(pd.DataFrame({"a": [4.0, 5.0, 6.0]}))
    assert list(out.columns) == ["a"]
    assert report.streaming["schema_drift_detected"] is True
    assert any("'b' is absent" in w for w in report.warnings)
    # The column's accumulated state survives the gap.
    assert set(cleaner.state.columns) == {"a", "b"}
    assert cleaner.finalize().cols_after == 2


def test_new_column_in_a_later_batch_is_drift_and_gets_its_own_state() -> None:
    cleaner = fd.StreamingCleaner(warmup_batches=0)
    cleaner.clean_batch(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    _, report = cleaner.clean_batch(pd.DataFrame({"a": [4.0, 5.0, 6.0], "z": [7.0, 8.0, 9.0]}))
    assert any("new column 'z'" in w for w in report.warnings)
    assert cleaner.state.columns["z"].first_seen_batch == 2
    assert cleaner.state.schema_baseline == ["a"]  # baseline stays locked to batch 1


def test_one_huge_batch_matches_the_single_batch_path() -> None:
    df = sample_frame(n=5000, seed=11)
    out, cleaner = run_split(df, 1)
    assert len(out) == 5000
    assert cleaner.n_batches_seen == 1
    assert cleaner.is_warmed_up is True


def test_empty_batch_scores_zero_trust_and_fails_the_gate() -> None:
    """FINDING (S3): an empty micro-batch — an idle Kafka poll or a trailing
    chunk — is scored 0.0 and therefore *fails* ``fail_under_trust``, counts as
    a gate failure (non-zero CLI exit) and drags the unweighted rolling trust
    down. Only the row-weighted cumulative score ignores it."""
    cfg = StreamingCleanConfig(fail_under_trust=50.0, warmup_batches=0)
    cleaner = fd.StreamingCleaner(streaming_config=cfg)
    _, first = cleaner.clean_batch(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]}))
    assert first.streaming["trust_gate_passed"] is True

    empty = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype=object)})
    out, report = cleaner.clean_batch(empty)
    assert len(out) == 0
    assert report.streaming["batch_trust_score"] == 0.0
    assert report.streaming["trust_gate_passed"] is False
    assert report.streaming["rolling_trust_score"] < first.streaming["rolling_trust_score"]
    final = cleaner.finalize()
    assert final.streaming["trust_gate_failures"] == 1
    # The row-weighted cumulative score is unaffected.
    assert final.streaming["cumulative_trust_score"] == 100.0


# --------------------------------------------------------------------------- #
# Memory-bound claim (docs/streaming.md: "100M rows costs the same as 100k")
# --------------------------------------------------------------------------- #


def oscillating_batches(n_batches: int):
    """Batches whose missing rate alternates 0% / 100% -> drift every batch."""
    for i in range(n_batches):
        values = [np.nan] * 10 if i % 2 else list(range(10))
        yield pd.DataFrame({"v": values, "k": list(range(10))})


def test_trust_history_is_capped() -> None:
    """The control case: ``StreamingState._trust_history`` is a bounded deque."""
    cleaner = fd.StreamingCleaner(warmup_batches=0)
    assert cleaner.state._trust_history.maxlen == _TRUST_HISTORY_CAP
    for batch in oscillating_batches(20):
        cleaner.clean_batch(batch)
    assert len(cleaner.state.trust_history) == 20


def test_drift_log_grows_without_bound_on_a_persistently_drifting_stream() -> None:
    """FINDING (S2): ``StreamingState.drift_log`` (_state.py:224) is a plain
    list with no cap. Under a stream that drifts every batch it grows strictly
    linearly in the number of batches — i.e. linearly in rows at a fixed batch
    size — which contradicts ``_state.py``'s "nothing here grows with the
    number of rows" and docs/streaming.md's constant-memory claim.

    Measured on this fixture: ~1.5 entries/batch, ~310 bytes/entry
    (50 batches -> 74 entries / 23 KB; 4000 batches -> 5999 entries / 1.85 MB).
    Proposed fix: back it with ``collections.deque(maxlen=...)`` exactly like
    ``_trust_history`` (cap 10_000). Note that both ``state_["n_drift_events"]``
    and ``finalize().streaming["drift_events"]`` are today *derived* from
    ``len(drift_log)``, and ``finalize()`` replays one warning per retained
    entry — so a cap needs a separate monotonic total alongside it, and should
    disclose how many entries were dropped.
    """
    sizes = {}
    for n_batches in (40, 160):
        cleaner = fd.StreamingCleaner(warmup_batches=0)
        for batch in oscillating_batches(n_batches):
            cleaner.clean_batch(batch)
        sizes[n_batches] = len(cleaner.state.drift_log)
        assert cleaner.finalize().streaming["drift_events"] == sizes[n_batches]
    assert sizes[40] > 40  # at least one event per batch
    # Strictly proportional: a 4x longer stream retains ~4x the log.
    ratio = sizes[160] / sizes[40]
    assert 3.6 < ratio < 4.4, sizes
    # No cap exists today (tripwire for the proposed fix).
    assert isinstance(cleaner.state.drift_log, list)


def late_data_batches(n_batches: int):
    base = pd.Timestamp("2026-01-01")
    for i in range(n_batches):
        ts = [base + pd.Timedelta(10 * i, unit="h")] + [base] * 19
        yield pd.DataFrame({"ts": ts, "v": np.arange(20, dtype="float64")})


def test_exception_batches_retain_every_quarantined_row() -> None:
    """FINDING (S2): ``StreamingCleaner._exception_batches`` (_cleaner.py:93)
    accumulates the **full DataFrame** of every quarantined row for the whole
    stream. On a heavily-quarantining stream the retained rows grow linearly
    with the rows consumed, so a 100M-row stream with a few percent late data
    holds millions of rows in memory — the single largest hole in the
    constant-memory claim.

    Measured on this fixture: 19 retained rows per batch
    (20 batches -> 361 rows / 29 KB; 400 batches -> 7581 rows / 614 KB).
    Proposed fix: cap the retained exception rows (a
    ``max_retained_exceptions`` knob on ``StreamingCleanConfig``, defaulting to
    something of ``_TRUST_HISTORY_CAP``'s order) and/or offer a
    spill-to-parquet sink, keeping ``late_quarantined_total`` as the exact
    uncapped counter; ``last_exceptions_`` already covers the per-batch case.
    """
    tcfg = TimeSeriesCleanConfig(
        timestamp_column="ts", event_time_column="ts",
        allowed_lateness="1s", late_data_action="quarantine",
    )
    retained = {}
    for n_batches in (10, 40):
        cleaner = fd.StreamingCleaner(time_series_config=tcfg, warmup_batches=0)
        for batch in late_data_batches(n_batches):
            cleaner.clean_batch(batch)
        exceptions = cleaner.exceptions_
        retained[n_batches] = len(exceptions)
        summary = cleaner.finalize().streaming["time_series"]
        assert summary["exceptions_total"] == len(exceptions)
        assert summary["late_quarantined_total"] == len(exceptions)
        assert "_quarantine_reason" in exceptions.columns
    assert retained[10] > 0
    assert retained[40] / retained[10] == pytest.approx(4.0, rel=0.15), retained
    assert isinstance(cleaner._exception_batches, list)


# --------------------------------------------------------------------------- #
# Context policy: compiled once against batch 1
# --------------------------------------------------------------------------- #


def test_policy_protects_a_column_present_in_the_first_batch() -> None:
    cleaner = fd.StreamingCleaner(warmup_batches=0, context="Never modify bonus.")
    bonus = [10.0 + i for i in range(20)]
    bonus[3] = np.nan
    batch = pd.DataFrame({"bonus": bonus, "name": [f"n{i}" for i in range(20)]})
    out, _ = cleaner.clean_batch(batch)
    assert cleaner.policy_ is not None
    assert "bonus" in cleaner.policy_.protected_columns
    assert np.isnan(out["bonus"].iloc[3])  # gap preserved, column untouched


def test_policy_does_not_protect_a_column_absent_from_the_first_batch() -> None:
    """FINDING (S2): the policy is compiled once against batch 1. A column that
    only appears later never resolves, so its ``never modify`` constraint is
    silently inert and the streaming imputer writes into it. docs/streaming.md
    documents the converse ("a batch that happens to be missing a column can't
    make the policy drift") but not this direction.

    The only signal is a batch-1 unresolved-reference warning — there is no
    per-batch re-check when the column finally arrives.
    """
    cleaner = fd.StreamingCleaner(warmup_batches=0, context="Never modify bonus.")
    first = pd.DataFrame({
        "salary": [100.0 + i for i in range(20)],
        "name": [f"n{i}" for i in range(20)],
    })
    _, report = cleaner.clean_batch(first)
    assert cleaner.policy_ is not None
    assert cleaner.policy_.protected_columns == ()
    assert [u.ref for u in cleaner.policy_.unresolved] == ["bonus"]
    assert any("unresolved column reference 'bonus'" in w for w in report.warnings)

    bonus = [10.0 + i for i in range(20)]
    bonus[3] = np.nan
    second = pd.DataFrame({
        "salary": [300.0 + i for i in range(20)],
        "name": [f"m{i}" for i in range(20)],
        "bonus": bonus,
    })
    out, second_report = cleaner.clean_batch(second)
    # The protected-by-intent column was mutated.
    assert not np.isnan(out["bonus"].iloc[3])
    assert "bonus" in second_report.columns_imputed
    # ... and batch 2 repeats no context warning.
    assert not any("bonus" in w and "context" in w for w in second_report.warnings)


def test_per_column_state_grows_with_distinct_column_names() -> None:
    """FINDING (S3): ``StreamingState.columns`` (and ``StreamingCleaner._roles``)
    is keyed by column name with no eviction, and every entry owns a full
    reservoir (``quantile_reservoir_size`` floats, 20k by default) plus a
    Space-Saving counter. A stream whose schema churns — partition- or
    date-named columns, a wide sparse event feed — therefore grows state
    linearly in *distinct column names* seen over the whole stream, which the
    constant-memory claim does not carve out.

    Measured: 20 churning batches -> 21 column states; 80 -> 81.
    """
    sizes = {}
    for n_batches in (20, 80):
        cleaner = fd.StreamingCleaner(warmup_batches=0)
        for i in range(n_batches):
            cleaner.clean_batch(
                pd.DataFrame({"stable": [1.0, 2.0, 3.0], f"day_{i}": [1.0, 2.0, 3.0]})
            )
        sizes[n_batches] = len(cleaner.state.columns)
        assert len(cleaner._roles) == sizes[n_batches]
    assert sizes == {20: 21, 80: 81}
