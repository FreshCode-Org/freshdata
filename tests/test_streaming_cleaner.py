"""Behavioural tests for StreamingCleaner: imputation, safety gates, connectors."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.report import Action, CleanReport

HAS_POLARS = importlib.util.find_spec("polars") is not None
HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


def make_batch(i, n=200, seed=None):
    rng = np.random.default_rng(seed if seed is not None else i)
    age = rng.lognormal(3.4, 0.5, n)  # skewed numeric
    age[rng.random(n) < 0.12] = np.nan
    return pd.DataFrame({
        "customer_id": np.arange(i * n, i * n + n, dtype="float64"),
        "churn": rng.integers(0, 2, n),
        "age": age,
        "region": rng.choice(["north", "north", "north", "south", None], n),
        "notes": ["a fairly long free text note describing the account history"] * n,
    })


def run(cleaner, n_batches=4, **kw):
    out = [cleaner.clean_batch(make_batch(i, **kw)) for i in range(n_batches)]
    return out[-1]  # (cleaned, report) of the last (post-warmup) batch


def test_cleans_multiple_batches_and_persists_state():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
                            warmup_batches=1, window_size=100_000)
    run(c, 4)
    assert c.n_batches_seen == 4
    assert c.n_rows_seen == 800
    assert c.is_warmed_up


def test_numeric_imputation_after_warmup():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",), warmup_batches=1)
    cleaned, report = run(c, 4)
    assert cleaned["age"].isna().sum() == 0
    fills = [a for a in report if a.column == "age" and a.count > 0]
    assert fills and "running" in fills[0].description


def test_categorical_mode_imputation_after_warmup():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",), warmup_batches=1)
    cleaned, report = run(c, 4)
    assert cleaned["region"].isna().sum() == 0
    fills = [a for a in report if a.column == "region" and a.count > 0]
    assert fills and ("mode" in fills[0].description or "sentinel" in fills[0].description)


def test_id_column_never_imputed():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",), warmup_batches=1)

    def batch(i):
        df = make_batch(i)
        df.loc[df.index[:10], "customer_id"] = np.nan
        return df

    for i in range(3):
        cleaned, report = c.clean_batch(batch(i))
    assert cleaned["customer_id"].isna().sum() > 0
    assert "customer_id" not in report.columns_imputed
    assert "customer_id" in report.columns_preserved


def test_target_column_never_modified():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",), warmup_batches=1)

    def batch(i):
        df = make_batch(i).astype({"churn": "float64"})
        df.loc[df.index[:5], "churn"] = np.nan
        return df

    for i in range(3):
        cleaned, report = c.clean_batch(batch(i))
    assert cleaned["churn"].isna().sum() > 0
    assert "churn" not in report.columns_imputed


def test_free_text_not_force_filled():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",), warmup_batches=1)

    def batch(i):
        df = make_batch(i)
        df.loc[df.index[:20], "notes"] = np.nan
        return df

    for i in range(3):
        cleaned, report = c.clean_batch(batch(i))
    assert cleaned["notes"].isna().sum() > 0
    assert "notes" not in report.columns_imputed


def test_high_missing_column_preserved_with_warning():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",), warmup_batches=1)

    def batch(i):
        df = make_batch(i)
        sparse = np.full(len(df), np.nan)
        sparse[: len(df) // 5] = 1.0  # 80% missing
        df["sparse"] = sparse
        return df

    for i in range(3):
        cleaned, report = c.clean_batch(batch(i))
    assert "sparse" not in report.columns_imputed
    assert any("sparse" in w for w in report.warnings)


def test_actions_carry_rationale_risk_confidence():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",), warmup_batches=1)
    _, report = run(c, 3)
    acted = [a for a in report if a.step == "missing"]
    assert acted
    for a in acted:
        assert a.rationale
        assert a.risk in ("low", "medium", "high")
        assert 0.0 <= a.confidence <= 1.0


def test_warmup_defers_imputation():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",), warmup_batches=2)
    cleaned, report = c.clean_batch(make_batch(0))
    assert report.streaming["warmup_phase"] is True
    assert cleaned["age"].isna().sum() > 0  # not yet imputed
    assert any("deferred" in a.description for a in report)


def test_fail_under_trust_marks_gate():
    c = fd.StreamingCleaner(warmup_batches=0, fail_under_trust=99.999)
    _, report = c.clean_batch(make_batch(0))
    assert report.streaming["trust_gate_passed"] is False
    assert c._gate_failures == 1


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
def test_pyarrow_batch_input():
    import pyarrow as pa

    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",), warmup_batches=0)
    table = pa.Table.from_pandas(make_batch(0), preserve_index=False)
    cleaned, _ = c.clean_batch(table)
    assert isinstance(cleaned, pd.DataFrame) and len(cleaned) == 200


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_polars_batch_input():
    import polars as pl

    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",), warmup_batches=0)
    cleaned, _ = c.clean_batch(pl.from_pandas(make_batch(0)))
    assert isinstance(cleaned, pd.DataFrame) and len(cleaned) == 200


def test_finalize_summarizes_stream():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",), warmup_batches=1)
    run(c, 4)
    final = c.finalize()
    assert isinstance(final, CleanReport)
    assert final.streaming["batches"] == 4
    assert final.streaming["rows_seen_total"] == 800
    assert final.streaming["cells_imputed"] > 0


def test_report_action_types_unchanged():
    # Streaming reuses the existing Action shape — no new fields needed.
    c = fd.StreamingCleaner(warmup_batches=0)
    _, report = c.clean_batch(make_batch(0))
    assert all(isinstance(a, Action) for a in report)


# -- imputation branches that draw on running state --------------------------------


def test_datetime_imputation_within_ordered_window():
    cleaner = fd.StreamingCleaner(warmup_batches=1)
    base = pd.DataFrame({"event_time": pd.date_range("2020-01-01", periods=200, freq="min"),
                         "value": np.arange(200, dtype="float64")})
    cleaner.clean_batch(base)  # monotonic timestamps -> ordering signal is set
    nxt = pd.DataFrame({"event_time": pd.date_range("2020-01-01 03:20", periods=200, freq="min"),
                        "value": np.arange(200, 400, dtype="float64")})
    nxt.loc[nxt.index[5:8], "event_time"] = pd.NaT
    cleaned, report = cleaner.clean_batch(nxt)

    assert cleaned["event_time"].isna().sum() == 0  # ffill/bfill closed the gap
    fills = [a for a in report if a.step == "missing" and a.column == "event_time"]
    assert fills and fills[0].count == 3


def test_unordered_datetime_is_preserved():
    cleaner = fd.StreamingCleaner(warmup_batches=1)
    rng = np.random.default_rng(1)
    t0 = pd.Timestamp("2020-01-01")
    base = pd.DataFrame({"event_time": t0 + pd.to_timedelta(rng.integers(0, 10**6, 200), "s"),
                         "value": rng.normal(0, 1, 200)})
    cleaner.clean_batch(base)  # shuffled timestamps -> no usable order
    nxt = pd.DataFrame({"event_time": t0 + pd.to_timedelta(rng.integers(0, 10**6, 200), "s"),
                        "value": rng.normal(0, 1, 200)})
    nxt.loc[nxt.index[:4], "event_time"] = pd.NaT
    cleaned, report = cleaner.clean_batch(nxt)

    # Without a usable order, a datetime gap is preserved, never invented.
    assert cleaned["event_time"].isna().sum() == 4
    assert any(a.step == "missing" and a.column == "event_time" and a.count == 0 for a in report)


def test_categorical_sentinel_when_no_dominant_mode():
    # Pin target/id so 'label' is classified as a plain categorical column.
    cleaner = fd.StreamingCleaner(target_column="churn", id_columns=("rowid",), warmup_batches=1)
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"rowid": np.arange(200), "churn": rng.integers(0, 2, 200),
                         "segment": rng.choice(list("ABCDEFGH"), 200)})
    cleaner.clean_batch(base)  # ~8 equal categories -> no dominant mode
    nxt = pd.DataFrame({"rowid": np.arange(200, 400), "churn": rng.integers(0, 2, 200),
                        "segment": rng.choice(list("ABCDEFGH"), 200)})
    nxt.loc[nxt.index[:5], "segment"] = None
    cleaned, report = cleaner.clean_batch(nxt)

    assert cleaned["segment"].isna().sum() == 0
    fills = [a for a in report if a.step == "missing" and a.column == "segment"]
    assert fills and "sentinel" in fills[0].description.lower()


def test_global_duplicate_window_dedup():
    cleaner = fd.StreamingCleaner(global_duplicates=True, warmup_batches=5)
    batch = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "p", "q"]})
    cleaner.clean_batch(batch)
    cleaned, report = cleaner.clean_batch(batch.copy())  # identical rows again

    assert len(cleaned) == 0  # every row is a cross-batch duplicate
    assert report.duplicates_removed == 5
    assert any(a.step == "duplicates" for a in report)
