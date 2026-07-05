"""Context-policy support in StreamingCleaner.

A supplied ``context=``/``policy=`` is compiled once against the first batch and
then governs every batch: protected columns are never imputed and stay
byte-identical, and the policy is surfaced on the report exactly once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import freshdata as fd


def make_batch(i, n=200, seed=None):
    rng = np.random.default_rng(seed if seed is not None else i)
    revenue = rng.lognormal(6.0, 0.4, n)
    revenue[rng.random(n) < 0.15] = np.nan  # missing cells the imputer *would* fill
    age = rng.lognormal(3.4, 0.5, n)
    age[rng.random(n) < 0.12] = np.nan
    return pd.DataFrame({
        "customer_id": np.arange(i * n, i * n + n, dtype="float64"),
        "churn": rng.integers(0, 2, n),
        "age": age,
        "revenue": revenue,
    })


def run(cleaner, n_batches=4, **kw):
    return [cleaner.clean_batch(make_batch(i, **kw)) for i in range(n_batches)]


def test_protected_column_never_imputed_across_stream():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
                            warmup_batches=1, context="Never modify revenue.")
    out = run(c, 4)
    # ``age`` is imputed post-warmup; ``revenue`` keeps its missing cells because
    # the policy protects it — for every batch, including the last.
    for cleaned, _ in out[1:]:
        assert cleaned["revenue"].isna().sum() > 0
    last_cleaned, _ = out[-1]
    assert last_cleaned["age"].isna().sum() == 0
    assert "revenue" in c.policy_.protected_columns


def test_protected_column_values_are_byte_identical():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
                            warmup_batches=1, context="Never modify revenue.")
    batch = make_batch(7)
    cleaned, _ = c.clean_batch(batch)
    before = batch["revenue"]
    after = cleaned["revenue"].reindex(before.index)
    pd.testing.assert_series_equal(before, after, check_names=False)


def test_policy_compiled_and_reported_once():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
                            warmup_batches=1, context="Never modify revenue.")
    out = run(c, 3)

    def has_compile_entry(report):
        return any(a.step == "context" and "compiled context policy" in a.description
                   for a in report)

    assert has_compile_entry(out[0][1])          # surfaced on the first batch
    assert not has_compile_entry(out[1][1])      # ...and never again
    assert not has_compile_entry(out[2][1])
    # ...but the compact summary rides along on every batch's report.
    for _, report in out:
        summary = report.streaming["context_policy"]
        assert summary["protected_columns"] == ["revenue"]


def test_precompiled_policy_object_is_reused():
    policy = fd.compile_context("Never modify revenue.", df=make_batch(0))
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
                            warmup_batches=1, policy=policy)
    out = run(c, 3)
    last_cleaned, _ = out[-1]
    assert last_cleaned["revenue"].isna().sum() > 0
    assert "revenue" in c.policy_.protected_columns


def test_no_context_is_zero_behaviour_change():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
                            warmup_batches=1)
    out = run(c, 3)
    last_cleaned, last_report = out[-1]
    assert c.policy_ is None
    assert "context_policy" not in last_report.streaming
    # Without protection the numeric imputer fills revenue post-warmup.
    assert last_cleaned["revenue"].isna().sum() == 0


def test_finalize_reports_context_policy():
    c = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
                            warmup_batches=1, context="Never modify revenue.")
    run(c, 3)
    rep = c.finalize()
    assert rep.streaming["context_policy"]["protected_columns"] == ["revenue"]
