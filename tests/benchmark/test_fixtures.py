"""Fixture generators: shape, determinism, manifest accuracy, gold consistency."""

from __future__ import annotations

import pandas as pd
import pytest

from fixtures import FRAME_FIXTURES, REGISTRY
from fixtures import gold as gold_mod


@pytest.mark.parametrize("name", FRAME_FIXTURES)
def test_generates_requested_size_and_width(name):
    mod = REGISTRY[name]
    n = 3_000
    df = mod.generate(n, seed=42)
    assert isinstance(df, pd.DataFrame)
    # row count is n plus the appended duplicate family (<= ~3%)
    assert n <= df.shape[0] <= int(n * 1.05)
    assert df.shape[1] == mod.N_COLS


@pytest.mark.parametrize("name", FRAME_FIXTURES)
def test_deterministic(name):
    mod = REGISTRY[name]
    a = mod.generate(2_000, seed=7)
    b = mod.generate(2_000, seed=7)
    assert a.equals(b)
    # a different seed must change the data
    c = mod.generate(2_000, seed=8)
    assert not a.equals(c)


def test_wide_schema_column_count_variants():
    from fixtures import wide_schema

    for n_cols in (100, 500):
        df = wide_schema.generate(1_000, seed=1, n_cols=n_cols)
        assert df.shape == (1_000, n_cols)
        assert len(wide_schema.gold_labels(n_cols)) == n_cols


@pytest.mark.parametrize("name", FRAME_FIXTURES + ("gold",))
def test_manifest_duplicate_count_within_tolerance(name):
    """The exact-duplicate family count must match its manifest rate +/-10%."""
    mod = REGISTRY[name]
    n = 5_000
    out = mod.generate(n, seed=42)
    df = out.dirty_df if name == "gold" else out
    # only the exact-duplicate family is full-row measurable via df.duplicated();
    # event_log's "replay_duplicate" is a key/time collision, not a row dup.
    dup_defects = [d for d in mod.DEFECT_MANIFEST if d["defect_type"] == "exact_duplicate_row"]
    if not dup_defects:
        pytest.skip("fixture has no exact-duplicate family")
    rate = dup_defects[0]["rate"]
    expected = rate * n
    actual = int(df.duplicated().sum())
    assert expected * 0.9 <= actual <= expected * 1.1, (name, expected, actual)


def test_crm_missing_and_null_families_within_tolerance():
    from fixtures import crm

    n = 10_000
    df = crm.generate(n, seed=42)
    base = df.iloc[:n]  # before appended duplicates
    # 6% missing lifetime_value, 1% null customer_id
    ltv_missing = int(base["lifetime_value"].isna().sum())
    id_null = int(base["customer_id"].isna().sum())
    assert 0.06 * n * 0.9 <= ltv_missing <= 0.06 * n * 1.1
    assert 0.01 * n * 0.9 <= id_null <= 0.01 * n * 1.1


def test_gold_bundle_shapes_consistent():
    n = 4_000
    b = gold_mod.generate(n, seed=42)
    assert b.clean_df.shape[0] == n
    assert b.dirty_df.shape[0] == n + b.n_duplicates
    # masks share clean_df's shape and columns
    for mask in (b.preservation_mask, b.repair_mask, b.false_repair_traps):
        assert mask.shape == b.clean_df.shape
        assert list(mask.columns) == list(b.clean_df.columns)
        assert mask.dtypes.map(lambda d: d == bool).all()


def test_gold_protected_columns_have_traps_everywhere():
    b = gold_mod.generate(2_000, seed=1)
    for col in gold_mod.PROTECTED:
        assert b.false_repair_traps[col].all()
        assert b.preservation_mask[col].all()
        assert not b.repair_mask[col].any()


def test_defect_rate_zero_is_pristine_for_gold():
    """defect_rate=0 must inject no defects (used by the trust sweep)."""
    b = gold_mod.generate(2_000, seed=3, defect_rate=0.0)
    assert b.n_duplicates == 0
    assert not b.dirty_df.duplicated().any()
    assert b.dirty_df["gid"].isna().sum() == 0
