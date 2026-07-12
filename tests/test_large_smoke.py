"""Bounded real-scale smoke: 1M rows through the native cleaning subset on
every PR, so 'works at scale' is exercised continuously rather than only in
the nightly lane. Deterministic (seeded), no network, target < 60s."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd

ROWS = 1_000_000


@pytest.fixture(scope="module")
def million_row_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = ROWS
    return pd.DataFrame(
        {
            " User ID ": np.arange(n),
            "value": rng.normal(100.0, 15.0, n),
            "category": rng.choice(["a", "b", "c", "N/A"], n),
            "note": rng.choice([" ok ", "fine", "-", None], n),
        }
    )


@pytest.mark.large_smoke
def test_million_row_clean_conservative(million_row_frame):
    df = million_row_frame
    cleaned, report = fd.clean(
        df,
        strategy="conservative",
        fix_dtypes=False,
        return_report=True,
        verbose=False,
    )
    assert report.rows_before == ROWS
    assert list(cleaned.columns)[0] == "user_id"
    # sentinel "N/A" and "-" became real missing values
    assert cleaned["category"].isna().sum() > 0
    assert cleaned["note"].isna().sum() > df["note"].isna().sum()
    # whitespace trimmed
    assert (cleaned["note"].dropna() == "ok").sum() > 0
    assert report.duration_seconds < 60


@pytest.mark.large_smoke
def test_million_row_polars_native_parity(million_row_frame):
    pytest.importorskip("polars")
    df = million_row_frame
    kw = {"strategy": "conservative", "fix_dtypes": False, "verbose": False}
    ref = fd.clean(df, **kw)
    nat, report = fd.clean(
        df, engine="polars", fallback_policy="error", return_report=True, **kw
    )
    assert report.backend == "polars"
    assert report.fallback_events == []
    key = list(ref.columns)
    # normalize None vs nan in object columns (polars round-trip artefact)
    ref = ref.where(ref.notna(), np.nan)
    nat = nat.where(nat.notna(), np.nan)
    pd.testing.assert_frame_equal(
        ref.sort_values(key).reset_index(drop=True),
        nat.sort_values(key).reset_index(drop=True),
        check_dtype=False,  # polars round-trips object->string-backed columns
    )
