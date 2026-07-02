"""Cross-cutting guarantees: determinism, idempotency, silence, scale."""

import warnings

import numpy as np
import pandas as pd

import freshdata as fd


def test_cleaning_is_deterministic(messy):
    a = fd.clean(messy)
    b = fd.clean(messy)
    pd.testing.assert_frame_equal(a, b)


def test_cleaning_is_idempotent(messy):
    once = fd.clean(messy)
    twice = fd.clean(once)
    pd.testing.assert_frame_equal(once, twice)


def test_clean_and_profile_emit_no_warnings(messy):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        fd.clean(messy, impute="auto", outliers="clip", optimize_memory=True)
        fd.profile(messy)


def test_moderately_large_frame(messy):
    rng = np.random.default_rng(0)
    n = 20_000
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "amount": [f"${x:,.2f}" for x in rng.uniform(10, 99_999, n)],
            "when": pd.date_range("2020-01-01", periods=n, freq="h").astype(str),
            "flag": rng.choice(["yes", "no"], n),
            "junk": rng.choice(["N/A", "-", "ok", " padded "], n),
        }
    )
    out, report = fd.clean(df, report=True)
    assert out["amount"].dtype == "float64"
    assert str(out["when"].dtype).startswith("datetime64")
    assert out["flag"].dtype == bool
    assert report.rows_before == n


def test_sentinel_only_frame_collapses_gracefully():
    df = pd.DataFrame({"a": ["N/A", "-"], "b": ["", "null"]})
    out, report = fd.clean(df, report=True)
    # Everything was a sentinel: all cells -> missing, then rows/cols pruned.
    assert out.empty
    assert report.rows_after == 0 or report.cols_after == 0


def test_single_row_frame():
    df = pd.DataFrame({"A Col": ["  x  "]})
    out = fd.clean(df)
    assert out.shape == (1, 1)
    assert out["a_col"].iloc[0] == "x"


def test_original_index_preserved_by_default():
    df = pd.DataFrame({"v": ["1", "2", "2"]}, index=["r1", "r2", "r3"])
    out = fd.clean(df)
    assert out.index.tolist() == ["r1", "r2"]
