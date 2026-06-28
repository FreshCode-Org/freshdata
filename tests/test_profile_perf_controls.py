"""Tests for F6 wide-schema perf controls on ``fd.profile``."""

from __future__ import annotations

import numpy as np
import pandas as pd

import freshdata as fd


def _wide_tall() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({f"c{i}": rng.integers(0, 5, 2000) for i in range(120)})


def test_full_profile_has_no_materialization():
    prof = fd.profile(_wide_tall())
    assert prof.materialization is None
    assert prof.n_cols == 120
    assert prof.n_rows == 2000


def test_max_columns_caps_profiled_columns():
    prof = fd.profile(_wide_tall(), max_columns=10)
    assert prof.n_cols == 10
    assert len(prof.columns) == 10
    assert prof.materialization["columns_total"] == 120
    assert prof.materialization["columns_omitted"] == 110


def test_profile_sample_caps_rows_deterministically():
    df = _wide_tall()
    p1 = fd.profile(df, profile_sample=300)
    p2 = fd.profile(df, profile_sample=300)
    assert p1.n_rows == 300
    assert p1.materialization["rows_total"] == 2000
    assert p1.materialization["sampled"] is True
    # deterministic: same sample both times
    assert p1.missing_cells == p2.missing_cells


def test_lazy_report_skips_duplicate_scan():
    prof = fd.profile(_wide_tall(), lazy_report=True)
    assert prof.duplicate_rows is None
    assert prof.materialization["lazy"] is True
    assert prof.materialization["duplicate_scan"] is False


def test_controls_compose_and_export():
    prof = fd.profile(_wide_tall(), max_columns=5, profile_sample=200, lazy_report=True)
    assert prof.n_cols == 5
    assert prof.n_rows == 200
    assert prof.duplicate_rows is None
    d = prof.to_dict()
    assert d["materialization"]["columns_profiled"] == 5
    assert d["materialization"]["rows_profiled"] == 200


def test_sample_larger_than_frame_is_noop():
    df = _wide_tall()
    prof = fd.profile(df, profile_sample=10_000)
    # no sampling happened -> describes full frame, no materialization from sampling
    assert prof.n_rows == 2000
    assert prof.materialization is None


def test_does_not_mutate_input():
    df = _wide_tall()
    before = df.copy(deep=True)
    fd.profile(df, max_columns=5, profile_sample=100, lazy_report=True)
    pd.testing.assert_frame_equal(df, before)
