"""The safety contract: id / target / free-text columns are never mutated.

These are the load-bearing tests. A non-zero false-repair rate on a protected
column is a library bug to block on, not a benchmark result (HARD CONSTRAINT 7).
"""

from __future__ import annotations

import freshdata as fd
import pytest

import harness_metrics as hm
from fixtures import FRAME_FIXTURES


@pytest.fixture(scope="module")
def cleaned_frames():
    out = {}
    for name in FRAME_FIXTURES:
        df = hm.make_frame(name, 3_000, seed=42)
        cleaned = fd.clean(df, config=hm.config_for(name, df)).reset_index(drop=True)
        out[name] = (df, cleaned)
    return out


@pytest.mark.parametrize("name", FRAME_FIXTURES)
def test_no_false_repair_on_protected_columns(name, cleaned_frames):
    df, cleaned = cleaned_frames[name]
    report = hm.preservation_report(name, df, cleaned)
    for role, stats in report["per_role"].items():
        assert stats["false_repair_rate_pct"] == 0.0, (name, role, stats)
        assert stats["preservation_rate_pct"] == 100.0, (name, role, stats)


@pytest.mark.parametrize("name", FRAME_FIXTURES)
def test_aggregate_preservation_perfect(name, cleaned_frames):
    df, cleaned = cleaned_frames[name]
    report = hm.preservation_report(name, df, cleaned)
    assert report["false_repair_rate_pct"] == 0.0
    assert report["preservation_rate_pct"] == 100.0
    assert report["ids_never_filled"] is True


def test_gold_false_repair_zero_and_preservation_full():
    gr = hm.gold_repair_report(10_000, seed=42)
    assert gr["false_repair_rate_pct"] == 0.0
    assert gr["preservation_rate_pct"] == 100.0
    for col, stats in gr["per_trap"].items():
        assert stats["false_repair_rate_pct"] == 0.0, (col, stats)


def test_gold_repair_fidelity_meets_threshold():
    gr = hm.gold_repair_report(10_000, seed=42)
    assert gr["repair_fidelity_pct"] >= 90.0
    # every defect family must individually clear the bar
    for family, pct in gr["per_family"].items():
        assert pct >= 90.0, (family, pct)
