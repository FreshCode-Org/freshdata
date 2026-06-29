"""Tests for the baseline-drift convenience (raw DataFrame baseline + key-level)."""

from __future__ import annotations

import pandas as pd

import freshdata as fd
from freshdata import build_baseline


def _baseline() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "spend": [10, 20, 30, 40],
        "tier": ["a", "a", "b", "b"],
        "updated_at": ["2026-06-01"] * 4,
    })


def test_added_removed_columns_detected() -> None:
    base = _baseline()
    current = base.drop(columns=["tier"]).assign(region=["e", "w", "e", "w"])
    diff = fd.compare_to_baseline(current, baseline=base)
    ids = {f.check_id for f in diff.findings}
    assert "schema.new_column" in ids
    assert "schema.removed_column" in ids


def test_dtype_drift_detected() -> None:
    base = _baseline()
    current = base.copy()
    current["spend"] = current["spend"].astype(str)
    diff = fd.compare_to_baseline(current, baseline=base)
    assert any(f.check_id == "schema.dtype_change" for f in diff.findings)


def test_key_level_changes() -> None:
    base = _baseline()
    current = pd.DataFrame({
        "customer_id": [2, 3, 4, 5],
        "spend": [20, 99, 40, 50],  # cust 3 changed
        "tier": ["a", "b", "b", "c"],
        "updated_at": ["2026-06-20"] * 4,
    })
    diff = fd.compare_to_baseline(current, baseline=base, key="customer_id",
                                  event_time="updated_at")
    kc = diff.key_changes
    assert kc["added"] == 1 and kc["removed"] == 1
    assert kc["changed"] == 1  # only customer 3, not the moving timestamp
    assert "latest_event_time" in kc


def test_business_summary_and_serialization() -> None:
    base = _baseline()
    current = base.drop(columns=["tier"])
    diff = fd.compare_to_baseline(current, baseline=base, key="customer_id")
    matters = diff.what_likely_matters()
    assert matters and isinstance(matters[0], str)
    d = diff.to_dict()
    assert "key_changes" in d
    assert "<div class=\"fd-report\"" in diff.to_html()


def test_prebuilt_baseline_still_works() -> None:
    base = _baseline()
    bl = build_baseline(base, name="weekly")
    diff = fd.compare_to_baseline(base, bl)  # positional, original API
    assert diff.passed
