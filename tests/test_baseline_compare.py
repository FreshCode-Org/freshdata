"""Tests for the baseline-drift convenience (raw DataFrame baseline + key-level)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


# ---------------------------------------------------------------------------
# Distribution drift maths: ties, constant columns and saved precision
# (#234, #235, #275)
# ---------------------------------------------------------------------------


def _drift_problems(report: fd.DriftReport) -> list[tuple[str | None, str, str]]:
    return [
        (f.check_id, f.status, f.message)
        for f in report.findings
        if (f.check_id or "").startswith("drift.") and f.status != "passed"
    ]


_rs = np.random.RandomState(3)
_TIED_COLUMNS = {
    "zero_inflated_counts": [0] * 99 + [1],
    "small_int_ratings": [1, 2, 3] * 40,
    "constant": [5.0] * 100,
    # The median lands on a value that holds most of the rows between p25 and p75.
    "atom_straddling_quantiles": [0] * 9 + [1] * 15 + [2] * 8,
    "zero_inflated_continuous": list(np.where(_rs.rand(500) < 0.6, 0.0, _rs.exponential(5, 500))),
    "thirds": [k / 3 for k in (0, 1, 2)] * 40,
    "inexact_tenths": [0.1 * k for k in (1, 2, 3)] * 40,  # 0.30000000000000004
}


def test_issue_234_repro_frames_match_their_own_baseline() -> None:
    df = pd.DataFrame({"refunds": [0] * 99 + [1]})
    rep = fd.compare_to_baseline(df, fd.build_baseline(df, name="self"))
    df2 = pd.DataFrame({"rating": [1, 2, 3] * 40})
    rep2 = fd.compare_to_baseline(df2, fd.build_baseline(df2, name="self"))
    assert rep.passed and rep2.passed
    assert _drift_problems(rep) == [] and _drift_problems(rep2) == []


@pytest.mark.parametrize("values", list(_TIED_COLUMNS.values()), ids=list(_TIED_COLUMNS))
def test_frame_with_tied_values_matches_its_own_baseline(values: list[float], tmp_path) -> None:
    df = pd.DataFrame({"x": values})
    base = build_baseline(df, name="self")
    fd.save_baseline(base, tmp_path / "self.json")
    for bl in (base, fd.load_baseline(tmp_path / "self.json")):
        report = fd.compare_to_baseline(df, bl)
        assert report.passed
        assert _drift_problems(report) == []


def test_tied_baseline_still_flags_real_drift() -> None:
    refunds = build_baseline(pd.DataFrame({"refunds": [0] * 99 + [1]}), name="b")
    more_refunds = pd.DataFrame({"refunds": [0] * 80 + [1] * 20})
    assert not fd.compare_to_baseline(more_refunds, refunds).passed
    ratings = build_baseline(pd.DataFrame({"rating": [1, 2, 3] * 40}), name="b")
    assert not fd.compare_to_baseline(pd.DataFrame({"rating": [3] * 120}), ratings).passed


def test_point_mass_on_a_continuous_baseline_quantile_is_still_drift() -> None:
    # e.g. missing values imputed with the median: the baseline had no such spike.
    rs = np.random.RandomState(0)
    values = rs.normal(0, 1, 2000)
    current = values.copy()
    current[:1200] = np.median(values)
    base = build_baseline(pd.DataFrame({"x": values}), name="b")
    report = fd.compare_to_baseline(pd.DataFrame({"x": current}), base)
    assert ("drift.ks", "failed") in {(f.check_id, f.status) for f in report.findings}


def test_constant_baseline_column_still_checks_distribution() -> None:
    base = build_baseline(pd.DataFrame({"price": [5.0] * 100}), name="b")

    shifted = fd.compare_to_baseline(pd.DataFrame({"price": [1000.0] * 100}), base)
    assert not shifted.passed
    ks = [f for f in shifted.findings if f.check_id == "drift.ks"]
    assert [(f.status, f.current_value) for f in ks] == [("failed", 1.0)]
    assert shifted.distribution_drift["price"]["ks"] == 1.0

    partly = fd.compare_to_baseline(pd.DataFrame({"price": [4.0] * 20 + [5.0] * 80}), base)
    assert [(f[0], f[1]) for f in _drift_problems(partly)] == [("drift.ks", "warned")]
    assert fd.compare_to_baseline(pd.DataFrame({"price": [5.0] * 100}), base).passed


def _verdict(report: fd.DriftReport) -> tuple[bool, list[str]]:
    return report.passed, sorted(f.message for f in report.findings if f.status != "passed")


def test_saved_baseline_keeps_precision_for_small_values(tmp_path) -> None:
    rs = np.random.RandomState(7)
    train = pd.DataFrame({"p": rs.normal(2e-6, 4e-7, 2000)})
    today = pd.DataFrame({"p": rs.normal(2e-6, 4e-7, 2000)})
    base = build_baseline(train, name="b")
    fd.save_baseline(base, tmp_path / "b.json")
    loaded = fd.load_baseline(tmp_path / "b.json")

    col, saved = base.columns["p"], loaded.columns["p"]
    assert saved.quantiles == col.quantiles
    assert (saved.min, saved.max, saved.mean, saved.std) == (col.min, col.max, col.mean, col.std)
    assert _verdict(fd.compare_to_baseline(today, loaded)) == (True, [])
    assert _verdict(fd.compare_to_baseline(today, base)) == (True, [])


def test_saved_baseline_stats_drop_non_finite_values() -> None:
    col = build_baseline(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), name="b").columns["x"]
    col.mean, col.std = float("nan"), float("inf")
    d = col.to_dict()
    assert d["mean"] is None and d["std"] is None and d["max"] == 3.0
