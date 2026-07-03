"""CleanBench mini fixtures in CI: release gates and metric self-checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(BENCH_DIR))

cleanbench = pytest.importorskip("cleanbench")


@pytest.fixture(scope="module")
def t2_run():
    truth, corrupted, kwargs = cleanbench.make_t2_semantic_fixture()
    repaired, report = fd.clean(corrupted, return_report=True, **kwargs)
    return truth, corrupted, repaired, report


@pytest.fixture(scope="module")
def t3_run():
    truth, corrupted, kwargs = cleanbench.make_t3_context_fixture()
    repaired, report = fd.clean(corrupted, return_report=True, **kwargs)
    return truth, corrupted, repaired, report


def test_t2_gates(t2_run):
    truth, corrupted, repaired, _ = t2_run
    pcv = cleanbench.protected_column_violation_rate(
        corrupted, repaired, ["monthly_revenue"]
    )
    assert pcv == cleanbench.GATE_PROTECTED_VIOLATION_RATE == 0.0
    fmr = cleanbench.false_modification_rate(truth, corrupted, repaired)
    assert fmr <= cleanbench.GATE_FALSE_MODIFICATION_RATE


def test_t2_repair_quality(t2_run):
    truth, corrupted, repaired, _ = t2_run
    assert cleanbench.cell_repair_precision(truth, corrupted, repaired) >= 0.99
    assert cleanbench.cell_repair_recall(truth, corrupted, repaired) >= 0.9
    assert cleanbench.cell_repair_f1(truth, corrupted, repaired) >= 0.95


def test_t3_context_compliance(t3_run):
    truth, corrupted, repaired, report = t3_run
    # Protected revenue: the deliberate dirt must SURVIVE untouched.
    assert repaired["monthly_revenue"].equals(corrupted["monthly_revenue"])
    # Age threshold: gaps stay missing.
    assert repaired["age"].isna().sum() == 3
    # Duplicate cust_id: reported by validate, not silently repaired.
    result = fd.validate(corrupted, context=cleanbench.ECOMMERCE_CONTEXT,
                         verbose=False)
    findings = result if isinstance(result, list) else getattr(result, "findings", result)
    assert findings, "uniqueness violation should produce a validation finding"
    fmr = cleanbench.false_modification_rate(truth, corrupted, repaired)
    assert fmr <= cleanbench.GATE_FALSE_MODIFICATION_RATE


def test_corruptors_are_deterministic():
    truth, _, _ = cleanbench.make_t2_semantic_fixture()
    for name, corruptor in cleanbench.CORRUPTORS.items():
        a, b = corruptor(truth, seed=42), corruptor(truth, seed=42)
        pd.testing.assert_frame_equal(a, b), name
        assert not a.equals(truth) or name == "date_phrase", (
            f"{name} corrupted nothing"
        )


def test_metrics_catch_protected_cell_change():
    """The violation metric must fire when protected cells are (deliberately) changed."""
    truth, corrupted, kwargs = cleanbench.make_t3_context_fixture()
    tampered = corrupted.copy(deep=True)
    tampered.loc[0, "monthly_revenue"] = "999999"
    rate = cleanbench.protected_column_violation_rate(
        corrupted, tampered, ["monthly_revenue"]
    )
    assert rate == 1.0


def test_false_modification_metric_works():
    truth = pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})
    corrupted = truth.copy()
    corrupted.loc[0, "a"] = "X"  # one corrupted cell
    repaired = corrupted.copy()
    repaired.loc[0, "a"] = "x"   # correct repair
    repaired.loc[1, "b"] = "!"   # false modification
    fmr = cleanbench.false_modification_rate(truth, corrupted, repaired)
    assert fmr == pytest.approx(1 / 3)
    assert cleanbench.cell_repair_recall(truth, corrupted, repaired) == 1.0
    assert cleanbench.cell_repair_precision(truth, corrupted, repaired) == 0.5


def test_duplicate_row_injection_row_level():
    truth, _, _ = cleanbench.make_t2_semantic_fixture()
    injected = cleanbench.duplicate_row_injection(truth, seed=1, n_duplicates=3)
    assert len(injected) == len(truth) + 3
    cleaned = fd.clean(injected, verbose=False)
    assert len(cleaned) == len(truth)
