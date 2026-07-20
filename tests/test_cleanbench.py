"""CleanBench mini fixtures in CI: release gates and metric self-checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
from freshdata.models import runtime as model_runtime

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
    # Removal is opt-in since the P1-1 audit fix; the injection round-trip is
    # what this test pins, so request it explicitly.
    cleaned = fd.clean(injected, drop_duplicates=True, verbose=False)
    assert len(cleaned) == len(truth)


# --------------------------------------------------------------------------- #
# Phase 3: calibration metrics + embedding mini-suite
# --------------------------------------------------------------------------- #


def test_expected_calibration_error_math():
    perfect = [(1.0, True)] * 10
    assert cleanbench.expected_calibration_error(perfect) == pytest.approx(0.0)
    overconfident = [(0.99, False)] * 10
    assert cleanbench.expected_calibration_error(overconfident) == pytest.approx(0.99)
    mixed = [(0.75, True), (0.75, True), (0.75, False), (0.75, False)]  # acc 0.5 @ 0.75
    assert cleanbench.expected_calibration_error(mixed) == pytest.approx(0.25)
    assert cleanbench.expected_calibration_error([]) == 0.0


def test_precision_at_confidence_bucket_math():
    pairs = [(0.99, True), (0.97, True), (0.96, False), (0.5, False)]
    assert cleanbench.precision_at_confidence_bucket(pairs, floor=0.95) == pytest.approx(2 / 3)
    assert cleanbench.precision_at_confidence_bucket([], floor=0.95) == 1.0
    assert cleanbench.precision_at_confidence_bucket([(0.5, False)], floor=0.95) == 1.0


def test_coverage_at_precision_math():
    pairs = [(0.99, True), (0.98, True), (0.9, True), (0.6, False)]
    # Threshold 0.9 gives precision 1.0 over 3/4 of proposals.
    assert cleanbench.coverage_at_precision(pairs, target_precision=0.98) == pytest.approx(0.75)
    # Unachievable target -> zero coverage, never a crash.
    assert cleanbench.coverage_at_precision([(0.9, False)], target_precision=0.98) == 0.0
    assert cleanbench.coverage_at_precision([], target_precision=0.98) == 0.0


@pytest.fixture(scope="module")
def phase3_run():
    model_runtime.set_encoder_factory(lambda model_id: cleanbench.BigramStubEncoder())
    try:
        truth, corrupted, kwargs = cleanbench.make_phase3_embedding_fixture()
        repaired, report = fd.clean(corrupted, return_report=True, **kwargs)
        yield truth, corrupted, repaired, report
    finally:
        model_runtime.set_encoder_factory(None)


def test_phase3_release_gates(phase3_run):
    truth, corrupted, repaired, report = phase3_run
    pcv = cleanbench.protected_column_violation_rate(
        corrupted, repaired, ["monthly_revenue"]
    )
    assert pcv == cleanbench.GATE_PROTECTED_VIOLATION_RATE == 0.0
    fmr = cleanbench.false_modification_rate(truth, corrupted, repaired)
    assert fmr <= cleanbench.GATE_FALSE_MODIFICATION_RATE
    pairs = cleanbench.confidence_outcomes(report, truth, corrupted)
    assert pairs, "the mini-suite must produce scored semantic outcomes"
    assert cleanbench.expected_calibration_error(pairs) <= cleanbench.GATE_ECE
    assert (
        cleanbench.precision_at_confidence_bucket(pairs, floor=0.95)
        >= cleanbench.GATE_PRECISION_AT_95
    )
    assert cleanbench.coverage_at_precision(pairs, target_precision=0.98) > 0.0


def test_phase3_embedding_rescues_with_provenance(phase3_run):
    truth, corrupted, repaired, report = phase3_run
    embedding_actions = [
        a for a in report.actions if a.model_id == "semantic:reference_value:embedding"
    ]
    assert {a.metadata["raw_value"] for a in embedding_actions} == {"activvee", "penddingg"}
    for action in embedding_actions:
        assert action.status == "automatic"
        assert action.metadata["backend"] == "embedding"
        assert action.metadata["calibration_version"] == "calib-default-2"
        assert action.metadata["model_evidence"]["model_id"] == "fd-col-encoder-v1"
        assert 0.95 <= action.metadata["calibrated_confidence"] < 1.0
    assert repaired["status"].tolist()[0] == "active"
    assert repaired["status"].tolist()[2] == "pending"


def test_phase3_ambiguous_value_never_auto_applied(phase3_run):
    truth, corrupted, repaired, report = phase3_run
    # "nactive" is one edit from both allowed values: nothing may repair it.
    assert repaired["status"].tolist()[3] == "nactive"
    applied = [
        a
        for a in report.actions
        if a.step == "semantic"
        and a.status == "automatic"
        and a.metadata.get("raw_value") == "nactive"
    ]
    assert applied == []
