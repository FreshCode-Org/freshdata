"""Optional encoder contrastive distillation: safety gates independent of torch."""

from __future__ import annotations

from training.distill import train_encoder_contrastive as tec


def test_build_pairs_includes_dangerous_negatives():
    pairs = tec.build_pairs()
    assert len(pairs["dangerous_negatives"]) >= len(tec.DANGEROUS_NEGATIVES)
    assert ("Austria", "Australia") in pairs["dangerous_negatives"]


def test_build_pairs_includes_alias_positives():
    pairs = tec.build_pairs()
    assert pairs["positives"]


def test_false_merge_rate_is_zero_for_baseline_encoder():
    encoder = tec.HashedBigramEncoder()
    rate = tec.false_merge_rate(encoder, list(tec.DANGEROUS_NEGATIVES))
    assert rate == 0.0


def test_false_merge_rate_flags_identical_strings():
    encoder = tec.HashedBigramEncoder()
    rate = tec.false_merge_rate(encoder, [("active", "active")])
    assert rate == 1.0


def test_check_gates_fails_on_nonzero_false_merge():
    result = {
        "resolver_accuracy": 0.9, "baseline_resolver_accuracy": 0.9,
        "dangerous_negative_false_merge_rate": 0.2,
    }
    failures = tec._check_gates(result)
    assert any("false-merge" in f for f in failures)


def test_check_gates_fails_on_regression():
    result = {
        "resolver_accuracy": 0.5, "baseline_resolver_accuracy": 0.9,
        "dangerous_negative_false_merge_rate": 0.0,
    }
    failures = tec._check_gates(result)
    assert any("regressed" in f for f in failures)


def test_check_gates_passes_on_good_metrics():
    result = {
        "resolver_accuracy": 0.95, "baseline_resolver_accuracy": 0.9,
        "dangerous_negative_false_merge_rate": 0.0,
    }
    assert tec._check_gates(result) == []


def test_torch_unavailable_path_is_skipped_not_failed(tmp_path):
    # This dev environment has no torch installed; verify the documented
    # "skipped" outcome rather than a hard failure.
    if tec.torch_available():
        return
    result = tec.train(out_dir=tmp_path)
    assert result["status"] == "skipped"
    assert result["dangerous_negative_false_merge_rate"] == 0.0
    assert result["gates"]["failures"] == []
    assert (tmp_path / "encoder_contrastive.metrics.json").is_file()
