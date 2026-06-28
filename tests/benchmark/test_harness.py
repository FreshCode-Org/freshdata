"""Harness: bench.py run completes, emits all nine metrics, validates schema."""

from __future__ import annotations

import bench
import jsonschema
import pytest
from results_schema import METRIC_FIELDS, RESULTS_SCHEMA


@pytest.fixture(scope="module")
def gold_result():
    # gold at the canonical 10k-row variant is fast (~0.15s clean) and the
    # strictest fixture, so it doubles as the "run completes on 10k" check.
    return bench.run_single("gold", 10_000, seed=42, repeat=2, sweep_size=1_500)


def test_run_single_completes(gold_result):
    assert gold_result["fixture"] == "gold"
    assert gold_result["n_rows"] >= 10_000
    assert gold_result["mode"] == "balanced"


def test_all_nine_metrics_present_and_non_null(gold_result):
    metrics = gold_result["metrics"]
    for field in METRIC_FIELDS:
        assert field in metrics, f"missing metric {field}"
        assert metrics[field] is not None, f"null metric {field}"


def test_result_validates_against_schema(gold_result):
    jsonschema.validate(gold_result, RESULTS_SCHEMA)


@pytest.mark.parametrize("name", ["crm", "event_log", "provenance"])
def test_frame_fixture_results_validate(name):
    result = bench.run_single(name, 3_000, seed=42, repeat=2, sweep_size=1_500)
    jsonschema.validate(result, RESULTS_SCHEMA)
    m = result["metrics"]
    assert all(m[f] is not None for f in METRIC_FIELDS)


def test_safety_metrics_pass_on_gold(gold_result):
    m = gold_result["metrics"]
    assert m["false_repair_rate_pct"] == 0.0
    assert m["preservation_rate_pct"] == 100.0
    assert m["repair_fidelity_pct"] >= 90.0
    assert m["export_completeness_pct"] == 100.0
    assert m["trust_monotonic_valid"] is True
