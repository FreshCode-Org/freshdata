"""Trust score must strictly decrease as injected defect rate rises (Metric 8)."""

from __future__ import annotations

import harness_metrics as hm
import pytest


@pytest.mark.parametrize("name", ["crm", "finance", "event_log"])
def test_trust_strictly_monotonic(name):
    result = hm.metric_trust(name, sweep_size=2_000, seed=42)
    assert result["trust_monotonic_valid"] is True, (name, result["sweep"])


@pytest.mark.parametrize("name", ["crm", "finance", "event_log"])
def test_trust_sweep_covers_all_rates(name):
    result = hm.metric_trust(name, sweep_size=2_000, seed=42)
    rates = {str(r) for r in hm.TRUST_DEFECT_RATES}
    assert set(result["sweep"]) == rates
    scores = [result["sweep"][str(r)] for r in hm.TRUST_DEFECT_RATES]
    # strict decrease, verified explicitly
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)), scores


def test_gold_trust_monotonic():
    result = hm.metric_trust("gold", sweep_size=2_000, seed=42)
    assert result["trust_monotonic_valid"] is True, result["sweep"]
