"""CleanBench T4 gates: learned-profile replay (Phase 4).

T4 measures whether a profile learned from one messy/clean pair actually
helps on the *next* batch of the same corruption family — and, before any
lift is celebrated, that replay never makes the safety story worse:

- mean repair-F1 lift over the replayable families >= +15 points;
- per-fixture false-modification rate never exceeds the no-profile run;
- protected-column violation rate is exactly 0.0;
- zero raw sensitive literals in the serialized profile under mask;
- the drift gate opens for matched batches and blocks unrelated frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
from freshdata.learning import learn

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

t4 = pytest.importorskip("cleanbench.t4_profiles")

#: Spec gate: replaying a learned profile on the next batch of the same
#: corruption family must lift cell-repair F1 by at least 15 points on
#: average across the families designed to be replayable.
GATE_T4_MEAN_LIFT_F1 = 0.15

#: Families whose lift comes from literal value-map replay (never within
#: reach of the deterministic experts), so profile-attributed actions must
#: appear in the report.
_VALUE_MAP_FAMILIES = {"category_map_departments", "condition_abbreviations"}


@pytest.fixture(scope="module")
def t4_results():
    return t4.run_t4()


def test_t4_mean_f1_lift_gate(t4_results) -> None:
    lifted = [r for r in t4_results if r.expect_lift]
    assert lifted, "no replayable fixtures ran"
    mean_lift = sum(r.lift_f1 for r in lifted) / len(lifted)
    assert mean_lift >= GATE_T4_MEAN_LIFT_F1, (
        f"mean F1 lift {mean_lift:+.3f} below gate {GATE_T4_MEAN_LIFT_F1:+.3f}: "
        + ", ".join(f"{r.name}={r.lift_f1:+.3f}" for r in lifted)
    )


def test_t4_profile_never_hurts_f1(t4_results) -> None:
    for r in t4_results:
        assert r.profile_f1 >= r.baseline_f1 - 1e-9, (
            f"{r.name}: profile F1 {r.profile_f1:.3f} < baseline {r.baseline_f1:.3f}"
        )


def test_t4_false_modification_rate_gate(t4_results) -> None:
    for r in t4_results:
        assert r.profile_fmr <= r.baseline_fmr + 1e-9, (
            f"{r.name}: profile FMR {r.profile_fmr:.4f} > baseline {r.baseline_fmr:.4f}"
        )


def test_t4_protected_violation_gate(t4_results) -> None:
    for r in t4_results:
        assert r.protected_violation_rate == 0.0, r.name


def test_t4_privacy_leak_gate(t4_results) -> None:
    for r in t4_results:
        assert r.privacy_leaks == 0, r.name


def test_t4_replay_gate_open_for_matched_batches(t4_results) -> None:
    for r in t4_results:
        assert r.replay_ok, f"{r.name}: drift gate blocked a same-family batch"


def test_t4_value_map_families_record_profile_actions(t4_results) -> None:
    by_name = {r.name: r for r in t4_results}
    for name in _VALUE_MAP_FAMILIES:
        assert by_name[name].profile_actions > 0, (
            f"{name}: no profile-attributed actions in the report"
        )


def test_t4_drift_gate_blocks_unrelated_frame() -> None:
    fixture = t4.make_t4_fixtures()[0]
    profile = learn(
        fixture.train_messy,
        fixture.train_clean,
        context=fixture.context,
        key=fixture.key,
        **fixture.learn_kwargs,
    )
    unrelated = pd.DataFrame({"alpha": [1, 2, 3], "beta": ["x", "y", "z"]})
    _, rep = fd.clean(
        unrelated,
        profile=profile,
        semantic_mode="auto",
        verbose=False,
        return_report=True,
    )
    assert t4.profile_drift_block_rate([rep]) == 1.0
    assert t4.profile_drift_block_rate([]) == 0.0
