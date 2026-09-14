"""Same-run A/B performance gate: ordering, statistics, confirmation, CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(BENCH_DIR))

pytest.importorskip("cleanbench")

from cleanbench import ab  # noqa: E402

MIB = 1024 * 1024


def _worker(timings):
    """Fake worker: each call pops the next ``(seconds, rss)`` for that side."""
    calls: list[str] = []
    queues = {side: list(values) for side, values in timings.items()}

    def worker(src, *, target_rows, warmups, runs):
        calls.append(src)
        seconds, rss = queues[src].pop(0)
        return {"seconds": seconds, "peak_rss_delta_bytes": rss, "freshdata_path": src}

    return worker, calls


def _steady(seconds, rss=64 * MIB, count=20):
    return [([seconds, seconds * 1.05], rss)] * count


def test_sides_are_interleaved_with_alternating_order():
    worker, calls = _worker({"base": _steady(1.0), "head": _steady(1.0)})
    ab.measure("base", "head", pairs=3, target_rows=10, warmups=0, runs=2, worker=worker)
    assert calls == ["base", "head", "head", "base", "base", "head"]


def test_summary_uses_fastest_run_and_median_rss():
    sides = {
        "base": ab.SideSamples(seconds=[1.2, 1.0, 3.0], rss_deltas=[10 * MIB, 50 * MIB, 12 * MIB]),
        "head": ab.SideSamples(seconds=[1.1, 1.3, 9.0], rss_deltas=[11 * MIB, 13 * MIB, 90 * MIB]),
    }
    summary = ab.summarize(sides)
    assert summary["runtime_slowdown"] == pytest.approx(0.10, abs=1e-4)  # 1.1 / 1.0
    assert summary["memory_overhead"] == pytest.approx(13 / 12 - 1, abs=1e-4)


def test_memory_gate_not_applicable_for_tiny_base_delta():
    sides = {
        "base": ab.SideSamples(seconds=[1.0], rss_deltas=[MIB]),
        "head": ab.SideSamples(seconds=[1.0], rss_deltas=[10 * MIB]),
    }
    summary = ab.summarize(sides)
    assert summary["memory_overhead"] is None
    assert ab.gate_failures(summary) == {}


def test_within_threshold_passes_without_confirmation():
    worker, calls = _worker({"base": _steady(1.0), "head": _steady(1.15)})
    result = ab.run_ab("base", "head", pairs=2, worker=worker)
    assert result["passed"]
    assert result["confirmation"] is None
    assert len(calls) == 4


def test_breach_that_does_not_reproduce_passes():
    head = _steady(1.5, count=2) + _steady(1.02, count=2)
    worker, calls = _worker({"base": _steady(1.0), "head": head})
    result = ab.run_ab("base", "head", pairs=2, worker=worker)
    assert result["passed"], result["failures"]
    assert result["measurement"]["runtime_slowdown"] > 0.2
    assert result["confirmation"]["runtime_slowdown"] < 0.2
    assert len(calls) == 8


def test_reproduced_runtime_breach_fails():
    worker, _ = _worker({"base": _steady(1.0), "head": _steady(1.4)})
    result = ab.run_ab("base", "head", pairs=2, worker=worker)
    assert not result["passed"]
    [failure] = result["failures"]
    assert failure.startswith("runtime slowdown +40.0% > 20%")


def test_metric_must_breach_on_both_runs_to_fail():
    # First run breaches runtime only; confirmation breaches memory only.
    head = _steady(1.5, rss=64 * MIB, count=2) + _steady(1.0, rss=128 * MIB, count=2)
    worker, _ = _worker({"base": _steady(1.0, rss=64 * MIB), "head": head})
    result = ab.run_ab("base", "head", pairs=2, worker=worker)
    assert result["passed"], result["failures"]


def test_reproduced_memory_breach_fails():
    worker, _ = _worker({"base": _steady(1.0, rss=64 * MIB), "head": _steady(1.0, rss=96 * MIB)})
    result = ab.run_ab("base", "head", pairs=2, worker=worker)
    assert [f.split(" (")[0] for f in result["failures"]] == ["memory overhead +50.0% > 15%"]


@pytest.mark.parametrize(("check_gates", "expected_code"), [(True, 1), (False, 0)])
def test_cli_writes_results_and_summary(monkeypatch, tmp_path, check_gates, expected_code):
    worker, _ = _worker({"base": _steady(1.0), "head": _steady(1.4)})
    real_run_ab = ab.run_ab
    monkeypatch.setattr(ab, "run_ab", lambda *a, **k: real_run_ab(*a, **{**k, "worker": worker}))
    output = tmp_path / "ab.json"
    summary = tmp_path / "summary.md"
    argv = ["--base-src", "base", "--head-src", "head", "--pairs", "2",
            "--base-label", "main@abc1234", "--output", str(output), "--summary", str(summary)]
    if check_gates:
        argv.append("--check-gates")
    assert ab.main(argv) == expected_code
    payload = json.loads(output.read_text())
    assert payload["passed"] is False
    text = summary.read_text()
    assert "main@abc1234" in text
    assert "**FAIL**" in text


def test_real_worker_imports_freshdata_from_the_requested_src():
    src = Path(ab.REPO_ROOT) / "src"
    payload = ab._run_worker(src, target_rows=300, warmups=0, runs=1, timeout=300)
    assert Path(payload["freshdata_path"]).resolve().is_relative_to(src.resolve())
    assert len(payload["seconds"]) == 1


def test_real_worker_refuses_a_src_without_freshdata(tmp_path):
    with pytest.raises(RuntimeError, match="expected under"):
        ab._run_worker(tmp_path, target_rows=300, warmups=0, runs=1, timeout=300)
