"""CleanBench full suite: T1-T5 tracks, metrics, release gates, site report."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(BENCH_DIR))

cleanbench = pytest.importorskip("cleanbench")

from cleanbench import fixtures  # noqa: E402
from cleanbench import metrics as cb_metrics  # noqa: E402
from cleanbench import report as cb_report  # noqa: E402
from cleanbench import runner as cb_runner  # noqa: E402
from cleanbench import tasks as cb_tasks  # noqa: E402


class TestFixturesLoad:
    def test_t1_loads(self):
        truth, corrupted, kwargs = fixtures.make_t1_representation_fixture()
        assert not truth.empty and not corrupted.empty
        assert isinstance(kwargs, dict)

    def test_t2_loads(self):
        truth, corrupted, kwargs = fixtures.make_t2_semantic_fixture()
        assert not truth.empty

    def test_t3_loads(self):
        truth, corrupted, kwargs = fixtures.make_t3_context_fixture()
        assert "context" in kwargs

    def test_t4_loads(self):
        result = fixtures.make_t4_profile_fixture()
        assert len(result) == 8

    def test_t5_loads_small(self):
        truth, corrupted, kwargs = fixtures.make_t5_scale_fixture(target_rows=200)
        assert len(corrupted) == 200
        assert corrupted["cust_id"].is_unique


class TestTaskDirectories:
    def test_build_all_writes_five_tasks(self, tmp_path):
        written = cb_tasks.build_all(tmp_path)
        assert len(written) == 5
        for directory in written:
            assert (directory / "meta.json").is_file()
            assert (directory / "context.txt").is_file()

    def test_load_task_roundtrip(self, tmp_path):
        [directory] = [d for d in cb_tasks.build_all(tmp_path) if d.name == "t2_semantic_values"]
        task = cb_tasks.load_task(directory)
        assert task.track == "T2"
        assert task.truth is not None
        assert task.corrupted is not None
        assert "context" not in task.clean_kwargs


class TestMetricsCompute:
    def test_all_tracks_compute_without_error(self):
        result = cb_runner.run_full(("T1", "T2", "T3", "T4"))
        assert set(result["tracks"]) == {"T1", "T2", "T3", "T4"}
        for values in result["tracks"].values():
            assert isinstance(values, dict) and values

    def test_t5_computes_perf_metrics(self):
        result = cb_runner.run_t5(target_rows=500, repeats=1)
        assert result["rows"] == 500
        assert result["speed_rows_per_sec"] > 0
        assert result["peak_rss_delta_bytes"] >= 0

    def test_explainability_rubric_in_range(self):
        truth, corrupted, kwargs = fixtures.make_t2_semantic_fixture()
        import freshdata as fd

        _, report = fd.clean(corrupted, return_report=True, **kwargs)
        score = cb_metrics.explainability_rubric_score(report)
        assert 0.0 <= score <= 1.0


class TestReleaseGates:
    def test_gates_pass_on_healthy_tracks(self):
        healthy = {
            "T1": {"protected_column_violation_rate": 0.0, "false_modification_rate": 0.0},
            "T2": {"protected_column_violation_rate": 0.0, "false_modification_rate": 0.0,
                  "confidence_ece": 0.01, "precision_at_conf_95": 1.0},
            "T4": {"protected_column_violation_rate": 0.0, "false_modification_rate": 0.0,
                  "profile_fmr_non_increase": True, "privacy_leak_count": 0},
        }
        assert cb_runner.check_release_gates(healthy) == []

    def test_gates_fail_on_protected_violation(self):
        broken = {"T1": {"protected_column_violation_rate": 1.0, "false_modification_rate": 0.0}}
        failures = cb_runner.check_release_gates(broken)
        assert any("protected-column" in f for f in failures)

    def test_gates_fail_on_false_modification_rate(self):
        broken = {"T2": {"protected_column_violation_rate": 0.0, "false_modification_rate": 0.5}}
        failures = cb_runner.check_release_gates(broken)
        assert any("false modification" in f for f in failures)

    def test_gates_fail_on_bad_ece(self):
        broken = {"T2": {"protected_column_violation_rate": 0.0, "false_modification_rate": 0.0,
                         "confidence_ece": 0.5, "precision_at_conf_95": 1.0}}
        failures = cb_runner.check_release_gates(broken)
        assert any("ECE" in f for f in failures)

    def test_gates_fail_on_profile_fmr_increase(self):
        broken = {"T4": {"protected_column_violation_rate": 0.0, "false_modification_rate": 0.0,
                         "profile_fmr_non_increase": False}}
        failures = cb_runner.check_release_gates(broken)
        assert any("profile replay" in f for f in failures)

    def test_gates_fail_on_privacy_leak(self):
        broken = {"T4": {"protected_column_violation_rate": 0.0, "false_modification_rate": 0.0,
                         "profile_fmr_non_increase": True, "privacy_leak_count": 3}}
        failures = cb_runner.check_release_gates(broken)
        assert any("privacy leak" in f for f in failures)

    def test_gates_fail_on_perf_regression(self):
        broken = {"T5": {"runtime_slowdown_vs_baseline": 0.5}}
        failures = cb_runner.check_release_gates(broken)
        assert any("slowdown" in f for f in failures)


class TestGateFatality:
    """Runtime/memory perf gates are only fatal under an explicit --check-gates
    (the perf-regression workflow, which runs T5 in isolation). In the full
    T1-T5 suite they are reported but do not fail the run."""

    def test_classifier_separates_perf_from_correctness(self):
        assert cb_runner.is_perf_gate_failure("runtime slowdown 0.5 > 20% vs baseline")
        assert cb_runner.is_perf_gate_failure("memory overhead 0.3 > 15% vs baseline")
        assert not cb_runner.is_perf_gate_failure("protected-column violation rate 1.0 > 0")
        assert not cb_runner.is_perf_gate_failure("privacy leak count 3 > 0")

    def test_run_full_payload_splits_perf_and_correctness(self):
        # A T5-only run whose sole failure is a perf gate: the full-suite path
        # (reproduce_headline) must not treat it as fatal.
        from cleanbench.__main__ import _fatal_gate_failures

        gates = {
            "failures": ["runtime slowdown 0.5 > 20% vs baseline"],
            "passed": False,
            "correctness_failures": [],
            "perf_failures": ["runtime slowdown 0.5 > 20% vs baseline"],
        }
        assert _fatal_gate_failures(gates, check_gates=False) == []      # full suite: green
        assert _fatal_gate_failures(gates, check_gates=True) != []       # perf-regression: red

    def test_correctness_failures_are_always_fatal(self):
        from cleanbench.__main__ import _fatal_gate_failures

        gates = {
            "failures": ["privacy leak count 3 > 0"],
            "passed": False,
            "correctness_failures": ["privacy leak count 3 > 0"],
            "perf_failures": [],
        }
        assert _fatal_gate_failures(gates, check_gates=False) != []
        assert _fatal_gate_failures(gates, check_gates=True) != []

    def test_falls_back_to_flat_failures_for_legacy_results(self):
        from cleanbench.__main__ import _fatal_gate_failures

        legacy = {"failures": ["runtime slowdown 0.5 > 20% vs baseline"], "passed": False}
        # No split keys: stay strict (fatal) rather than silently dropping.
        assert _fatal_gate_failures(legacy, check_gates=False) != []


class TestSiteReport:
    def test_write_results_and_docs(self, tmp_path):
        result = cb_runner.run_full(("T1",))
        json_path, md_path = cb_report.write_results(result, results_dir=tmp_path)
        assert json_path.is_file() and md_path.is_file()
        docs_path = cb_report.update_docs_site(result, docs_path=tmp_path / "benchmarks.md")
        text = docs_path.read_text(encoding="utf-8")
        assert cb_report._BLOCK_BEGIN in text and cb_report._BLOCK_END in text

    def test_update_docs_site_replaces_existing_block(self, tmp_path):
        docs_path = tmp_path / "benchmarks.md"
        docs_path.write_text("# Benchmarks\n\nsome hand-written prose.\n")
        result = cb_runner.run_full(("T1",))
        cb_report.update_docs_site(result, docs_path=docs_path)
        first = docs_path.read_text(encoding="utf-8")
        cb_report.update_docs_site(result, docs_path=docs_path)
        second = docs_path.read_text(encoding="utf-8")
        assert first.count(cb_report._BLOCK_BEGIN) == 1
        assert second.count(cb_report._BLOCK_BEGIN) == 1
        assert "hand-written prose" in second
