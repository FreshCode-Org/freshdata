"""Public CleanBench release: baselines, reproducibility, and README audit.

Covers what tests/test_cleanbench_full.py doesn't: the baseline harness
(pandas/pyjanitor/Great Expectations/disclosed LLM-agent), the new metrics
(network_call_count, determinism_score, cost_usd_per_1m_rows), and the
reproducibility tooling (--verify-results / audit-readme).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(BENCH_DIR))

cleanbench = pytest.importorskip("cleanbench")

from cleanbench import metrics as cb_metrics  # noqa: E402
from cleanbench import reproducibility  # noqa: E402
from cleanbench import runner as cb_runner  # noqa: E402
from cleanbench.baselines import BASELINES  # noqa: E402
from cleanbench.baselines import run_all as run_all_baselines  # noqa: E402

# --------------------------------------------------------------------------- #
# new metrics
# --------------------------------------------------------------------------- #

class TestNewMetrics:
    def test_runtime_network_call_count_is_zero(self):
        assert cb_metrics.runtime_network_call_count() == 0

    def test_determinism_score_all_agree(self):
        assert cb_metrics.determinism_score([0.9, 0.9, 0.9]) == 1.0

    def test_determinism_score_partial_agreement(self):
        assert cb_metrics.determinism_score([0.9, 0.9, 0.5]) == pytest.approx(2 / 3)

    def test_determinism_score_empty(self):
        assert cb_metrics.determinism_score([]) == 0.0

    def test_cost_usd_per_1m_rows_scales(self):
        assert cb_metrics.cost_usd_per_1m_rows(1.0, 1000) == 1000.0

    def test_cost_usd_per_1m_rows_none_when_unknown(self):
        assert cb_metrics.cost_usd_per_1m_rows(None, 1000) is None
        assert cb_metrics.cost_usd_per_1m_rows(1.0, 0) is None


# --------------------------------------------------------------------------- #
# baseline harness
# --------------------------------------------------------------------------- #

class TestBaselines:
    def test_run_all_covers_every_registered_baseline(self):
        results = run_all_baselines()
        assert set(results) == set(BASELINES)

    def test_pandas_baseline_runs_and_scores(self):
        results = run_all_baselines()
        pandas_result = results["pandas"]
        assert pandas_result["status"] == "ran"
        assert pandas_result["cell_repair_f1"] is not None
        assert pandas_result["network_call_count"] == 0

    def test_pyjanitor_skips_gracefully_when_unavailable(self):
        results = run_all_baselines()
        pyjanitor_result = results["pyjanitor"]
        assert pyjanitor_result["status"] in ("ran", "skipped")
        if pyjanitor_result["status"] == "skipped":
            assert "reason" in pyjanitor_result and pyjanitor_result["reason"]

    def test_great_expectations_separates_validation_from_repair(self):
        results = run_all_baselines()
        ge_result = results["great_expectations"]
        assert ge_result["status"] == "ran"
        assert ge_result["cell_repair_f1"] is None  # never fabricated
        assert ge_result["cells_failing"] >= 0

    def test_llm_agent_skipped_by_default(self, monkeypatch):
        monkeypatch.delenv("FRESHDATA_LLM_BASELINE", raising=False)
        results = run_all_baselines()
        llm_result = results["llm_agent"]
        assert llm_result["status"] == "skipped"
        assert llm_result["network_call_count"] == 0
        assert llm_result["cost_usd"] is None


# --------------------------------------------------------------------------- #
# reproducibility: environment + hashing
# --------------------------------------------------------------------------- #

class TestEnvironmentAndHashing:
    def test_environment_info_has_required_keys(self):
        env = reproducibility.environment_info()
        for key in ("freshdata_version", "git_commit", "python_version", "platform"):
            assert env.get(key), key

    def test_dataset_hashes_are_stable_across_calls(self):
        # Fixtures are seeded/deterministic: two calls must hash identically.
        assert reproducibility.dataset_hashes() == reproducibility.dataset_hashes()

    def test_frame_hash_changes_with_content(self):
        import pandas as pd

        a = pd.DataFrame({"x": [1, 2, 3]})
        b = pd.DataFrame({"x": [1, 2, 4]})
        assert reproducibility.frame_hash(a) != reproducibility.frame_hash(b)


# --------------------------------------------------------------------------- #
# verify-results
# --------------------------------------------------------------------------- #

class TestVerifyResults:
    def _fresh_result(self, tmp_path: Path) -> Path:
        result = cb_runner.run_full(("T1",))
        path = tmp_path / "result.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        return path

    def test_verify_passes_on_fresh_result(self, tmp_path):
        path = self._fresh_result(tmp_path)
        assert reproducibility.verify_results(path) == []

    def test_verify_fails_on_missing_file(self, tmp_path):
        failures = reproducibility.verify_results(tmp_path / "nope.json")
        assert failures and "does not exist" in failures[0]

    def test_verify_fails_on_invalid_json(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        failures = reproducibility.verify_results(path)
        assert failures and "not valid JSON" in failures[0]

    def test_verify_fails_on_missing_required_key(self, tmp_path):
        path = tmp_path / "incomplete.json"
        path.write_text(json.dumps({"tracks_run": ["T1"]}), encoding="utf-8")
        failures = reproducibility.verify_results(path)
        assert any("missing required top-level key" in f for f in failures)

    def test_verify_fails_on_tampered_dataset_hash(self, tmp_path):
        result = cb_runner.run_full(("T1",))
        result["dataset_hashes"]["T1"] = "not-the-real-hash"
        path = tmp_path / "tampered.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        failures = reproducibility.verify_results(path)
        assert any("dataset hash mismatch" in f for f in failures)

    def test_verify_fails_on_inconsistent_gate_verdict(self, tmp_path):
        result = cb_runner.run_full(("T1",))
        result["release_gates"] = {"failures": ["something broke"], "passed": True}
        path = tmp_path / "inconsistent.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        failures = reproducibility.verify_results(path)
        assert any("inconsistent" in f for f in failures)


# --------------------------------------------------------------------------- #
# README claim audit
# --------------------------------------------------------------------------- #

class TestReadmeAudit:
    def test_audit_readme_passes_on_real_repo(self):
        assert reproducibility.audit_readme() == []

    def test_audit_fails_when_claim_text_removed(self, tmp_path, monkeypatch):
        stub_readme = tmp_path / "README.md"
        stub_readme.write_text("nothing to see here", encoding="utf-8")
        monkeypatch.setattr(reproducibility, "README_PATH", stub_readme)
        failures = reproducibility.audit_readme()
        assert len(failures) == len(reproducibility.CLAIM_REGISTRY)
        assert all("no longer in README.md" in f for f in failures)

    def test_audit_fails_when_backing_test_missing(self, tmp_path, monkeypatch):
        stub_readme = tmp_path / "README.md"
        claim_text = "totally-fake-claim-for-this-test"
        stub_readme.write_text(claim_text, encoding="utf-8")
        monkeypatch.setattr(reproducibility, "README_PATH", stub_readme)
        monkeypatch.setattr(
            reproducibility, "CLAIM_REGISTRY",
            (reproducibility.Claim(claim_text, ("tests/does_not_exist.py::test_nope",)),),
        )
        failures = reproducibility.audit_readme()
        assert any("backing test file missing" in f for f in failures)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

class TestCLI:
    def test_reproduce_headline_writes_baselines_section(self, tmp_path, monkeypatch):
        from cleanbench import report as cb_report

        monkeypatch.setattr(cb_report, "RESULTS_DIR", tmp_path)
        result = cb_runner.run_full(("T1",), include_baselines=True)
        assert "baselines" in result
        assert set(result["baselines"]) == set(BASELINES)

    def test_main_verify_results_nonzero_on_failure(self, tmp_path, capsys):
        from cleanbench.__main__ import main as cli_main

        bad = tmp_path / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        code = cli_main(["--verify-results", str(bad)])
        assert code == 1
        captured = capsys.readouterr()
        assert "FAIL:" in captured.err

    def test_main_verify_results_zero_on_success(self, tmp_path, capsys):
        from cleanbench.__main__ import main as cli_main

        result = cb_runner.run_full(("T1",))
        path = tmp_path / "good.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        code = cli_main(["--verify-results", str(path)])
        assert code == 0
        captured = capsys.readouterr()
        assert "VERIFIED" in captured.out
