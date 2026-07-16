"""Runner orchestration, minimizer, artifact writing, and CLI contracts.

The full 8-domain release run is exercised by the release command itself;
these tests pin the component contracts on a single domain so they stay fast.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pandas as pd
import pytest
from benchmarks.truthbench import cli
from benchmarks.truthbench import runner as runner_module
from benchmarks.truthbench.minimize import minimize_failure
from benchmarks.truthbench.models import GateResult, RunResult
from benchmarks.truthbench.report import compare_to_baseline, write_artifacts
from benchmarks.truthbench.runner import (
    TruthBenchRunError,
    parity_fixture,
    run_release,
)
from benchmarks.truthbench.schema import validate_run
from benchmarks.truthbench.surfaces import backends as backends_module
from benchmarks.truthbench.surfaces.backends import BackendUnavailableError


@pytest.fixture(scope="module")
def finance_outcome(tmp_path_factory):
    results = tmp_path_factory.mktemp("tb-results")
    return run_release(domains=("finance",), results_dir=results), results


def test_release_run_produces_complete_graded_evidence(finance_outcome):
    outcome, _ = finance_outcome
    records = outcome.run.records
    assert len(records) > 0
    surfaces = {record.surface for record in records}
    assert surfaces == {"cleaning", "backend_parity"}
    backends = {record.requested_backend for record in records}
    assert backends == {"pandas", "polars", "duckdb"}
    repeats = {record.repeat for record in records}
    assert repeats == {0, 1}
    # every record participates in determinism verification
    assert all(record.repeat_hash for record in records)
    # gate evaluation ran for both contexts plus the runner-level gates
    names = {gate.name for gate in outcome.run.gates}
    assert "cleaning:completeness" in names
    assert "parity:required_backend" in names
    assert "backend_parity_comparison" in names
    assert "generated_code_sandbox" in names
    validate_run(outcome.run.to_dict())


def test_release_run_grades_every_finance_cell_against_its_oracle(finance_outcome):
    outcome, _ = finance_outcome
    # Grading actually happened: every labelled finance cell produced a
    # decision record on every backend/repeat, and the value-semantic gates
    # were evaluated (not skipped). The finance oracle is now coherent, so a
    # correct product run passes it — the proof that grading ran is the
    # presence of graded records and evaluated gates, not a planted failure.
    finance_records = [
        r for r in outcome.run.records
        if r.domain == "finance" and r.surface == "cleaning"
    ]
    assert finance_records
    graded_gates = {g.name for g in outcome.run.gates}
    assert "cleaning:valid_value_corruption" in graded_gates
    assert "cleaning:review_routing" in graded_gates
    # A preserve-labelled valid value must never be reported as corrupted.
    corruption = next(
        g for g in outcome.run.gates if g.name == "cleaning:valid_value_corruption"
    )
    assert corruption.passed, corruption.failures


def test_release_run_writes_schema_valid_artifacts(finance_outcome):
    outcome, results = finance_outcome
    latest = json.loads((results / "latest.json").read_text())
    validate_run(latest)
    # The finance oracle is coherent and the product handles it correctly, so
    # this single-domain run passes — the artifact is written atomically and
    # validates either way.
    assert isinstance(latest["summary"]["overall_passed"], bool)
    assert (results / "latest.md").read_text().startswith("# TruthBench run")
    for case in outcome.failures:
        path = results / "failures" / f"{case.failure_id}.json"
        assert path.is_file()
        payload = json.loads(path.read_text())
        assert payload["reproduce_command"].startswith("PYTHONPATH=src")
        assert "example.invalid" not in path.read_text()


def test_release_run_requires_two_repeats_and_domains():
    with pytest.raises(TruthBenchRunError, match="two repeats"):
        run_release(repeats=1, domains=("finance",), write=False)
    with pytest.raises(TruthBenchRunError, match="domain"):
        run_release(domains=(), write=False)


def test_missing_required_backend_is_an_infrastructure_failure(monkeypatch):
    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(backends_module.metadata, "version", missing)
    with pytest.raises(BackendUnavailableError):
        run_release(domains=("finance",), write=False)


def test_minimizer_never_removes_the_target_cell():
    fixture = parity_fixture()
    target = next(c for c in fixture.cells if c.row_id == "par-01" and c.column == "name")
    evaluated: list[pd.DataFrame] = []

    def still_fails(frame: pd.DataFrame) -> bool:
        evaluated.append(frame)
        return True  # everything reproduces: maximal reduction pressure

    case = minimize_failure(
        fixture,
        cell_id=target.cell_id,
        gate="exact_repair",
        surface="cleaning",
        expected="alpha",
        actual="'  alpha '",
        component="freshdata.clean",
        still_fails=still_fails,
        budget=15,
    )
    assert case.evaluations <= 15
    for frame in evaluated:
        assert "par-01" in frame.index
        assert "name" in frame.columns
    rows = {row["row_id"] for row in case.frame_records}
    assert "par-01" in rows
    assert len(rows) == 1  # background rows reduced away


def test_baseline_comparison_is_regression_evidence_only(tmp_path):
    run = RunResult(
        run_id="r",
        profile="release",
        fixture_hashes=(("parity", "h"),),
        required_backends=("pandas",),
        records=(),
        gates=(GateResult("cleaning:exact_repair", False, ("x",)),),
        summary=(("records", 0), ("overall_passed", False)),
        environment=(("python", "3"),),
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"gates": [{"name": "cleaning:exact_repair", "passed": True}]})
    )
    notes = compare_to_baseline(run, baseline)
    assert any("regression" in note for note in notes)


def test_artifact_writer_rejects_schema_invalid_runs(tmp_path):
    bogus = RunResult(
        run_id="r",
        profile="release",
        fixture_hashes=(("demo", "h"),),
        required_backends=("pandas",),
        records=(),  # schema demands at least one record
        gates=(GateResult("g", True, ()),),
        summary=(("records", 0), ("overall_passed", True)),
        environment=(("python", "3"),),
    )
    with pytest.raises(Exception, match="schema|records"):
        write_artifacts(bogus, results_dir=tmp_path)
    assert not (tmp_path / "latest.json").exists()


def test_cli_reports_infrastructure_failures_as_exit_2(monkeypatch, capsys):
    def boom(**_kwargs):
        raise TruthBenchRunError("no evidence")

    monkeypatch.setattr(runner_module, "run_release", boom)
    code = cli.main(
        ["run", "--profile", "release", "--backends", "pandas", "--check"]
    )
    assert code == 2
    assert "INFRASTRUCTURE FAILURE" in capsys.readouterr().err


def test_cli_regression_ratchet_passes_on_known_red_and_fails_on_regression(
    monkeypatch, capsys, tmp_path
):
    """--check-regressions: known-red gates (already failing in the committed
    baseline) do not fail a PR, a newly-failing gate does, and the release
    --check stays absolute."""
    gates = (
        GateResult("cleaning:known_red", False, ("documented blocker",)),
        GateResult("cleaning:green", True, ()),
    )
    run = RunResult(
        run_id="r",
        profile="release",
        fixture_hashes=(("parity", "h"),),
        required_backends=("pandas",),
        records=(),
        gates=gates,
        summary=(("records", 0), ("overall_passed", False)),
        environment=(("python", "3"),),
    )

    def fake_run_release(**kwargs):
        return SimpleNamespace(
            run=run,
            passed=False,
            failures=(),
            artifacts={},
            baseline_notes=tuple(kwargs.pop("_notes", ())) or notes,
            generated_code_results=(),
        )

    # Case 1: no regressions — ratchet passes, absolute check fails.
    notes = ("no gate-level changes vs baseline",)
    monkeypatch.setattr(runner_module, "run_release", fake_run_release)
    assert cli.main(["run", "--check-regressions"]) == 0
    out = capsys.readouterr().out
    assert "KNOWN-RED" in out
    assert cli.main(["run", "--check"]) == 1

    # Case 2: a regression — ratchet fails.
    notes = ("regression: gate cleaning:green passed in baseline, fails now",)
    assert cli.main(["run", "--check-regressions"]) == 1
    assert "REGRESSION" in capsys.readouterr().err
