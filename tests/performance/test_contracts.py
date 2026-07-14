from __future__ import annotations

import json
from pathlib import Path

import yaml
from benchmarks.performance.cli import main
from benchmarks.performance.render import render_report

ROOT = Path(__file__).parents[2]


def test_ci_sized_matrix_completes_and_renders(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    assert (
        main(
            [
                "run",
                "--rows",
                "500",
                "--widths",
                "narrow,medium",
                "--configs",
                "default,conservative",
                "--report-modes",
                "false,true",
                "--warmups",
                "0",
                "--repetitions",
                "1",
                "--timeout",
                "120",
                "--output",
                str(raw),
            ]
        )
        == 0
    )
    results = [json.loads(path.read_text()) for path in raw.glob("*.json")]
    assert len(results) == 8
    assert all(item["status"] == "completed" for item in results)
    rendered = render_report(
        {
            "environment": results[0]["environment"],
            "results": results,
            "hypotheses": {},
        }
    )
    assert "500" in rendered
    assert "narrow" in rendered and "medium" in rendered


def test_report_modes_preserve_behavioral_contract(tmp_path: Path) -> None:
    assert (
        main(
            [
                "run",
                "--rows",
                "250",
                "--widths",
                "narrow",
                "--configs",
                "default",
                "--report-modes",
                "false,true",
                "--warmups",
                "0",
                "--repetitions",
                "1",
                "--timeout",
                "120",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    results = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert {item["case"]["return_report"] for item in results} == {False, True}


def test_performance_artifact_ignore_contract() -> None:
    rules = (ROOT / ".gitignore").read_text()
    assert "benchmarks/results/*" in rules
    assert "!benchmarks/results/performance/" in rules
    assert "benchmarks/results/performance/*" in rules
    assert "!benchmarks/results/performance/*-summary.json" in rules
    assert "!benchmarks/results/performance/*-report.md" in rules
    assert "benchmarks/results/" not in rules.splitlines()


def test_make_targets_use_the_performance_cli() -> None:
    makefile = (ROOT / "Makefile").read_text()
    assert "performance-ci:" in makefile
    assert "$(PY) -m pytest tests/performance -q --no-cov" in makefile
    assert "$(PY) -m benchmarks.performance run --output" in makefile
    assert (
        "$(PY) -m benchmarks.performance profile --rows 100000 --widths medium "
        "--configs default --report-modes true" in makefile
    )
    assert "$(PY) -m benchmarks.performance analyze --input" in makefile
    assert "$(PY) -m benchmarks.performance render --input" in makefile


def test_large_workflow_is_scheduled_manual_and_preserves_failures() -> None:
    workflow_path = ROOT / ".github/workflows/performance-large.yml"
    workflow_text = workflow_path.read_text()
    workflow = yaml.safe_load(workflow_text)

    # PyYAML 1.1 treats the unquoted GitHub Actions key `on` as a boolean.
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert len(triggers["schedule"]) == 1
    assert workflow["jobs"]["performance-large"]["timeout-minutes"] == 180
    assert 'pip install -e ".[dev,bench,ml]"' in workflow_text
    for expected in (
        "--rows 100000,500000,1000000",
        "--widths narrow,medium,wide",
        "--configs default,conservative,representation_off,statistical_off,explicit",
        "--report-modes false,true",
        "--warmups 1",
        "--repetitions 5",
    ):
        assert expected in workflow_text
    assert "actions/upload-artifact@" in workflow_text
    assert "if: always()" in workflow_text
    assert "continue-on-error: true" in workflow_text
    assert "exit 1" in workflow_text
