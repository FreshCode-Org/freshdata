from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from benchmarks.performance.cli import main
from benchmarks.performance.models import BenchmarkCase
from benchmarks.performance.runner import expand_cases, run_case_subprocess, run_matrix
from benchmarks.performance.schema import validate_result


def _case(**overrides: object) -> BenchmarkCase:
    arguments = {
        "rows": 100,
        "width": "narrow",
        "config_name": "default",
        "options": {"verbose": False},
        "warmups": 0,
        "repetitions": 1,
    }
    arguments.update(overrides)
    return BenchmarkCase(**arguments)  # type: ignore[arg-type]


def test_expand_cases_crosses_required_dimensions() -> None:
    cases = expand_cases(
        rows=[10_000, 100_000],
        widths=["narrow", "wide"],
        dataset_types=["mixed"],
        configs=["default", "conservative"],
        report_modes=[False, True],
        backends=["pandas"],
        output_formats=["pandas"],
        seed=42,
        warmups=1,
        repetitions=5,
    )
    assert len(cases) == 16
    assert len({case.case_id for case in cases}) == 16


def test_cli_small_matrix_writes_valid_result(tmp_path) -> None:
    exit_code = main(
        [
            "run",
            "--rows",
            "250",
            "--widths",
            "narrow",
            "--configs",
            "default",
            "--report-modes",
            "false",
            "--repetitions",
            "1",
            "--warmups",
            "0",
            "--timeout",
            "60",
            "--output",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    payloads = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(payloads) == 1
    assert payloads[0]["status"] == "completed"


def test_timeout_writes_a_schema_valid_result_with_exact_command(tmp_path) -> None:
    result = run_case_subprocess(_case(), tmp_path, timeout_seconds=0)

    payload = json.loads((tmp_path / f"{result.case.case_id}.json").read_text())
    validate_result(payload)
    assert result.status == "timeout"
    assert result.error_type == "TimeoutExpired"
    assert result.command == payload["command"]
    assert result.command.startswith(str(Path(sys.executable)))
    assert "worker_main" in result.command


@pytest.mark.parametrize(
    "stderr,expected_status,expected_error_type",
    [
        ("worker failed", "failed", "ChildProcessError"),
        ("Traceback: MemoryError", "oom", "MemoryError"),
    ],
)
def test_nonzero_worker_exit_writes_failure_record(
    tmp_path, monkeypatch, stderr: str, expected_status: str, expected_error_type: str
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_case_subprocess(_case(), tmp_path, timeout_seconds=60)

    payload = json.loads((tmp_path / f"{result.case.case_id}.json").read_text())
    validate_result(payload)
    assert result.status == expected_status
    assert result.error_type == expected_error_type
    assert result.error_message == stderr


def test_missing_worker_result_writes_failure_record(tmp_path, monkeypatch) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_case_subprocess(_case(), tmp_path, timeout_seconds=60)

    payload = json.loads((tmp_path / f"{result.case.case_id}.json").read_text())
    validate_result(payload)
    assert result.status == "failed"
    assert result.error_type == "FileNotFoundError"


def test_result_text_containing_memory_error_is_recorded_as_oom(tmp_path, monkeypatch) -> None:
    real_run = subprocess.run

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] != sys.executable:
            return real_run(command, **kwargs)  # type: ignore[call-overload]
        Path(command[4]).write_text("MemoryError: allocation failed", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_case_subprocess(_case(), tmp_path, timeout_seconds=60)

    assert result.status == "oom"
    assert result.error_type == "MemoryError"
    validate_result(json.loads((tmp_path / f"{result.case.case_id}.json").read_text()))


def test_matrix_continues_after_failure(tmp_path) -> None:
    cases = [_case(backend="unsupported"), _case(rows=101)]

    results = run_matrix(cases, tmp_path, timeout_seconds=60)

    assert [result.status for result in results] == ["failed", "completed"]
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_matrix_continues_after_schema_invalid_worker_result(tmp_path, monkeypatch) -> None:
    real_run = subprocess.run
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            Path(command[4]).write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return real_run(command, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = run_matrix([_case(), _case(rows=101)], tmp_path, timeout_seconds=60)

    assert [result.status for result in results] == ["failed", "completed"]
    assert results[0].error_type == "ValidationError"
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_matrix_rejects_duplicate_case_ids(tmp_path) -> None:
    case = _case()

    with pytest.raises(ValueError, match="duplicate case IDs"):
        run_matrix([case, case], tmp_path, timeout_seconds=60)


def test_existing_semantically_different_result_is_not_overwritten(tmp_path, monkeypatch) -> None:
    case = _case()
    output_path = tmp_path / f"{case.case_id}.json"
    original = json.dumps({"case": asdict(_case(rows=999))})
    output_path.write_text(original, encoding="utf-8")

    def unexpected_run(command: list[str], **kwargs: object) -> None:
        raise AssertionError("worker must not start for a case ID collision")

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    with pytest.raises(RuntimeError, match="case ID collision"):
        run_case_subprocess(case, tmp_path, timeout_seconds=60)

    assert output_path.read_text(encoding="utf-8") == original


def test_cli_returns_failure_only_after_all_cases_finish(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--rows",
            "100,101",
            "--widths",
            "narrow",
            "--configs",
            "default",
            "--report-modes",
            "false",
            "--backends",
            "unsupported",
            "--repetitions",
            "1",
            "--warmups",
            "0",
            "--timeout",
            "60",
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert capsys.readouterr().out.count(" failed\n") == 2
