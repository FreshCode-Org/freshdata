from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

from jsonschema.exceptions import ValidationError

from .environment import capture_environment
from .models import BenchmarkCase, BenchmarkResult
from .schema import validate_result

CONFIGS = {
    "default": {"verbose": False},
    "conservative": {"strategy": "conservative", "verbose": False},
    "representation_off": {
        "strategy": "conservative",
        "column_names": False,
        "strip_whitespace": False,
        "normalize_sentinels": False,
        "fix_dtypes": False,
        "drop_duplicates": False,
        "verbose": False,
    },
    "statistical_off": {
        "strategy": "conservative",
        "impute": None,
        "outliers": None,
        "verbose": False,
    },
    "explicit": {
        "strategy": "conservative",
        "impute": "median",
        "outliers": "flag",
        "verbose": False,
    },
    "aggressive": {"strategy": "aggressive", "verbose": False},
    "semantic": {"semantic_mode": "assist", "verbose": False},
    "missforest": {
        "strategy": "conservative",
        "impute": "missforest",
        "verbose": False,
    },
}

_WORKER_SCRIPT = (
    "from benchmarks.performance.worker import worker_main; "
    "worker_main(__import__('sys').argv[1], __import__('sys').argv[2], "
    "__import__('sys').argv[3])"
)


def expand_cases(
    *,
    rows: list[int],
    widths: list[str],
    dataset_types: list[str],
    configs: list[str],
    report_modes: list[bool],
    backends: list[str],
    output_formats: list[str],
    seed: int,
    warmups: int,
    repetitions: int,
) -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            rows=row_count,
            width=width,
            config_name=config_name,
            options=dict(CONFIGS[config_name]),
            dataset_type=dataset_type,
            return_report=return_report,
            backend=backend,
            output_format=output_format,
            seed=seed,
            warmups=warmups,
            repetitions=repetitions,
        )
        for (
            row_count,
            width,
            dataset_type,
            config_name,
            return_report,
            backend,
            output_format,
        ) in product(
            rows,
            widths,
            dataset_types,
            configs,
            report_modes,
            backends,
            output_formats,
        )
    ]


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(f"result JSON contains non-standard constant: {value}")


def _write_result(output_path: Path, result: BenchmarkResult) -> None:
    payload = result.to_dict()
    validate_result(payload)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _guard_existing_result(output_path: Path, case: BenchmarkCase) -> None:
    if not output_path.exists():
        return
    try:
        payload = json.loads(
            output_path.read_text(encoding="utf-8"),
            parse_constant=_reject_non_standard_json_constant,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"existing result cannot be verified: {output_path}") from exc
    if not isinstance(payload, dict) or payload.get("case") != asdict(case):
        raise RuntimeError(f"case ID collision for existing result: {output_path}")


def _failure_result(
    *,
    case: BenchmarkCase,
    status: str,
    command: str,
    error_type: str,
    error_message: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        schema_version=1,
        status=status,
        case=case,
        environment=capture_environment(),
        command=command,
        error_type=error_type,
        error_message=error_message,
    )


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _load_result(result_text: str) -> BenchmarkResult:
    payload: Any = json.loads(
        result_text,
        parse_constant=_reject_non_standard_json_constant,
    )
    if not isinstance(payload, dict):
        raise TypeError("result JSON must be an object")
    validate_result(payload)
    return BenchmarkResult.from_dict(payload)


def run_case_subprocess(
    case: BenchmarkCase, output_dir: Path, timeout_seconds: int
) -> BenchmarkResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case.case_id}.json"
    _guard_existing_result(output_path, case)
    with tempfile.TemporaryDirectory(prefix="freshdata-benchmark-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        case_path = temporary_path / "case.json"
        result_path = temporary_path / "result.json"
        worker_command = [
            sys.executable,
            "-c",
            _WORKER_SCRIPT,
            str(case_path),
            str(result_path),
        ]
        command = shlex.join(worker_command)
        worker_command.append(command)
        case_path.write_text(
            json.dumps(asdict(case), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                worker_command,
                timeout=timeout_seconds,
                capture_output=True,
                text=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            result = _failure_result(
                case=case,
                status="timeout",
                command=command,
                error_type="TimeoutExpired",
                error_message=str(exc),
            )
            _write_result(output_path, result)
            return result

        result_text = ""
        loaded_result: BenchmarkResult | None = None
        result_error: Exception | None = None
        try:
            if not result_path.exists():
                raise FileNotFoundError("worker did not write a result file")
            result_text = result_path.read_text(encoding="utf-8")
            loaded_result = _load_result(result_text)
        except (OSError, TypeError, ValueError, ValidationError) as exc:
            result_error = exc

        process_output = "\n".join(
            part
            for part in (_text(completed.stdout), _text(completed.stderr), result_text)
            if part
        )
        if "MemoryError" in process_output:
            result = _failure_result(
                case=case,
                status="oom",
                command=command,
                error_type="MemoryError",
                error_message=process_output,
            )
        elif completed.returncode != 0:
            message = _text(completed.stderr) or _text(completed.stdout)
            result = _failure_result(
                case=case,
                status="failed",
                command=command,
                error_type="ChildProcessError",
                error_message=message or f"worker exited with code {completed.returncode}",
            )
        elif result_error is not None:
            result = _failure_result(
                case=case,
                status="failed",
                command=command,
                error_type=type(result_error).__name__,
                error_message=str(result_error),
            )
        else:
            assert loaded_result is not None
            result = loaded_result
        _write_result(output_path, result)
        return result


def run_matrix(
    cases: list[BenchmarkCase], output_dir: Path, timeout_seconds: int
) -> list[BenchmarkResult]:
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("input matrix contains duplicate case IDs")
    return [run_case_subprocess(case, output_dir, timeout_seconds) for case in cases]
