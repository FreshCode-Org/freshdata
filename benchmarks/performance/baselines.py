from __future__ import annotations

import gc
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Callable, NoReturn, Union

import pandas as pd

from .datasets import DatasetSpec, make_mixed_frame
from .environment import capture_environment
from .memory import PeakRSS
from .models import BenchmarkCase, BenchmarkResult
from .schema import validate_result

Baseline = Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]


def _numeric_median_fill(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        **{
            column: frame[column].fillna(frame[column].median())
            for column in frame.select_dtypes(include="number").columns
            if frame[column].notna().any()
        }
    )


BASELINES: dict[str, Baseline] = {
    "shallow_copy": lambda frame: frame.copy(deep=False),
    "numeric_median_fill": _numeric_median_fill,
    "duplicates": lambda frame: frame.drop_duplicates(),
    "null_counts": lambda frame: frame.isna().sum(),
}

_BASELINE_WORKER_SCRIPT = (
    "from benchmarks.performance.baselines import baseline_worker_main; "
    "baseline_worker_main(__import__('sys').argv[1], __import__('sys').argv[2], "
    "__import__('sys').argv[3], __import__('sys').argv[4])"
)


def _component_case(case: BenchmarkCase, baseline_name: str) -> BenchmarkCase:
    if baseline_name not in BASELINES:
        choices = ", ".join(BASELINES)
        raise ValueError(f"unknown pandas baseline {baseline_name!r}; choose from {choices}")
    return replace(
        case,
        config_name=f"pandas_{baseline_name}",
        options={},
        return_report=False,
        backend="pandas-component-baseline",
        output_format="pandas",
    )


def _fingerprint(result: pd.DataFrame | pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(result, index=True).values.tobytes())
    if isinstance(result, pd.DataFrame):
        digest.update(repr(list(result.columns)).encode())
        digest.update(repr([str(dtype) for dtype in result.dtypes]).encode())
    else:
        digest.update(repr(result.name).encode())
        digest.update(str(result.dtype).encode())
    return digest.hexdigest()


def measure_pandas_baseline(
    case: BenchmarkCase, baseline_name: str, *, command: str
) -> BenchmarkResult:
    measured_case = _component_case(case, baseline_name)
    operation = BASELINES[baseline_name]
    frame = make_mixed_frame(
        DatasetSpec(
            rows=measured_case.rows,
            width=measured_case.width,
            seed=measured_case.seed,
            dataset_type=measured_case.dataset_type,
        )
    )
    input_bytes = int(frame.memory_usage(index=True, deep=True).sum())

    for _ in range(measured_case.warmups):
        operation(frame)

    samples: list[float] = []
    for _ in range(measured_case.repetitions):
        gc.collect()
        started = time.perf_counter()
        operation(frame)
        samples.append(time.perf_counter() - started)

    gc.collect()
    tracemalloc.start()
    try:
        with PeakRSS() as rss:
            measured_result = operation(frame)
        _current, python_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return BenchmarkResult.completed(
        case=measured_case,
        environment=capture_environment(),
        samples_seconds=samples,
        peak_rss_bytes=rss.increase_bytes,
        peak_python_bytes=python_peak,
        input_bytes=input_bytes,
        command=command,
        output_fingerprint=_fingerprint(measured_result),
        result_type=type(measured_result).__name__,
        baseline_name=baseline_name,
    )


def _reject_non_standard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"case JSON contains non-standard constant: {value}")


def baseline_worker_main(
    case_path: str, result_path: str, baseline_name: str, command: str
) -> None:
    payload = json.loads(
        Path(case_path).read_text(encoding="utf-8"),
        parse_constant=_reject_non_standard_json_constant,
    )
    result = measure_pandas_baseline(BenchmarkCase(**payload), baseline_name, command=command)
    serialized = result.to_dict()
    validate_result(serialized)
    Path(result_path).write_text(
        json.dumps(serialized, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _failure_result(
    case: BenchmarkCase,
    baseline_name: str,
    status: str,
    command: str,
    error_type: str,
    error_message: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        schema_version=1,
        status=status,
        case=_component_case(case, baseline_name),
        environment=capture_environment(),
        command=command,
        error_type=error_type,
        error_message=error_message,
        baseline_name=baseline_name,
    )


def run_baseline_subprocess(
    case: BenchmarkCase,
    baseline_name: str,
    output_dir: Path,
    timeout_seconds: int,
) -> BenchmarkResult:
    measured_case = _component_case(case, baseline_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{measured_case.case_id}.json"
    with tempfile.TemporaryDirectory(prefix="freshdata-pandas-baseline-") as directory:
        case_path = Path(directory) / "case.json"
        result_path = Path(directory) / "result.json"
        worker_command = [
            sys.executable,
            "-c",
            _BASELINE_WORKER_SCRIPT,
            str(case_path),
            str(result_path),
            baseline_name,
        ]
        command = shlex.join(worker_command)
        worker_command.append(command)
        case_path.write_text(
            json.dumps(asdict(measured_case), indent=2, sort_keys=True, allow_nan=False),
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
                measured_case,
                baseline_name,
                "timeout",
                command,
                "TimeoutExpired",
                str(exc),
            )
        else:
            process_output = "\n".join((completed.stdout, completed.stderr))
            if "MemoryError" in process_output:
                result = _failure_result(
                    measured_case,
                    baseline_name,
                    "oom",
                    command,
                    "MemoryError",
                    process_output,
                )
            elif completed.returncode != 0:
                result = _failure_result(
                    measured_case,
                    baseline_name,
                    "failed",
                    command,
                    "ChildProcessError",
                    completed.stderr
                    or completed.stdout
                    or f"worker exited with code {completed.returncode}",
                )
            else:
                try:
                    payload = json.loads(
                        result_path.read_text(encoding="utf-8"),
                        parse_constant=_reject_non_standard_json_constant,
                    )
                    validate_result(payload)
                    result = BenchmarkResult.from_dict(payload)
                except Exception as exc:  # converted to a durable benchmark result
                    result = _failure_result(
                        measured_case,
                        baseline_name,
                        "failed",
                        command,
                        type(exc).__name__,
                        str(exc),
                    )

    serialized = result.to_dict()
    validate_result(serialized)
    output_path.write_text(
        json.dumps(serialized, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return result


def expand_baseline_cases(
    *,
    rows: list[int],
    widths: list[str],
    dataset_types: list[str],
    baseline_names: list[str],
    seed: int,
    warmups: int,
    repetitions: int,
) -> list[tuple[BenchmarkCase, str]]:
    unknown = sorted(set(baseline_names) - set(BASELINES))
    if unknown:
        raise ValueError(f"unknown pandas baseline(s): {', '.join(unknown)}")
    return [
        (
            BenchmarkCase(
                rows=row_count,
                width=width,
                config_name=f"pandas_{baseline_name}",
                options={},
                dataset_type=dataset_type,
                backend="pandas-component-baseline",
                seed=seed,
                warmups=warmups,
                repetitions=repetitions,
            ),
            baseline_name,
        )
        for row_count, width, dataset_type, baseline_name in product(
            rows, widths, dataset_types, baseline_names
        )
    ]


def run_baseline_matrix(
    cases: list[tuple[BenchmarkCase, str]], output_dir: Path, timeout_seconds: int
) -> list[BenchmarkResult]:
    return [
        run_baseline_subprocess(case, baseline_name, output_dir, timeout_seconds)
        for case, baseline_name in cases
    ]
