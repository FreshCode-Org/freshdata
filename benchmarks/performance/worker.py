from __future__ import annotations

import gc
import hashlib
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd

import freshdata as fd

from .datasets import DatasetSpec, make_mixed_frame
from .environment import capture_environment
from .memory import PeakRSS
from .models import BenchmarkCase, BenchmarkResult
from .schema import validate_result


def _run_clean(frame: pd.DataFrame, case: BenchmarkCase) -> Any:
    return fd.clean(
        frame,
        config=case.options,
        return_report=case.return_report,
        engine=case.backend,
        output_format=case.output_format,
    )


def _fingerprints(result: Any, return_report: bool) -> tuple[str, str, str]:
    if return_report:
        cleaned, report = result
        result_type = "tuple[CleanResult,CleanReport]"
    else:
        cleaned, report = result, result.report()
        result_type = "CleanResult"

    frame_hash = hashlib.sha256()
    frame_hash.update(pd.util.hash_pandas_object(cleaned, index=True).values.tobytes())
    frame_hash.update(repr(list(cleaned.columns)).encode())
    frame_hash.update(repr([str(dtype) for dtype in cleaned.dtypes]).encode())

    report_payload = report.to_dict()
    report_payload.pop("duration_seconds", None)
    report_payload.pop("stage_timings", None)
    report_hash = hashlib.sha256(
        json.dumps(report_payload, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()
    return frame_hash.hexdigest(), report_hash, result_type


def execute_case(case: BenchmarkCase, *, command: str) -> BenchmarkResult:
    frame = make_mixed_frame(
        DatasetSpec(
            rows=case.rows,
            width=case.width,
            seed=case.seed,
            dataset_type=case.dataset_type,
        )
    )
    input_bytes = int(frame.memory_usage(index=True, deep=True).sum())

    for _ in range(case.warmups):
        _run_clean(frame, case)

    samples: list[float] = []
    for _ in range(case.repetitions):
        gc.collect()
        started = time.perf_counter()
        _run_clean(frame, case)
        samples.append(time.perf_counter() - started)

    gc.collect()
    tracemalloc.start()
    try:
        with PeakRSS() as rss:
            measured_result = _run_clean(frame, case)
        _current, python_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    output_fingerprint, report_fingerprint, result_type = _fingerprints(
        measured_result, case.return_report
    )
    return BenchmarkResult.completed(
        case=case,
        environment=capture_environment(),
        samples_seconds=samples,
        peak_rss_bytes=rss.increase_bytes,
        peak_python_bytes=python_peak,
        input_bytes=input_bytes,
        command=command,
        output_fingerprint=output_fingerprint,
        report_fingerprint=report_fingerprint,
        result_type=result_type,
    )


def worker_main(case_path: str, result_path: str, command: str) -> None:
    case_payload = json.loads(Path(case_path).read_text(encoding="utf-8"))
    case = BenchmarkCase(**case_payload)
    result = execute_case(case, command=command)
    payload = result.to_dict()
    validate_result(payload)
    Path(result_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
