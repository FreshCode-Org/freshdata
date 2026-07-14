from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace

import pytest
from benchmarks.performance.models import BenchmarkCase, BenchmarkResult
from benchmarks.performance.schema import validate_result
from benchmarks.performance.worker import execute_case, worker_main


def _reject_json_constant(value: str) -> None:
    raise ValueError(value)


def test_worker_records_warm_samples_memory_and_embedded_report() -> None:
    case = BenchmarkCase(
        rows=500,
        width="narrow",
        config_name="default",
        options={"verbose": False},
        return_report=False,
        warmups=1,
        repetitions=2,
    )

    result = execute_case(case, command="unit-test")

    assert result.status == "completed"
    assert len(result.samples_seconds) == 2
    assert result.peak_rss_bytes is not None and result.peak_rss_bytes >= 0
    assert result.peak_python_bytes is not None and result.peak_python_bytes > 0
    assert result.input_bytes is not None and result.input_bytes > 0


def test_worker_report_flag_does_not_change_cleaned_values() -> None:
    base = BenchmarkCase(
        rows=500,
        width="narrow",
        config_name="default",
        options={"verbose": False},
        warmups=0,
        repetitions=1,
    )

    without_report = execute_case(base, command="unit-test")
    with_report = execute_case(replace(base, return_report=True), command="unit-test")

    assert without_report.status == with_report.status == "completed"
    assert without_report.output_fingerprint == with_report.output_fingerprint
    assert without_report.report_fingerprint == with_report.report_fingerprint
    assert without_report.result_type == "CleanResult"
    assert with_report.result_type == "tuple[CleanResult,CleanReport]"


def test_worker_main_writes_a_strict_schema_valid_result(tmp_path) -> None:
    case = BenchmarkCase(
        rows=100,
        width="narrow",
        config_name="default",
        options={"verbose": False},
        warmups=0,
        repetitions=1,
    )
    case_path = tmp_path / "case.json"
    result_path = tmp_path / "result.json"
    case_path.write_text(json.dumps(asdict(case), allow_nan=False), encoding="utf-8")

    worker_main(str(case_path), str(result_path), "unit-test worker")

    raw_payload = result_path.read_text(encoding="utf-8")
    payload = json.loads(raw_payload, parse_constant=_reject_json_constant)
    validate_result(payload)
    result = BenchmarkResult.from_dict(payload)
    assert result.status == "completed"
    assert result.command == "unit-test worker"


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_worker_main_rejects_non_standard_case_json_constants(tmp_path, constant: str) -> None:
    case = BenchmarkCase(
        rows=100,
        width="narrow",
        config_name="invalid-json",
        options={"bad": "__NON_FINITE__"},
        warmups=0,
        repetitions=1,
    )
    case_path = tmp_path / "case.json"
    result_path = tmp_path / "result.json"
    case_json = json.dumps(asdict(case)).replace('"__NON_FINITE__"', constant)
    case_path.write_text(case_json, encoding="utf-8")

    with pytest.raises(ValueError, match="case JSON contains non-standard constant"):
        worker_main(str(case_path), str(result_path), "unit-test worker")

    assert not result_path.exists()


def test_worker_subprocesses_preserve_report_mode_fingerprints(tmp_path) -> None:
    base = BenchmarkCase(
        rows=100,
        width="narrow",
        config_name="default",
        options={"verbose": False},
        warmups=0,
        repetitions=1,
    )
    payloads = []
    worker_script = (
        "from benchmarks.performance.worker import worker_main; "
        "worker_main(__import__('sys').argv[1], __import__('sys').argv[2], "
        "__import__('sys').argv[3])"
    )

    for index, case in enumerate((base, replace(base, return_report=True))):
        case_path = tmp_path / f"case-{index}.json"
        result_path = tmp_path / f"result-{index}.json"
        case_path.write_text(
            json.dumps(asdict(case), allow_nan=False),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                worker_script,
                str(case_path),
                str(result_path),
                "unit-test subprocess",
            ],
            timeout=60,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(
            result_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
        validate_result(payload)
        payloads.append(payload)

    without_report, with_report = payloads
    assert without_report["status"] == with_report["status"] == "completed"
    assert without_report["output_fingerprint"] == with_report["output_fingerprint"]
    assert without_report["report_fingerprint"] == with_report["report_fingerprint"]
    assert without_report["result_type"] == "CleanResult"
    assert with_report["result_type"] == "tuple[CleanResult,CleanReport]"
