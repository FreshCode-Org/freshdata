from __future__ import annotations

from dataclasses import replace

import pytest
from benchmarks.performance.environment import capture_environment
from benchmarks.performance.models import BenchmarkCase, BenchmarkResult
from benchmarks.performance.schema import validate_result


def test_case_id_is_stable_and_configuration_sensitive() -> None:
    case = BenchmarkCase(rows=10_000, width="narrow", config_name="default", options={})
    assert (
        case.case_id
        == BenchmarkCase(rows=10_000, width="narrow", config_name="default", options={}).case_id
    )
    assert case.case_id != replace(case, return_report=True).case_id


def test_environment_contains_required_reproduction_fields() -> None:
    env = capture_environment()
    assert env.python_version
    assert env.pandas_version
    assert env.numpy_version
    assert env.git_commit
    assert env.platform
    assert env.cpu_count_logical >= 1


def test_result_round_trip_validates() -> None:
    case = BenchmarkCase(rows=10_000, width="narrow", config_name="default", options={})
    result = BenchmarkResult.completed(
        case=case,
        environment=capture_environment(),
        samples_seconds=[1.0, 1.1, 0.9],
        peak_rss_bytes=1_000_000,
        peak_python_bytes=500_000,
        input_bytes=250_000,
        command="python -m benchmarks.performance worker",
    )
    payload = result.to_dict()
    validate_result(payload)
    assert BenchmarkResult.from_dict(payload).case == case


def test_completed_result_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="samples_seconds"):
        BenchmarkResult.completed(
            case=BenchmarkCase(rows=10_000, width="narrow", config_name="default", options={}),
            environment=capture_environment(),
            samples_seconds=[],
            peak_rss_bytes=0,
            peak_python_bytes=0,
            input_bytes=1,
            command="x",
        )
