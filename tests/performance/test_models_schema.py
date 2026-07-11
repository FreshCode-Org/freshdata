from __future__ import annotations

import json
from dataclasses import replace
from math import inf, nan

import jsonschema
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


def test_case_options_support_nested_json_values() -> None:
    options = {
        "enabled": True,
        "thresholds": [1, 2.5, None],
        "nested": {"label": "strict", "values": ["a", "b"]},
    }
    case = BenchmarkCase(rows=10_000, width="narrow", config_name="nested", options=options)
    result = BenchmarkResult.completed(
        case=case,
        environment=capture_environment(),
        samples_seconds=[1.0],
        peak_rss_bytes=1_000_000,
        peak_python_bytes=500_000,
        input_bytes=250_000,
        command="x",
    )

    payload = result.to_dict()
    json.dumps(payload, allow_nan=False)
    assert BenchmarkResult.from_dict(payload).case == case


@pytest.mark.parametrize(
    "options",
    [
        {"bad": {1, 2}},
        {"bad": (1, 2)},
        {"bad": object()},
        {1: "non-string-key"},
    ],
)
def test_case_options_reject_non_json_values(options: object) -> None:
    with pytest.raises(TypeError, match="options"):
        BenchmarkCase(
            rows=10_000,
            width="narrow",
            config_name="invalid",
            options=options,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_case_options_reject_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BenchmarkCase(
            rows=10_000,
            width="narrow",
            config_name="invalid",
            options={"bad": [value]},
        )


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


def _completed_payload() -> dict[str, object]:
    return BenchmarkResult.completed(
        case=BenchmarkCase(rows=10_000, width="narrow", config_name="default", options={}),
        environment=capture_environment(),
        samples_seconds=[1.0, 1.1, 0.9],
        peak_rss_bytes=1_000_000,
        peak_python_bytes=500_000,
        input_bytes=250_000,
        command="x",
    ).to_dict()


def test_schema_rejects_wrong_case_field_type() -> None:
    payload = _completed_payload()
    payload["case"]["rows"] = "10000"  # type: ignore[index]

    with pytest.raises(jsonschema.ValidationError):
        validate_result(payload)


def test_schema_rejects_wrong_metric_type() -> None:
    payload = _completed_payload()
    payload["median_seconds"] = "fast"

    with pytest.raises(jsonschema.ValidationError):
        validate_result(payload)


def test_schema_requires_every_environment_field() -> None:
    payload = _completed_payload()
    payload["environment"].pop("git_dirty")  # type: ignore[union-attr]

    with pytest.raises(jsonschema.ValidationError):
        validate_result(payload)


@pytest.mark.parametrize("location", ["result", "case", "environment"])
def test_schema_rejects_unknown_fields(location: str) -> None:
    payload = _completed_payload()
    if location == "result":
        payload["unknown"] = True
    else:
        payload[location]["unknown"] = True  # type: ignore[index]

    with pytest.raises(jsonschema.ValidationError):
        validate_result(payload)


@pytest.mark.parametrize("sample", [nan, inf, -inf])
def test_completed_result_rejects_non_finite_samples(sample: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BenchmarkResult.completed(
            case=BenchmarkCase(rows=10_000, width="narrow", config_name="default", options={}),
            environment=capture_environment(),
            samples_seconds=[sample],
            peak_rss_bytes=1_000_000,
            peak_python_bytes=500_000,
            input_bytes=250_000,
            command="x",
        )


def test_completed_result_rejects_non_finite_derived_metric() -> None:
    with pytest.raises(ValueError, match="finite"):
        BenchmarkResult.completed(
            case=BenchmarkCase(rows=10_000, width="narrow", config_name="default", options={}),
            environment=capture_environment(),
            samples_seconds=[5e-324],
            peak_rss_bytes=1_000_000,
            peak_python_bytes=500_000,
            input_bytes=250_000,
            command="x",
        )


def test_validation_rejects_non_strict_json_numbers() -> None:
    payload = _completed_payload()
    payload["median_seconds"] = nan

    with pytest.raises(ValueError, match="finite JSON"):
        validate_result(payload)
