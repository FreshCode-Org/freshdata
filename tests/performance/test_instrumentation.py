from __future__ import annotations

import json
import tracemalloc
from dataclasses import asdict
from types import SimpleNamespace

import jsonschema
import pandas as pd
import pytest
from benchmarks.performance import instrumentation
from benchmarks.performance.cli import main
from benchmarks.performance.environment import capture_environment
from benchmarks.performance.instrumentation import (
    STAGE_RULES,
    OperationCounter,
    _function_records,
    profile_case,
)
from benchmarks.performance.models import BenchmarkCase, BenchmarkResult
from benchmarks.performance.schema import validate_result


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        rows=500,
        width="narrow",
        config_name="default",
        options={"verbose": False},
        warmups=0,
        repetitions=1,
    )


def test_operation_counter_observes_copy_scan_and_conversion_calls() -> None:
    frame = pd.DataFrame({"a": [1, None, 3], "b": ["1", "2", "3"]})
    with OperationCounter() as counter:
        frame.copy(deep=False)
        frame["a"].isna()
        frame["a"].nunique(dropna=True)
        frame["b"].astype("string")
    assert counter.counts["dataframe.copy"] == 1
    assert counter.counts["series.isna"] >= 1
    assert counter.counts["series.nunique"] == 1
    assert counter.counts["series.astype"] == 1


def test_operation_counter_restores_every_method_after_exception() -> None:
    originals = {
        key: getattr(owner, method) for key, (owner, method) in OperationCounter.METHODS.items()
    }

    with pytest.raises(RuntimeError, match="stop"):
        with OperationCounter():
            raise RuntimeError("stop")

    assert {
        key: getattr(owner, method) for key, (owner, method) in OperationCounter.METHODS.items()
    } == originals


def test_operation_counter_wrappers_call_saved_originals_without_recursing() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3]})

    with OperationCounter() as counter:
        copied = frame.copy()

    assert copied.equals(frame)
    assert counter.counts["dataframe.copy"] == 1


def test_operation_counter_retains_zero_observations_for_every_method() -> None:
    with OperationCounter() as counter:
        pass

    assert counter.counts == dict.fromkeys(OperationCounter.METHODS, 0)


def test_profile_case_reports_exact_hot_functions_and_allocations() -> None:
    result = profile_case(_case())
    assert result.functions
    assert all(
        {"file", "line", "function", "self_seconds", "cumulative_seconds", "calls"} <= set(item)
        for item in result.functions
    )
    assert result.allocations
    assert all({"file", "line", "bytes", "count"} <= set(item) for item in result.allocations)
    assert result.stages["total"] > 0
    assert result.operations["dataframe.copy"] >= 1
    assert asdict(result)["operations"] == result.operations


def _synthetic_stats(
    *functions: tuple[str, str, float],
) -> SimpleNamespace:
    return SimpleNamespace(
        stats={
            (file, line, function): (1, 1, self_seconds, self_seconds, {})
            for line, (file, function, self_seconds) in enumerate(functions, start=1)
        }
    )


def test_specific_context_functions_take_precedence_over_context_path() -> None:
    stats = _synthetic_stats(
        ("/project/freshdata/engine/context.py", "numeric_corr_matrix", 1.0),
        ("/project/freshdata/engine/context.py", "infer_role", 2.0),
        ("/project/freshdata/engine/context.py", "build_context", 3.0),
        ("/project/freshdata/engine/context.py", "collect_metadata", 4.0),
    )

    _functions, stages = _function_records(stats)  # type: ignore[arg-type]

    assert stages["correlation"] == 1.0
    assert stages["role_inference"] == 5.0
    assert stages["context"] == 4.0
    assert sum(value for key, value in stages.items() if key != "total") == stages["total"]


def test_correlation_rule_does_not_match_partial_function_names() -> None:
    stats = _synthetic_stats(
        ("/project/freshdata/engine/context.py", "decorrelate_labels", 1.0),
    )

    _functions, stages = _function_records(stats)  # type: ignore[arg-type]

    assert stages["correlation"] == 0.0
    assert stages["context"] == 1.0


def test_profile_case_excludes_allocations_retained_before_baseline() -> None:
    tracemalloc.start()
    namespace: dict[str, object] = {}
    try:
        exec(
            compile("retained = bytearray(1000000)", "/tmp/freshdata/preexisting.py", "exec"),
            namespace,
        )

        result = profile_case(_case())

        assert tracemalloc.is_tracing()
        assert all(item["file"] != "/tmp/freshdata/preexisting.py" for item in result.allocations)
    finally:
        tracemalloc.stop()


def test_profile_case_stops_locally_started_tracer() -> None:
    if tracemalloc.is_tracing():
        tracemalloc.stop()

    profile_case(_case())

    assert not tracemalloc.is_tracing()


def test_profile_case_restores_methods_and_tracer_when_clean_raises(monkeypatch) -> None:
    originals = {
        key: getattr(owner, method) for key, (owner, method) in OperationCounter.METHODS.items()
    }
    if tracemalloc.is_tracing():
        tracemalloc.stop()

    def fail_clean(*args: object, **kwargs: object) -> None:
        raise RuntimeError("profile failure")

    monkeypatch.setattr(instrumentation.fd, "clean", fail_clean)

    with pytest.raises(RuntimeError, match="profile failure"):
        profile_case(_case())

    assert not tracemalloc.is_tracing()
    assert {
        key: getattr(owner, method) for key, (owner, method) in OperationCounter.METHODS.items()
    } == originals


def _profile_payload() -> dict[str, object]:
    result = BenchmarkResult.completed(
        case=_case(),
        environment=capture_environment(),
        samples_seconds=[1.0],
        peak_rss_bytes=1000,
        peak_python_bytes=500,
        input_bytes=250,
        command="profile-test",
    )
    payload = result.to_dict()
    payload["profile"] = {
        "functions": [
            {
                "file": "/project/freshdata/cleaner.py",
                "line": 12,
                "function": "clean",
                "self_seconds": 0.1,
                "cumulative_seconds": 0.2,
                "calls": 1,
            }
        ],
        "allocations": [
            {
                "file": "/project/freshdata/cleaner.py",
                "line": 13,
                "bytes": 1024,
                "count": 2,
            }
        ],
        "stages": {**dict.fromkeys(STAGE_RULES, 0.0), "total": 0.1},
        "operations": {**dict.fromkeys(OperationCounter.METHODS, 0), "dataframe.copy": 3},
    }
    return payload


def test_schema_accepts_strict_profile_records() -> None:
    validate_result(_profile_payload())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("functions", 0, "line"), "12"),
        (("allocations", 0, "bytes"), -1),
        (("stages", "total"), -0.1),
        (("operations", "dataframe.copy"), -1),
    ],
)
def test_schema_rejects_invalid_profile_records(path: tuple[object, ...], value: object) -> None:
    payload = _profile_payload()
    target: object = payload["profile"]
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(jsonschema.ValidationError):
        validate_result(payload)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        *(("stages", key) for key in (*STAGE_RULES, "total")),
        *(("operations", key) for key in OperationCounter.METHODS),
    ],
)
def test_schema_requires_every_profile_metric_key(section: str, key: str) -> None:
    payload = _profile_payload()
    del payload["profile"][section][key]  # type: ignore[index]

    with pytest.raises(jsonschema.ValidationError):
        validate_result(payload)


@pytest.mark.parametrize("section", ["stages", "operations"])
def test_schema_rejects_unknown_profile_metric_key(section: str) -> None:
    payload = _profile_payload()
    payload["profile"][section]["unknown"] = 1  # type: ignore[index]

    with pytest.raises(jsonschema.ValidationError):
        validate_result(payload)


@pytest.mark.parametrize("section", ["functions", "allocations"])
def test_schema_rejects_more_than_100_profile_records(section: str) -> None:
    payload = _profile_payload()
    records = payload["profile"][section]  # type: ignore[index]
    payload["profile"][section] = records * 101  # type: ignore[index,operator]

    with pytest.raises(jsonschema.ValidationError):
        validate_result(payload)


def test_cli_profile_writes_one_strict_schema_valid_case(tmp_path) -> None:
    exit_code = main(
        [
            "profile",
            "--rows",
            "100",
            "--widths",
            "narrow",
            "--configs",
            "default",
            "--report-modes",
            "false",
            "--backends",
            "pandas",
            "--output-formats",
            "pandas",
            "--warmups",
            "0",
            "--repetitions",
            "1",
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    paths = list(tmp_path.glob("*.profile.json"))
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text())
    validate_result(payload)
    serialized_case = BenchmarkCase(**payload["case"])
    assert paths[0].name == f"{serialized_case.case_id}.profile.json"
    assert payload["profile"]["functions"]


@pytest.mark.parametrize(
    ("selector", "values"),
    [
        ("--rows", "100,101"),
        ("--widths", "narrow,wide"),
        ("--configs", "default,conservative"),
        ("--report-modes", "false,true"),
        ("--backends", "pandas,unsupported"),
        ("--output-formats", "pandas,polars"),
    ],
)
def test_cli_profile_rejects_multi_case_selectors(tmp_path, selector: str, values: str) -> None:
    arguments = [
        "profile",
        "--rows",
        "100",
        "--widths",
        "narrow",
        "--configs",
        "default",
        "--report-modes",
        "false",
        "--backends",
        "pandas",
        "--output-formats",
        "pandas",
        "--warmups",
        "0",
        "--repetitions",
        "1",
        "--output",
        str(tmp_path),
    ]
    index = arguments.index(selector)
    arguments[index + 1] = values

    with pytest.raises(SystemExit):
        main(arguments)

    assert not list(tmp_path.glob("*.profile.json"))
