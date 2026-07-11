from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Union  # noqa: UP035

JsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    List["JsonValue"],  # noqa: UP006
    Dict[str, "JsonValue"],  # noqa: UP006
]

_SUPPORTED_WIDTHS = frozenset(("narrow", "medium", "wide"))
_SUPPORTED_DATASET_TYPES = frozenset(
    (
        "mixed",
        "numeric",
        "categorical",
        "string",
        "nullable",
        "datetime",
        "high_cardinality",
    )
)


def _validate_integer(name: str, value: object, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _validate_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")


def _validate_json_value(value: object, path: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


@dataclass(frozen=True)
class BenchmarkCase:
    rows: int
    width: str
    config_name: str
    options: dict[str, JsonValue]
    dataset_type: str = "mixed"
    return_report: bool = False
    backend: str = "pandas"
    output_format: str = "pandas"
    seed: int = 42
    warmups: int = 1
    repetitions: int = 5

    def __post_init__(self) -> None:
        _validate_integer("rows", self.rows, minimum=1)
        if self.width not in _SUPPORTED_WIDTHS:
            raise ValueError(f"width must be one of {sorted(_SUPPORTED_WIDTHS)}")
        _validate_nonempty_string("config_name", self.config_name)
        if not isinstance(self.options, dict):
            raise TypeError("options must be a JSON object")
        _validate_json_value(self.options, "options")
        if self.dataset_type not in _SUPPORTED_DATASET_TYPES:
            raise ValueError(f"dataset_type must be one of {sorted(_SUPPORTED_DATASET_TYPES)}")
        if not isinstance(self.return_report, bool):
            raise TypeError("return_report must be a boolean")
        _validate_nonempty_string("backend", self.backend)
        _validate_nonempty_string("output_format", self.output_format)
        _validate_integer("seed", self.seed)
        _validate_integer("warmups", self.warmups, minimum=0)
        _validate_integer("repetitions", self.repetitions, minimum=1)

    @property
    def case_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class EnvironmentInfo:
    git_commit: str
    git_dirty: bool
    python_version: str
    pandas_version: str
    numpy_version: str
    freshdata_version: str
    optional_versions: dict[str, str | None]
    platform: str
    processor: str
    cpu_count_logical: int
    cpu_count_physical: int | None
    total_ram_bytes: int | None


@dataclass(frozen=True)
class BenchmarkResult:
    schema_version: int
    status: str
    case: BenchmarkCase
    environment: EnvironmentInfo
    samples_seconds: list[float] = field(default_factory=list)
    median_seconds: float | None = None
    min_seconds: float | None = None
    max_seconds: float | None = None
    stdev_seconds: float | None = None
    coefficient_of_variation: float | None = None
    throughput_rows_per_second: float | None = None
    peak_rss_bytes: int | None = None
    peak_python_bytes: int | None = None
    input_bytes: int | None = None
    input_to_peak_ratio: float | None = None
    command: str = ""
    error_type: str | None = None
    error_message: str | None = None
    output_fingerprint: str | None = None
    report_fingerprint: str | None = None
    result_type: str | None = None

    @classmethod
    def completed(
        cls,
        *,
        case: BenchmarkCase,
        environment: EnvironmentInfo,
        samples_seconds: list[float],
        peak_rss_bytes: int,
        peak_python_bytes: int,
        input_bytes: int,
        command: str,
        output_fingerprint: str | None = None,
        report_fingerprint: str | None = None,
        result_type: str | None = None,
    ) -> BenchmarkResult:
        if not samples_seconds:
            raise ValueError("samples_seconds must not be empty")
        if any(
            isinstance(sample, bool) or not isinstance(sample, (int, float))
            for sample in samples_seconds
        ):
            raise TypeError("samples_seconds must contain only numbers")
        if not all(math.isfinite(sample) for sample in samples_seconds):
            raise ValueError("samples_seconds must contain only finite values")
        if any(sample < 0 for sample in samples_seconds):
            raise ValueError("samples_seconds must contain only non-negative values")
        for name, value in (
            ("peak_rss_bytes", peak_rss_bytes),
            ("peak_python_bytes", peak_python_bytes),
            ("input_bytes", input_bytes),
        ):
            _validate_integer(name, value, minimum=0)
        median = statistics.median(samples_seconds)
        stdev = statistics.stdev(samples_seconds) if len(samples_seconds) > 1 else 0.0
        coefficient_of_variation = (stdev / median) if median else 0.0
        throughput_rows_per_second = (case.rows / median) if median else None
        input_to_peak_ratio = (peak_rss_bytes / input_bytes) if input_bytes else None
        derived_metrics = (
            median,
            min(samples_seconds),
            max(samples_seconds),
            stdev,
            coefficient_of_variation,
            throughput_rows_per_second,
            input_to_peak_ratio,
        )
        if any(metric is not None and not math.isfinite(metric) for metric in derived_metrics):
            raise ValueError("completed result metrics must contain only finite values")
        return cls(
            schema_version=1,
            status="completed",
            case=case,
            environment=environment,
            samples_seconds=samples_seconds,
            median_seconds=median,
            min_seconds=min(samples_seconds),
            max_seconds=max(samples_seconds),
            stdev_seconds=stdev,
            coefficient_of_variation=coefficient_of_variation,
            throughput_rows_per_second=throughput_rows_per_second,
            peak_rss_bytes=peak_rss_bytes,
            peak_python_bytes=peak_python_bytes,
            input_bytes=input_bytes,
            input_to_peak_ratio=input_to_peak_ratio,
            command=command,
            output_fingerprint=output_fingerprint,
            report_fingerprint=report_fingerprint,
            result_type=result_type,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        json.dumps(payload, allow_nan=False)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkResult:
        data = dict(payload)
        data["case"] = BenchmarkCase(**data["case"])
        data["environment"] = EnvironmentInfo(**data["environment"])
        return cls(**data)
