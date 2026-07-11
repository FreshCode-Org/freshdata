from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    rows: int
    width: str
    config_name: str
    options: dict[str, Any]
    dataset_type: str = "mixed"
    return_report: bool = False
    backend: str = "pandas"
    output_format: str = "pandas"
    seed: int = 42
    warmups: int = 1
    repetitions: int = 5

    @property
    def case_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
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
        median = statistics.median(samples_seconds)
        stdev = statistics.stdev(samples_seconds) if len(samples_seconds) > 1 else 0.0
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
            coefficient_of_variation=(stdev / median) if median else 0.0,
            throughput_rows_per_second=(case.rows / median) if median else None,
            peak_rss_bytes=peak_rss_bytes,
            peak_python_bytes=peak_python_bytes,
            input_bytes=input_bytes,
            input_to_peak_ratio=(peak_rss_bytes / input_bytes) if input_bytes else None,
            command=command,
            output_fingerprint=output_fingerprint,
            report_fingerprint=report_fingerprint,
            result_type=result_type,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkResult:
        data = dict(payload)
        data["case"] = BenchmarkCase(**data["case"])
        data["environment"] = EnvironmentInfo(**data["environment"])
        return cls(**data)
