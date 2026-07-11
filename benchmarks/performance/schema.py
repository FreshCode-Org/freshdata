from __future__ import annotations

import json
from typing import Any

RESULT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$defs": {
        "json_value": {
            "anyOf": [
                {"type": ["null", "boolean", "number", "string"]},
                {"type": "array", "items": {"$ref": "#/$defs/json_value"}},
                {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/$defs/json_value"},
                },
            ]
        }
    },
    "type": "object",
    "required": [
        "schema_version",
        "status",
        "case",
        "environment",
        "samples_seconds",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "stdev_seconds",
        "coefficient_of_variation",
        "throughput_rows_per_second",
        "peak_rss_bytes",
        "peak_python_bytes",
        "input_bytes",
        "input_to_peak_ratio",
        "command",
        "error_type",
        "error_message",
        "output_fingerprint",
        "report_fingerprint",
        "result_type",
    ],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "status": {
            "type": "string",
            "enum": ["completed", "failed", "timeout", "oom", "skipped"],
        },
        "case": {
            "type": "object",
            "required": [
                "rows",
                "width",
                "config_name",
                "options",
                "dataset_type",
                "return_report",
                "backend",
                "output_format",
                "seed",
                "warmups",
                "repetitions",
            ],
            "additionalProperties": False,
            "properties": {
                "rows": {"type": "integer", "minimum": 1},
                "width": {"type": "string"},
                "config_name": {"type": "string"},
                "options": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/$defs/json_value"},
                },
                "dataset_type": {"type": "string"},
                "return_report": {"type": "boolean"},
                "backend": {"type": "string"},
                "output_format": {"type": "string"},
                "seed": {"type": "integer"},
                "warmups": {"type": "integer", "minimum": 0},
                "repetitions": {"type": "integer", "minimum": 1},
            },
        },
        "environment": {
            "type": "object",
            "required": [
                "git_commit",
                "git_dirty",
                "python_version",
                "pandas_version",
                "numpy_version",
                "freshdata_version",
                "optional_versions",
                "platform",
                "processor",
                "cpu_count_logical",
                "cpu_count_physical",
                "total_ram_bytes",
            ],
            "additionalProperties": False,
            "properties": {
                "git_commit": {"type": "string"},
                "git_dirty": {"type": "boolean"},
                "python_version": {"type": "string"},
                "pandas_version": {"type": "string"},
                "numpy_version": {"type": "string"},
                "freshdata_version": {"type": "string"},
                "optional_versions": {
                    "type": "object",
                    "additionalProperties": {"type": ["string", "null"]},
                },
                "platform": {"type": "string"},
                "processor": {"type": "string"},
                "cpu_count_logical": {"type": "integer", "minimum": 1},
                "cpu_count_physical": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                },
                "total_ram_bytes": {"type": ["integer", "null"], "minimum": 0},
            },
        },
        "samples_seconds": {
            "type": "array",
            "items": {"type": "number", "minimum": 0},
        },
        "median_seconds": {"type": ["number", "null"], "minimum": 0},
        "min_seconds": {"type": ["number", "null"], "minimum": 0},
        "max_seconds": {"type": ["number", "null"], "minimum": 0},
        "stdev_seconds": {"type": ["number", "null"], "minimum": 0},
        "coefficient_of_variation": {
            "type": ["number", "null"],
            "minimum": 0,
        },
        "throughput_rows_per_second": {
            "type": ["number", "null"],
            "minimum": 0,
        },
        "peak_rss_bytes": {"type": ["integer", "null"], "minimum": 0},
        "peak_python_bytes": {"type": ["integer", "null"], "minimum": 0},
        "input_bytes": {"type": ["integer", "null"], "minimum": 0},
        "input_to_peak_ratio": {"type": ["number", "null"], "minimum": 0},
        "command": {"type": "string"},
        "error_type": {"type": ["string", "null"]},
        "error_message": {"type": ["string", "null"]},
        "output_fingerprint": {"type": ["string", "null"]},
        "report_fingerprint": {"type": ["string", "null"]},
        "result_type": {"type": ["string", "null"]},
    },
}


def validate_result(payload: dict[str, Any]) -> None:
    import jsonschema  # noqa: PLC0415

    try:
        json.dumps(payload, allow_nan=False)
    except ValueError as exc:
        raise ValueError("result payload must contain only finite JSON numbers") from exc
    jsonschema.validate(payload, RESULT_SCHEMA)
    if payload["status"] == "completed":
        if not payload["samples_seconds"]:
            raise ValueError("completed result requires samples_seconds")
        for field in (
            "median_seconds",
            "min_seconds",
            "max_seconds",
            "peak_rss_bytes",
            "peak_python_bytes",
            "input_bytes",
        ):
            if payload.get(field) is None:
                raise ValueError(f"completed result requires {field}")
