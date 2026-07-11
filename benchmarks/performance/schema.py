from __future__ import annotations

from typing import Any

RESULT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schema_version",
        "status",
        "case",
        "environment",
        "samples_seconds",
        "command",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "status": {"enum": ["completed", "failed", "timeout", "oom", "skipped"]},
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
        },
        "environment": {
            "type": "object",
            "required": [
                "git_commit",
                "python_version",
                "pandas_version",
                "numpy_version",
                "freshdata_version",
                "platform",
            ],
        },
        "samples_seconds": {
            "type": "array",
            "items": {"type": "number", "minimum": 0},
        },
        "command": {"type": "string"},
    },
}


def validate_result(payload: dict[str, Any]) -> None:
    import jsonschema  # noqa: PLC0415

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
