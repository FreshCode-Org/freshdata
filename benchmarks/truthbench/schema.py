from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import jsonschema


class TruthBenchSchemaError(ValueError):
    """A serialized TruthBench run violates its schema or integrity contract."""


_NULLABLE_STRING = {"type": ["string", "null"]}
_NULLABLE_BOOLEAN = {"type": ["boolean", "null"]}
_NULLABLE_NUMBER = {"type": ["number", "null"]}
_NULLABLE_STRING_ARRAY = {
    "oneOf": [
        {"type": "null"},
        {"type": "array", "items": {"type": "string"}},
    ]
}

_TYPED_VALUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "type",
        "dtype",
        "value",
        "display",
        "digest",
        "redacted",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "type": {"type": "string", "minLength": 1},
        "dtype": _NULLABLE_STRING,
        "value": {},
        "display": {"type": "string"},
        "digest": _NULLABLE_STRING,
        "redacted": {"type": "boolean"},
    },
}

_RECORD_PROPERTIES: dict[str, Any] = {
    "schema_version": {"const": 1},
    "record_id": {"type": "string", "minLength": 1},
    "run_id": {"type": "string", "minLength": 1},
    "fixture_id": {"type": "string", "minLength": 1},
    "case_id": _NULLABLE_STRING,
    "cell_id": {"type": "string", "minLength": 1},
    "domain": {"type": "string", "minLength": 1},
    "row_id": {"type": "string", "minLength": 1},
    "column": {"type": "string", "minLength": 1},
    "surface": {"type": "string", "minLength": 1},
    "repeat": {"type": "integer", "minimum": 0},
    "expected_disposition": {
        "enum": ["preserve", "repair", "flag", "review"],
    },
    "actual_disposition": {
        "enum": ["preserve", "repair", "flag", "review", None],
    },
    "sensitive": {"type": "boolean"},
    "input": _TYPED_VALUE_SCHEMA,
    "input_type": {"type": "string", "minLength": 1},
    "expected_output": {
        "oneOf": [{"type": "null"}, _TYPED_VALUE_SCHEMA],
    },
    "expected_output_type": _NULLABLE_STRING,
    "actual_output": {
        "oneOf": [{"type": "null"}, _TYPED_VALUE_SCHEMA],
    },
    "actual_output_type": _NULLABLE_STRING,
    "confidence": _NULLABLE_NUMBER,
    "risk": _NULLABLE_STRING,
    "status": _NULLABLE_STRING,
    "rule_id": _NULLABLE_STRING,
    "rationale": _NULLABLE_STRING,
    "evidence_kinds": _NULLABLE_STRING_ARRAY,
    "mutated": _NULLABLE_BOOLEAN,
    "detected": _NULLABLE_BOOLEAN,
    "quarantined": _NULLABLE_BOOLEAN,
    "human_review": _NULLABLE_BOOLEAN,
    "audit_required": _NULLABLE_BOOLEAN,
    "audit_complete": _NULLABLE_BOOLEAN,
    "audit_ids": _NULLABLE_STRING_ARRAY,
    "trust_before": _NULLABLE_NUMBER,
    "trust_after": _NULLABLE_NUMBER,
    "trust_delta": _NULLABLE_NUMBER,
    "requested_backend": _NULLABLE_STRING,
    "actual_backend": _NULLABLE_STRING,
    "fallback_events": _NULLABLE_STRING_ARRAY,
    "backend_differences": _NULLABLE_STRING_ARRAY,
    "normalized_decision_hash": _NULLABLE_STRING,
    "repeat_hash": _NULLABLE_STRING,
    "repeat_consistent": _NULLABLE_BOOLEAN,
}

_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(_RECORD_PROPERTIES),
    "properties": _RECORD_PROPERTIES,
}

_GATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "name", "passed", "failure_count", "failures"],
    "properties": {
        "schema_version": {"const": 1},
        "name": {"type": "string", "minLength": 1},
        "passed": {"type": "boolean"},
        "failure_count": {"type": "integer", "minimum": 0},
        "failures": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}

_ENVIRONMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["python"],
    "properties": {
        "python": {"type": "string", "minLength": 1},
    },
}

RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "run_id",
        "profile",
        "fixture_hashes",
        "required_backends",
        "records",
        "gates",
        "summary",
        "environment",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "run_id": {"type": "string", "minLength": 1},
        "profile": {"enum": ["release", "extended"]},
        "fixture_hashes": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {"type": "string", "minLength": 1},
        },
        "required_backends": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "records": {
            "type": "array",
            "minItems": 1,
            "items": _RECORD_SCHEMA,
        },
        "gates": {
            "type": "array",
            "minItems": 1,
            "items": _GATE_SCHEMA,
        },
        "summary": {
            "type": "object",
            "required": ["records", "overall_passed"],
            "properties": {
                "records": {"type": "integer", "minimum": 0},
                "overall_passed": {"type": "boolean"},
            },
        },
        "environment": _ENVIRONMENT_SCHEMA,
    },
}

_VALIDATOR = jsonschema.Draft202012Validator(RESULT_SCHEMA)


def _schema_path(error: jsonschema.ValidationError) -> str:
    path = "$"
    for component in error.absolute_path:
        if isinstance(component, int):
            path += f"[{component}]"
        else:
            path += f".{component}"
    return path


def _validate_schema(payload: Mapping[str, Any]) -> None:
    try:
        _VALIDATOR.validate(dict(payload))
    except jsonschema.ValidationError as exc:
        raise TruthBenchSchemaError(
            f"schema validation failed at {_schema_path(exc)}: {exc.message}"
        ) from exc


def _child_path(path: str, component: str | int) -> str:
    if isinstance(component, int):
        return f"{path}[{component}]"
    if component.isidentifier():
        return f"{path}.{component}"
    return f"{path}[<key>]"


def _validate_finite_numbers(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TruthBenchSchemaError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            component = key if isinstance(key, str) else "<key>"
            _validate_finite_numbers(item, _child_path(path, component))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_numbers(item, _child_path(path, index))


def validate_run(payload: Mapping[str, Any]) -> None:
    """Validate a serialized ``RunResult`` and its aggregate integrity."""

    copied_payload = dict(payload)
    _validate_finite_numbers(copied_payload)
    _validate_schema(copied_payload)

    records = copied_payload["records"]
    for record in records:
        if record["sensitive"]:
            for field in ("input", "expected_output", "actual_output"):
                value = record[field]
                if value is not None and not (
                    value["value"] is None
                    and value["display"] == "[REDACTED]"
                    and bool(value["digest"])
                    and value["redacted"] is True
                ):
                    raise TruthBenchSchemaError(f"sensitive record {field} must be redacted")

    ids = [record["record_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise TruthBenchSchemaError("duplicate decision record id")

    fixture_hashes = copied_payload["fixture_hashes"]
    for record in records:
        domain = record["domain"]
        fixture_domain = record["fixture_id"].rsplit(":", 1)[-1]
        if domain != fixture_domain or domain not in fixture_hashes:
            raise TruthBenchSchemaError(
                f"decision record fixture hash is missing for {record['fixture_id']}"
            )
    record_domains = {record["domain"] for record in records}
    if set(fixture_hashes) != record_domains:
        raise TruthBenchSchemaError("fixture hash domains do not match record domains")

    if copied_payload["summary"]["records"] != len(ids):
        raise TruthBenchSchemaError("record aggregate does not match records")

    for gate in copied_payload["gates"]:
        if gate["failure_count"] != len(gate["failures"]):
            raise TruthBenchSchemaError(
                f"gate failure aggregate does not match failures for {gate['name']}"
            )
        if gate["passed"] is not (gate["failure_count"] == 0):
            raise TruthBenchSchemaError(f"gate passed claim is inconsistent for {gate['name']}")

    passed = all(gate["passed"] for gate in copied_payload["gates"])
    if copied_payload["summary"]["overall_passed"] is not passed:
        raise TruthBenchSchemaError("overall gate claim is inconsistent")
