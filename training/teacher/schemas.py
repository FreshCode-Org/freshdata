"""Strict JSON schemas for teacher outputs.

Every teacher task declares one of these schemas; a response that does not
validate is discarded (the pipeline degrades to corruptor/hook data rather
than accepting free-form teacher text). Validation is strict: required keys,
type checks, and **no unknown keys**.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1"


class SchemaError(ValueError):
    """A teacher response failed strict schema validation."""


#: schema name -> {field: (type, required)}
_SCHEMAS: dict[str, dict[str, tuple[type | tuple[type, ...], bool]]] = {
    "RealismConfig": {
        "corruptor_name": (str, True),
        "recommended_share": (float, True),
        "recommended_params": (dict, False),
        "rationale": (str, True),
    },
    "ColumnRoleLabel": {
        "column_name": (str, True),
        "masked_samples": (list, True),
        "semantic_type": (str, True),
        "confidence": (float, True),
        "rationale": (str, False),
    },
    "ContextParaphraseBatch": {
        "intent": (str, True),
        "canonical_sentence": (str, True),
        "paraphrases": (list, True),
        "language": (str, True),
        "author": (str, True),
    },
    "AmbiguityJudgment": {
        "raw_value": (str, True),
        "candidates": (list, True),
        "verdict": (str, True),  # "ambiguous" | "resolvable" | "unrepairable"
        "preferred": ((str, type(None)), False),
        "rationale": (str, True),
    },
    "RationaleTemplateBatch": {
        "issue_type": (str, True),
        "templates": (list, True),
    },
    "RedTeamCaseBatch": {
        "target_metric": (str, True),
        "cases": (list, True),
    },
}

_ALLOWED_VERDICTS = frozenset({"ambiguous", "resolvable", "unrepairable"})


def schema_names() -> tuple[str, ...]:
    return tuple(sorted(_SCHEMAS))


def validate_payload(schema: str, payload: Any) -> dict[str, Any]:
    """Validate one teacher output payload; return it or raise SchemaError."""
    spec = _SCHEMAS.get(schema)
    if spec is None:
        raise SchemaError(f"unknown teacher schema {schema!r}; known: {schema_names()}")
    if not isinstance(payload, dict):
        raise SchemaError(f"{schema}: payload must be an object, got {type(payload).__name__}")
    unknown = set(payload) - set(spec)
    if unknown:
        raise SchemaError(f"{schema}: unknown keys {sorted(unknown)}")
    for field, (types, required) in spec.items():
        if field not in payload:
            if required:
                raise SchemaError(f"{schema}: missing required field {field!r}")
            continue
        value = payload[field]
        if isinstance(types, type) and types is float and isinstance(value, int):
            value = float(value)
            payload[field] = value
        if not isinstance(value, types):
            raise SchemaError(
                f"{schema}: field {field!r} must be {types}, got {type(value).__name__}"
            )
    if schema == "AmbiguityJudgment" and payload["verdict"] not in _ALLOWED_VERDICTS:
        raise SchemaError(f"AmbiguityJudgment: invalid verdict {payload['verdict']!r}")
    if schema == "ColumnRoleLabel" and not (0.0 <= float(payload["confidence"]) <= 1.0):
        raise SchemaError("ColumnRoleLabel: confidence must be in [0, 1]")
    return payload


def validate_batch(schema: str, payloads: list[Any]) -> list[dict[str, Any]]:
    return [validate_payload(schema, p) for p in payloads]
