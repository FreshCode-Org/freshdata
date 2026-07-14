from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

JsonValue = Any


@dataclass(frozen=True)
class TypedValue:
    """A scalar encoded without erasing its Python/pandas representation."""

    kind: str
    value: JsonValue
    dtype: str | None
    display: str
    digest: str | None = None
    redacted: bool = False
    schema_version: int = field(default=1, init=False)

    @property
    def type_label(self) -> str:
        if self.dtype is None:
            return self.kind
        return f"{self.kind}[{self.dtype}]"

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "type": self.kind,
            "dtype": self.dtype,
            "value": self.value,
            "display": self.display,
            "digest": self.digest,
            "redacted": self.redacted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, JsonValue]) -> TypedValue:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported typed value schema version")
        return cls(
            kind=str(payload["type"]),
            dtype=None if payload["dtype"] is None else str(payload["dtype"]),
            value=payload["value"],
            display=str(payload["display"]),
            digest=None if payload["digest"] is None else str(payload["digest"]),
            redacted=bool(payload["redacted"]),
        )


def _dtype_name(dtype: Any, *, sensitive: bool) -> str | None:
    if dtype is None:
        return None
    normalized = pd.api.types.pandas_dtype(dtype)
    if isinstance(normalized, pd.CategoricalDtype):
        categories = normalized.categories
        if sensitive:
            encoded_categories: JsonValue = {"count": len(categories)}
        else:
            encoded_categories = [encode_typed(value).to_dict() for value in categories]
        category_json = canonical_json(encoded_categories)
        ordered = "true" if normalized.ordered else "false"
        return f"category[ordered={ordered};categories={category_json}]"
    if isinstance(normalized, pd.StringDtype):
        return f"string[{normalized.storage}]"
    return str(normalized)


def _timezone_identity(value: Any) -> str | None:
    if value is None:
        return None
    key = getattr(value, "key", None)
    if isinstance(key, str):
        return key
    return str(value)


def _encode_scalar(value: Any) -> tuple[str, JsonValue, str]:
    if value is pd.NA:
        return "pandas.NA", None, "<NA>"
    if value is pd.NaT:
        return "pandas.NaT", None, "NaT"
    if value is None:
        return "python.none", None, "None"

    if isinstance(value, np.bool_):
        scalar = bool(value)
        return "numpy.bool_", scalar, str(scalar)
    if isinstance(value, bool):
        return "python.bool", value, str(value)

    if isinstance(value, np.integer):
        return f"numpy.{value.dtype.name}", str(value), str(value)
    if isinstance(value, int):
        return "python.int", str(value), str(value)

    if isinstance(value, np.floating):
        kind = f"numpy.{value.dtype.name}"
        scalar = float(value)
        if math.isnan(scalar):
            return f"{kind}.nan", None, "NaN"
        if math.isinf(scalar):
            sign = "+" if scalar > 0 else "-"
            return f"{kind}.infinity", sign, f"{sign}Infinity"
        representation = repr(value.item())
        return kind, representation, representation
    if isinstance(value, float):
        if math.isnan(value):
            return "python.float.nan", None, "NaN"
        if math.isinf(value):
            sign = "+" if value > 0 else "-"
            return "python.float.infinity", sign, f"{sign}Infinity"
        representation = repr(value)
        return "python.float", representation, representation

    if isinstance(value, np.str_):
        scalar = str(value)
        return "numpy.str_", scalar, scalar
    if isinstance(value, str):
        return "python.str", value, value

    if isinstance(value, np.bytes_):
        scalar = bytes(value).hex()
        return "numpy.bytes_", scalar, repr(bytes(value))
    if isinstance(value, bytes):
        return "python.bytes", value.hex(), repr(value)

    if isinstance(value, pd.Timestamp):
        rendered = value.isoformat()
        encoded = {
            "iso": rendered,
            "timezone": _timezone_identity(value.tz),
        }
        return "pandas.Timestamp", encoded, rendered
    if isinstance(value, datetime):
        rendered = value.isoformat()
        encoded = {"iso": rendered, "timezone": _timezone_identity(value.tzinfo)}
        return "python.datetime", encoded, rendered
    if isinstance(value, date):
        rendered = value.isoformat()
        return "python.date", rendered, rendered
    if isinstance(value, time):
        rendered = value.isoformat()
        encoded = {"iso": rendered, "timezone": _timezone_identity(value.tzinfo)}
        return "python.time", encoded, rendered

    if isinstance(value, pd.Timedelta):
        rendered = value.isoformat()
        return "pandas.Timedelta", rendered, rendered
    if isinstance(value, np.datetime64):
        return f"numpy.{value.dtype}", str(value), str(value)
    if isinstance(value, np.timedelta64):
        return f"numpy.{value.dtype}", str(value), str(value)
    if isinstance(value, timedelta):
        rendered = repr(value)
        return "python.timedelta", rendered, rendered
    if isinstance(value, Decimal):
        rendered = str(value)
        return "python.Decimal", rendered, rendered

    raise TypeError(f"unsupported scalar type: {type(value).__name__}")


def encode_typed(
    value: Any,
    *,
    dtype: Any = None,
    sensitive: bool = False,
    digest_key: bytes | None = None,
) -> TypedValue:
    kind, encoded, display = _encode_scalar(value)
    dtype_name = _dtype_name(dtype, sensitive=sensitive)
    if not sensitive:
        return TypedValue(kind=kind, value=encoded, dtype=dtype_name, display=display)
    if digest_key is None:
        raise ValueError("digest_key is required for sensitive values")
    raw = TypedValue(kind=kind, value=encoded, dtype=dtype_name, display=display)
    digest = hmac.new(
        digest_key,
        canonical_json(raw).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return TypedValue(
        kind=kind,
        value=None,
        dtype=dtype_name,
        display="[REDACTED]",
        digest=digest,
        redacted=True,
    )


def exact_equal(
    left: Any,
    right: Any,
    *,
    left_dtype: Any = None,
    right_dtype: Any = None,
) -> bool:
    return encode_typed(left, dtype=left_dtype) == encode_typed(right, dtype=right_dtype)


def _json_safe(value: Any, path: str = "$") -> JsonValue:
    if isinstance(value, TypedValue):
        return _json_safe(value.to_dict(), path)
    if isinstance(value, Enum):
        return _json_safe(value.value, path)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains an unsupported non-string key")
            result[key] = _json_safe(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains unsupported type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_digest(value: Any, *, key: bytes) -> str:
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    encoded = canonical_json(encode_typed(value)).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()
