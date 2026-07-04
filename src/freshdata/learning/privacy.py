"""Privacy layer for learned profiles.

Under the default ``privacy="mask"`` no raw literal from a sensitive column
(email, phone, person_name, national_id, address, postal_code, free_text)
is ever written into a profile.  Rule-level learning (e.g. "this column is
an email column", "phones are region IN") carries no literals and replays
fine; literal value-map entries and examples on sensitive columns are stored
as deterministic HMAC tokens — auditable and countable, but not replayable
and not reversible.

Masking is deterministic per profile: the salt is derived from the training
pair's dataset signature so re-learning the same pair yields identical
tokens (and therefore identical profile hashes).
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Iterable, Mapping

import pandas as pd

from .types import SENSITIVE_SEMANTIC_TYPES

__all__ = [
    "SENSITIVE_SEMANTIC_TYPES",
    "derive_salt",
    "detect_sensitive_columns",
    "is_masked_token",
    "mask_value",
]

_MASK_PREFIX = "fdmask"
_MASK_RE = re.compile(r"^fdmask:[a-z_]+:[0-9a-f]{16}$")

#: Column-name fragments mapped to sensitive semantic types.  Value-based
#: detection (enterprise detect_pii) takes precedence; these catch columns
#: whose values are not self-identifying (names, addresses, ids).
_NAME_HINTS: tuple[tuple[str, str], ...] = (
    ("email", "email"),
    ("e_mail", "email"),
    ("phone", "phone"),
    ("mobile", "phone"),
    ("telephone", "phone"),
    ("contact_no", "phone"),
    ("first_name", "person_name"),
    ("last_name", "person_name"),
    ("full_name", "person_name"),
    ("customer_name", "person_name"),
    ("patient_name", "person_name"),
    ("person", "person_name"),
    ("surname", "person_name"),
    ("aadhaar", "national_id"),
    ("aadhar", "national_id"),
    ("pan_no", "national_id"),
    ("passport", "national_id"),
    ("ssn", "national_id"),
    ("national_id", "national_id"),
    ("voter_id", "national_id"),
    ("address", "address"),
    ("street", "address"),
    ("addr_line", "address"),
    ("postal", "postal_code"),
    ("pincode", "postal_code"),
    ("pin_code", "postal_code"),
    ("zip", "postal_code"),
    ("notes", "free_text"),
    ("comment", "free_text"),
    ("description", "free_text"),
    ("remarks", "free_text"),
    ("feedback", "free_text"),
)

#: enterprise ``detect_pii`` entity types -> profile sensitive types.
_ENTITY_MAP: Mapping[str, str] = {
    "EMAIL": "email",
    "PHONE": "phone",
    "PERSON": "person_name",
    "NAME": "person_name",
    "ADDRESS": "address",
    "POSTAL_CODE": "postal_code",
    "AADHAAR": "national_id",
    "PAN": "national_id",
    "SSN": "national_id",
    "PASSPORT": "national_id",
    "NATIONAL_ID": "national_id",
}

_FREE_TEXT_MIN_AVG_LEN = 40.0
_FREE_TEXT_MIN_UNIQUE_RATIO = 0.8


def _name_hint(column: str) -> str | None:
    lowered = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower())
    for fragment, semantic_type in _NAME_HINTS:
        if fragment in lowered:
            return semantic_type
    return None


def _looks_free_text(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) < 3 or non_null.dtype.kind not in "OU":
        return False
    text = non_null.astype(str)
    avg_len = float(text.str.len().mean())
    unique_ratio = text.nunique() / len(text)
    return avg_len >= _FREE_TEXT_MIN_AVG_LEN and unique_ratio >= _FREE_TEXT_MIN_UNIQUE_RATIO


def _pii_scan_types(df: pd.DataFrame) -> dict[str, str]:
    try:
        from ..enterprise.privacy import detect_pii  # noqa: PLC0415 - heavy import
    except ImportError:  # pragma: no cover - enterprise always ships in-tree
        return {}
    try:
        report = detect_pii(df)
        by_column = report.by_column()
    except Exception:  # pragma: no cover - detection must never break learning
        return {}
    found: dict[str, str] = {}
    for column, entities in by_column.items():
        for entity in entities:
            mapped = _ENTITY_MAP.get(str(entity.entity_type).upper())
            if mapped is not None:
                found[str(column)] = mapped
                break
    return found


def detect_sensitive_columns(
    df: pd.DataFrame,
    *,
    extra_sensitive: Iterable[str] = (),
) -> dict[str, str]:
    """Map column name -> sensitive semantic type for the given frame.

    Combines the enterprise PII scanner (value evidence) with column-name
    hints and a free-text heuristic.  ``extra_sensitive`` columns are always
    included (as ``free_text`` unless a stronger type is detected).
    """
    sensitive: dict[str, str] = {}
    for column in df.columns:
        hint = _name_hint(str(column))
        if hint is not None:
            sensitive[str(column)] = hint
    for column, semantic_type in _pii_scan_types(df).items():
        sensitive[column] = semantic_type
    for column in df.columns:
        name = str(column)
        if name not in sensitive and _looks_free_text(df[column]):
            sensitive[name] = "free_text"
    for name in extra_sensitive:
        sensitive.setdefault(str(name), "free_text")
    return sensitive


def derive_salt(dataset_signature: str) -> str:
    """Deterministic per-profile masking salt (stable across re-learning)."""
    digest = hashlib.sha256(f"freshdata-profile-salt:{dataset_signature}".encode())
    return digest.hexdigest()[:32]


def mask_value(value: object, *, salt: str, semantic_type: str) -> str:
    """One-way deterministic token for a sensitive literal."""
    payload = repr(value).encode("utf-8", "replace")
    digest = hmac.new(salt.encode(), payload, hashlib.sha256).hexdigest()[:16]
    return f"{_MASK_PREFIX}:{semantic_type}:{digest}"


def is_masked_token(value: object) -> bool:
    return isinstance(value, str) and bool(_MASK_RE.match(value))
