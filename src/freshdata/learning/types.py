"""Typed containers for paired-data learning (Phase 4).

Everything a :class:`~freshdata.learning.profile.LearningProfile` stores is
declared here: the manifest, learned value maps, the example bank, and the
intermediate pipeline artifacts (alignment, diff, classification, demotion).

Note: the architecture spec asks for ``slots=True`` dataclasses, but the
package still supports Python 3.9 where ``dataclass(slots=...)`` does not
exist, so the repo convention (see :mod:`freshdata.repairplan`) of frozen
dataclasses without slots is followed instead.

Values learned from cells are stored as JSON-safe scalars.  Missing values
(``NaN``/``None``/``NaT``) are encoded with an explicit marker so a literal
string ``"None"`` in the data can never be confused with real missingness.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MISSING_MARKER",
    "PROFILE_FORMAT_VERSION",
    "SENSITIVE_SEMANTIC_TYPES",
    "TRANSFORM_FAMILIES",
    "AlignedPair",
    "AlignmentReport",
    "ClassifiedDiff",
    "DemotionRecord",
    "DiffSummary",
    "ExampleBank",
    "ExamplePair",
    "ProfileError",
    "ProfileFormatError",
    "ProfileManifest",
    "ProfileVersionError",
    "RowDiffSummary",
    "SchemaDiffSummary",
    "ValueDiff",
    "ValueMap",
    "ValueMapEntry",
    "decode_value",
    "encode_value",
]

#: Version written into ``manifest.json``; bump the major part on breaking
#: layout changes.  Loaders reject profiles whose major version is newer.
PROFILE_FORMAT_VERSION = "1.0"

#: Semantic types whose raw literals are never stored under ``privacy="mask"``.
SENSITIVE_SEMANTIC_TYPES = frozenset(
    {
        "email",
        "phone",
        "person_name",
        "national_id",
        "address",
        "postal_code",
        "free_text",
    }
)

#: Transform families the classifier can assign, ordered most-specific first.
#: The order doubles as the tie-break: when several families explain the same
#: raw -> clean pair the earliest one wins.
TRANSFORM_FAMILIES = (
    "email_normalize",
    "phone_normalize",
    "reference_normalize",
    "date_dayfirst_inference",
    "date_parse",
    "currency_parse",
    "unit_strip",
    "spelled_number",
    "boolean_synonym",
    "sentinel_to_missing",
    "allowed_value_map",
    "category_map",
    "numeric_rounding",
    "dtype_coercion",
    "case_fold",
    "whitespace",
    "missing_imputation",
    "row_drop_evidence",
    "unexplained",
)


class ProfileError(ValueError):
    """Base error for learning-profile problems."""


class ProfileFormatError(ProfileError):
    """A ``.fdprofile`` archive is corrupt, incomplete, or fails hash checks."""


class ProfileVersionError(ProfileError):
    """A ``.fdprofile`` was written by an unsupported future format version."""


# ---------------------------------------------------------------------------
# JSON-safe value encoding
# ---------------------------------------------------------------------------

#: Marker object used to represent missing cells inside JSON members.
MISSING_MARKER = {"__fd_missing__": True}


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    # pandas NaT / pd.NA without importing pandas eagerly at call sites that
    # only touch JSON: fall back to the repr contract used across the repo.
    return type(value).__name__ in {"NaTType", "NAType"}


def encode_value(value: object) -> object:
    """Encode one learned cell value as a JSON-safe object."""
    if _is_missing_scalar(value):
        return dict(MISSING_MARKER)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value
    # Timestamps, Decimals, numpy scalars: store the canonical string form.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return encode_value(item())
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def decode_value(value: object) -> object:
    """Invert :func:`encode_value` (missing marker becomes ``None``)."""
    if isinstance(value, Mapping) and value.get("__fd_missing__") is True:
        return None
    return value


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileManifest:
    """Integrity and provenance header stored as ``manifest.json``."""

    profile_version: str
    freshdata_version: str
    created_at: str
    dataset_id: str | None
    dataset_signature: str
    source_schema_hash: str
    clean_schema_hash: str
    context_hash: str | None
    privacy_mode: str
    contains_raw_values: bool
    compartments: tuple[str, ...]
    member_hashes: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "freshdata_version": self.freshdata_version,
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "dataset_signature": self.dataset_signature,
            "source_schema_hash": self.source_schema_hash,
            "clean_schema_hash": self.clean_schema_hash,
            "context_hash": self.context_hash,
            "privacy_mode": self.privacy_mode,
            "contains_raw_values": self.contains_raw_values,
            "compartments": list(self.compartments),
            "member_hashes": dict(self.member_hashes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProfileManifest:
        return cls(
            profile_version=str(data["profile_version"]),
            freshdata_version=str(data.get("freshdata_version", "")),
            created_at=str(data.get("created_at", "")),
            dataset_id=data.get("dataset_id"),
            dataset_signature=str(data.get("dataset_signature", "")),
            source_schema_hash=str(data.get("source_schema_hash", "")),
            clean_schema_hash=str(data.get("clean_schema_hash", "")),
            context_hash=data.get("context_hash"),
            privacy_mode=str(data.get("privacy_mode", "mask")),
            contains_raw_values=bool(data.get("contains_raw_values", False)),
            compartments=tuple(data.get("compartments", ())),
            member_hashes=dict(data.get("member_hashes", {})),
        )


# ---------------------------------------------------------------------------
# Value maps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueMapEntry:
    """One learned ``raw_value -> clean_value`` repair with its evidence."""

    raw_value: object
    clean_value: object
    support: int
    precision: float
    transform_family: str
    masked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_value": encode_value(self.raw_value),
            "clean_value": encode_value(self.clean_value),
            "support": self.support,
            "precision": round(self.precision, 6),
            "transform_family": self.transform_family,
            "masked": self.masked,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ValueMapEntry:
        return cls(
            raw_value=decode_value(data["raw_value"]),
            clean_value=decode_value(data["clean_value"]),
            support=int(data.get("support", 0)),
            precision=float(data.get("precision", 0.0)),
            transform_family=str(data.get("transform_family", "category_map")),
            masked=bool(data.get("masked", False)),
        )


@dataclass
class ValueMap:
    """Learned literal repairs for one column."""

    column: str
    entries: list[ValueMapEntry]
    min_precision: float
    min_support: int
    capped: bool
    masked: bool

    def replayable_entries(self) -> list[ValueMapEntry]:
        """Entries safe to replay: masked literals are audit-only."""
        return [e for e in self.entries if not e.masked]

    def lookup(self) -> dict[object, ValueMapEntry]:
        return {e.raw_value: e for e in self.replayable_entries()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "entries": [e.to_dict() for e in self.entries],
            "min_precision": self.min_precision,
            "min_support": self.min_support,
            "capped": self.capped,
            "masked": self.masked,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ValueMap:
        return cls(
            column=str(data["column"]),
            entries=[ValueMapEntry.from_dict(e) for e in data.get("entries", [])],
            min_precision=float(data.get("min_precision", 0.0)),
            min_support=int(data.get("min_support", 0)),
            capped=bool(data.get("capped", False)),
            masked=bool(data.get("masked", False)),
        )


# ---------------------------------------------------------------------------
# Example bank
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExamplePair:
    """A raw/clean pair kept as evidence only (never replayed as a rule)."""

    column: str
    raw_value: object
    clean_value: object
    transform_family: str
    support: int
    masked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "raw_value": encode_value(self.raw_value),
            "clean_value": encode_value(self.clean_value),
            "transform_family": self.transform_family,
            "support": self.support,
            "masked": self.masked,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExamplePair:
        return cls(
            column=str(data["column"]),
            raw_value=decode_value(data["raw_value"]),
            clean_value=decode_value(data["clean_value"]),
            transform_family=str(data.get("transform_family", "unexplained")),
            support=int(data.get("support", 0)),
            masked=bool(data.get("masked", False)),
        )


@dataclass
class ExampleBank:
    """Unexplained/demoted pairs plus optional embedding vectors."""

    examples: list[ExamplePair]
    vectors_path: str | None
    embedding_model_id: str | None
    masked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": [e.to_dict() for e in self.examples],
            "vectors_path": self.vectors_path,
            "embedding_model_id": self.embedding_model_id,
            "masked": self.masked,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExampleBank:
        return cls(
            examples=[ExamplePair.from_dict(e) for e in data.get("examples", [])],
            vectors_path=data.get("vectors_path"),
            embedding_model_id=data.get("embedding_model_id"),
            masked=bool(data.get("masked", True)),
        )


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@dataclass
class AlignmentReport:
    """How the messy and clean frames were paired up."""

    mode: str  # "key" | "positional" | "column_only"
    key: tuple[str, ...] | None
    matched_rows: int
    unmatched_messy: int
    unmatched_clean: int
    duplicate_messy_keys: int
    duplicate_clean_keys: int
    warnings: list[str] = field(default_factory=list)

    @property
    def row_level(self) -> bool:
        """True when row-aligned learning (cell diffs) is safe."""
        return self.mode in {"key", "positional"} and self.matched_rows > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "key": list(self.key) if self.key else None,
            "matched_rows": self.matched_rows,
            "unmatched_messy": self.unmatched_messy,
            "unmatched_clean": self.unmatched_clean,
            "duplicate_messy_keys": self.duplicate_messy_keys,
            "duplicate_clean_keys": self.duplicate_clean_keys,
            "warnings": list(self.warnings),
        }


@dataclass
class AlignedPair:
    """Row-aligned views of the training pair plus the alignment report."""

    messy_aligned: Any  # pd.DataFrame
    clean_aligned: Any  # pd.DataFrame
    alignment_report: AlignmentReport


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueDiff:
    """One distinct cell difference with its support count."""

    column: str
    raw_value: object
    clean_value: object
    support: int
    kind: str  # "value_change" | "missing_to_value" | "value_to_missing"

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "raw_value": encode_value(self.raw_value),
            "clean_value": encode_value(self.clean_value),
            "support": self.support,
            "kind": self.kind,
        }


@dataclass
class RowDiffSummary:
    """Row-level differences (only trustworthy under key alignment)."""

    dropped_rows: int
    added_rows: int
    keyed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dropped_rows": self.dropped_rows,
            "added_rows": self.added_rows,
            "keyed": self.keyed,
        }


@dataclass
class SchemaDiffSummary:
    """Column-level schema differences between messy and clean frames."""

    added_columns: tuple[str, ...]
    removed_columns: tuple[str, ...]
    shared_columns: tuple[str, ...]
    dtype_changes: dict[str, tuple[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_columns": list(self.added_columns),
            "removed_columns": list(self.removed_columns),
            "shared_columns": list(self.shared_columns),
            "dtype_changes": {c: list(v) for c, v in self.dtype_changes.items()},
        }


@dataclass
class DiffSummary:
    """All distinct differences between the aligned pair; never whole rows."""

    column_diffs: dict[str, list[ValueDiff]]
    row_diffs: RowDiffSummary
    schema_diffs: SchemaDiffSummary

    def total_cell_diffs(self) -> int:
        return sum(d.support for diffs in self.column_diffs.values() for d in diffs)


# ---------------------------------------------------------------------------
# Classification / demotion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassifiedDiff:
    """A value diff tagged with the transform family that explains it."""

    diff: ValueDiff
    family: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "diff": self.diff.to_dict(),
            "family": self.family,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class DemotionRecord:
    """Why a learned artifact was demoted or dropped during holdout eval."""

    target: str  # e.g. "value_map:status:'pend-ing'" or "rule:sentinel:notes"
    column: str
    family: str
    outcome: str  # "demoted_to_example" | "dropped" | "suggest_only"
    reason: str
    precision: float | None = None
    support: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "column": self.column,
            "family": self.family,
            "outcome": self.outcome,
            "reason": self.reason,
            "precision": self.precision,
            "support": self.support,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DemotionRecord:
        return cls(
            target=str(data.get("target", "")),
            column=str(data.get("column", "")),
            family=str(data.get("family", "")),
            outcome=str(data.get("outcome", "")),
            reason=str(data.get("reason", "")),
            precision=data.get("precision"),
            support=data.get("support"),
        )
