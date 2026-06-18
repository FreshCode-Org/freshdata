"""Duplicate and replay defense for ingestion pipelines.

This module separates three operationally different problems:

* exact duplicate rows, which can be deterministically identified;
* near duplicate entities, which are always routed to review by default;
* replayed ingestion batches, which are detected through source manifests.
"""

from __future__ import annotations

import difflib
import hashlib
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from .review import ReviewDataset, ReviewQueue

DuplicateKind = Literal["exact_row", "near_entity", "replayed_batch"]


@dataclass(frozen=True)
class IdempotencyKey:
    """Deterministic row-level key for idempotent ingestion."""

    key: str
    columns: tuple[str, ...]
    row_index: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "key": self.key,
            "columns": list(self.columns),
            "row_index": _json_value(self.row_index),
        }


@dataclass(frozen=True)
class BatchManifest:
    """Source-load manifest used to detect replayed batches."""

    source_id: str
    load_id: str
    batch_fingerprint: str
    row_count: int
    columns: tuple[str, ...]
    created_at: str
    sample_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a manifest dictionary suitable for durable storage."""
        return {
            "source_id": self.source_id,
            "load_id": self.load_id,
            "batch_fingerprint": self.batch_fingerprint,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "created_at": self.created_at,
            "sample_keys": list(self.sample_keys),
        }


@dataclass(frozen=True)
class DuplicateExplanation:
    """Auditable explanation for a duplicate or replay finding."""

    duplicate_type: DuplicateKind
    row_indices: tuple[Any, ...] = ()
    idempotency_key: str = ""
    batch_fingerprint: str = ""
    confidence: float = 1.0
    reason: str = ""
    review_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly duplicate explanation."""
        return {
            "duplicate_type": self.duplicate_type,
            "row_indices": [_json_value(row) for row in self.row_indices],
            "idempotency_key": self.idempotency_key,
            "batch_fingerprint": self.batch_fingerprint,
            "confidence": self.confidence,
            "reason": self.reason,
            "review_required": self.review_required,
        }


@dataclass(frozen=True)
class DuplicateDefenseReport:
    """Combined duplicate and replay analysis result."""

    exact_duplicates: tuple[DuplicateExplanation, ...] = ()
    near_duplicates: tuple[DuplicateExplanation, ...] = ()
    replayed_batches: tuple[DuplicateExplanation, ...] = ()
    manifest: BatchManifest | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return all findings as a DataFrame."""
        findings = [
            *self.exact_duplicates,
            *self.near_duplicates,
            *self.replayed_batches,
        ]
        return pd.DataFrame([finding.to_dict() for finding in findings])

    def to_review_dataset(self) -> ReviewDataset:
        """Route fuzzy entity and replay findings to a review dataset."""
        queue = ReviewQueue(dataset_id="freshdata-duplicate-review")
        for finding in [*self.near_duplicates, *self.replayed_batches]:
            queue.add_candidate(
                review_id=_hash_payload(finding.to_dict()),
                source="duplicate_defense",
                patch_id=finding.idempotency_key or finding.batch_fingerprint,
                candidate_change=f"Resolve {finding.duplicate_type}",
                required_decision="Approve merge/drop, reject, or request data-owner review.",
                confidence=finding.confidence,
                reason=finding.reason,
                risk="high" if finding.duplicate_type == "replayed_batch" else "medium",
                row=finding.row_indices,
            )
        return queue.to_dataset()


class DuplicateDefense:
    """Generate deterministic duplicate and replay defenses for tabular data."""

    def __init__(self, *, near_threshold: float = 0.92, max_near_pairs: int = 10_000) -> None:
        self.near_threshold = near_threshold
        self.max_near_pairs = max_near_pairs

    def idempotency_key(
        self,
        row: pd.Series | dict[str, Any],
        *,
        key_columns: tuple[str, ...] | None = None,
    ) -> IdempotencyKey:
        """Return a deterministic row key from selected columns or all columns."""
        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        columns = key_columns or tuple(sorted(str(column) for column in data))
        payload = {column: _json_value(data.get(column)) for column in columns}
        return IdempotencyKey(key=_hash_payload(payload), columns=tuple(columns))

    def batch_fingerprint(
        self,
        df: pd.DataFrame,
        *,
        key_columns: tuple[str, ...] | None = None,
        source_id: str = "",
    ) -> str:
        """Return a deterministic fingerprint for the whole batch."""
        keys = [
            self.idempotency_key(row, key_columns=key_columns).key
            for _, row in df.iterrows()
        ]
        payload = {
            "source_id": source_id,
            "columns": [str(column) for column in df.columns],
            "row_count": len(df),
            "row_keys": sorted(keys),
        }
        return _hash_payload(payload)

    def build_manifest(
        self,
        df: pd.DataFrame,
        *,
        source_id: str,
        load_id: str,
        key_columns: tuple[str, ...] | None = None,
    ) -> BatchManifest:
        """Create a source-load manifest for durable replay defense."""
        keys = [
            self.idempotency_key(row, key_columns=key_columns).key
            for _, row in df.head(25).iterrows()
        ]
        return BatchManifest(
            source_id=source_id,
            load_id=load_id,
            batch_fingerprint=self.batch_fingerprint(
                df,
                key_columns=key_columns,
                source_id=source_id,
            ),
            row_count=len(df),
            columns=tuple(str(column) for column in df.columns),
            created_at=datetime.now(timezone.utc).isoformat(),
            sample_keys=tuple(keys),
        )

    def detect_exact_duplicates(
        self,
        df: pd.DataFrame,
        *,
        key_columns: tuple[str, ...] | None = None,
    ) -> tuple[DuplicateExplanation, ...]:
        """Detect exact duplicate rows by deterministic idempotency key."""
        groups: dict[str, list[Any]] = {}
        for index, row in df.iterrows():
            key = self.idempotency_key(row, key_columns=key_columns).key
            groups.setdefault(key, []).append(index)
        explanations = []
        for key, rows in groups.items():
            if len(rows) <= 1:
                continue
            explanations.append(DuplicateExplanation(
                duplicate_type="exact_row",
                row_indices=tuple(rows),
                idempotency_key=key,
                confidence=1.0,
                reason="Rows share the same deterministic idempotency key.",
                review_required=False,
            ))
        return tuple(explanations)

    def detect_near_duplicate_entities(
        self,
        df: pd.DataFrame,
        *,
        entity_columns: tuple[str, ...],
    ) -> tuple[DuplicateExplanation, ...]:
        """Detect fuzzy entity candidates and route every finding to review."""
        explanations: list[DuplicateExplanation] = []
        checked = 0
        records = [
            (index, _entity_text(row, entity_columns))
            for index, row in df.iterrows()
        ]
        for (left_idx, left_text), (right_idx, right_text) in itertools.combinations(records, 2):
            checked += 1
            if checked > self.max_near_pairs:
                break
            if not left_text or not right_text:
                continue
            score = _similarity(left_text, right_text)
            if score < self.near_threshold:
                continue
            explanations.append(DuplicateExplanation(
                duplicate_type="near_entity",
                row_indices=(left_idx, right_idx),
                idempotency_key=_hash_payload([left_text, right_text]),
                confidence=round(score, 4),
                reason="Entity fields are similar; route to review before merging.",
                review_required=True,
            ))
        return tuple(explanations)

    def detect_replayed_batch(
        self,
        manifest: BatchManifest,
        prior_manifests: tuple[BatchManifest, ...],
    ) -> tuple[DuplicateExplanation, ...]:
        """Detect replayed or double-loaded batches by manifest fingerprint."""
        findings = []
        for prior in prior_manifests:
            same_fingerprint = prior.batch_fingerprint == manifest.batch_fingerprint
            same_source = prior.source_id == manifest.source_id
            if same_fingerprint and same_source:
                findings.append(DuplicateExplanation(
                    duplicate_type="replayed_batch",
                    batch_fingerprint=manifest.batch_fingerprint,
                    confidence=1.0,
                    reason=(
                        f"Load {manifest.load_id!r} matches prior load "
                        f"{prior.load_id!r} for source {manifest.source_id!r}."
                    ),
                    review_required=True,
                ))
        return tuple(findings)

    def analyze(
        self,
        df: pd.DataFrame,
        *,
        source_id: str,
        load_id: str,
        key_columns: tuple[str, ...] | None = None,
        entity_columns: tuple[str, ...] = (),
        prior_manifests: tuple[BatchManifest, ...] = (),
    ) -> DuplicateDefenseReport:
        """Run exact duplicate, fuzzy entity, and replay checks together."""
        manifest = self.build_manifest(
            df,
            source_id=source_id,
            load_id=load_id,
            key_columns=key_columns,
        )
        near: tuple[DuplicateExplanation, ...] = ()
        if entity_columns:
            near = self.detect_near_duplicate_entities(df, entity_columns=entity_columns)
        return DuplicateDefenseReport(
            exact_duplicates=self.detect_exact_duplicates(df, key_columns=key_columns),
            near_duplicates=near,
            replayed_batches=self.detect_replayed_batch(manifest, prior_manifests),
            manifest=manifest,
        )


def _entity_text(row: pd.Series, columns: tuple[str, ...]) -> str:
    parts = []
    for column in columns:
        value = row.get(column)
        if pd.notna(value):
            parts.append(str(value).strip().lower())
    return " ".join(parts)


def _similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def _hash_payload(payload: Any) -> str:
    text = json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert a Python/pandas/numpy value into JSON-friendly primitives.

    This function intentionally handles array-like inputs (numpy arrays and
    pandas Series) before calling ``pd.isna`` to avoid ambiguous truth-value
    checks that raise DeprecationWarning in newer pandas/numpy. Scalars are
    converted to None when missing (pd.NA/np.nan), and containers are
    recursively converted.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value

    # Handle array-like inputs (pandas Series, numpy arrays, lists/tuples/sets)
    # before calling pd.isna so we don't get an array result used in a truth test.
    try:
        if isinstance(value, pd.Series):
            return [_json_value(v) for v in value.tolist()]
        if isinstance(value, np.ndarray):
            return [_json_value(v) for v in value.tolist()]
        if isinstance(value, (list, tuple, set)):
            return [_json_value(v) for v in value]
    except Exception:
        # defensive: fall through to scalar handling
        pass

    # Now safe to call pd.isna for scalar-like objects only.
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}

    # Fallback: stringify unknown objects to keep deterministic output.
    return str(value)
