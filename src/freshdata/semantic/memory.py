"""Retrieval-backed semantic memory: learn accepted semantic repairs from a
:class:`~freshdata.CleanReport` and replay them as candidate proposals on
similar future data.

This extends :class:`freshdata.memory.CleaningMemory` (via
``value_patterns["semantic_repairs"]``) rather than building a separate memory
system. Matching is local and deterministic: exact normalized-value matching
first, with a lightweight :mod:`difflib`-based similarity fallback for minor
value drift. No embeddings, no external services, no network.

Retrieved proposals are only ever *candidates* — they still flow through
:func:`freshdata.semantic.policy.decide` exactly like deterministic proposals,
so target/id/preserve protection and the confidence floor always apply.
"""

from __future__ import annotations

import difflib
import math
import re
from typing import Any

import pandas as pd

from .experts import VALUE_EXPERTS
from .scoring import make_proposal
from .types import (
    SemanticColumnInfo,
    SemanticContext,
    SemanticEvidence,
    SemanticProposal,
    SemanticProposalSet,
)

#: Statuses that mean "this semantic repair was actually applied/approved".
_LEARNABLE_STATUSES = frozenset({"approved", "accepted", "automatic"})

#: Date-of-birth-like column names: a raw value here (unlike most other
#: semantic repairs) is often identifying on its own, so date_phrase repairs
#: on these columns are never persisted into memory — the live, in-run repair
#: still applies normally; only the reusable, exportable memory record is
#: skipped.
_SENSITIVE_DATE_COLUMN_RE = re.compile(r"dob|birth", re.I)

#: Below this normalized-value similarity, a memory repair is never retrieved.
SIMILARITY_THRESHOLD = 0.92

_SEPARATORS_RE = re.compile(r"[\s_\-/]+")


def _normalize_text(value: object) -> str:
    return _SEPARATORS_RE.sub(" ", str(value).strip().casefold())


def semantic_memory_key(column: str, issue_type: str, raw_value: object) -> str:
    """A stable, local identifier for a ``(column, issue, value)`` repair."""
    return f"{_normalize_text(column)}::{issue_type}::{_normalize_text(raw_value)}"


def _json_safe(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _value_kind(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return "datetime"
    return type(value).__name__


def _column_signature(info: SemanticColumnInfo | None) -> dict[str, Any]:
    if info is None:
        return {}
    return {"role": info.role, "semantic_type": info.semantic_type, "free_text": info.free_text}


def _value_signature(raw_value: object) -> dict[str, Any]:
    return {"normalized": _normalize_text(raw_value), "kind": _value_kind(raw_value)}


def build_semantic_metadata(
    proposal: SemanticProposal, info: SemanticColumnInfo | None
) -> dict[str, Any]:
    """The audit ``Action.metadata`` attached to every recorded semantic decision."""
    metadata: dict[str, Any] = {
        "raw_value": _json_safe(proposal.raw_value),
        "proposed_value": _json_safe(proposal.proposed_value),
        "proposed_type": _value_kind(proposal.proposed_value),
        "issue_type": proposal.issue_type,
        "expert": proposal.expert,
        "evidence": [
            {"kind": e.kind, "detail": e.detail, "weight": e.weight} for e in proposal.evidence
        ],
        "memory_key": semantic_memory_key(
            proposal.column, proposal.issue_type, proposal.raw_value
        ),
        "column_signature": _column_signature(info),
        "value_signature": _value_signature(proposal.raw_value),
    }
    if info is not None and info.region is not None:
        metadata["region"] = info.region
    metadata["backend"] = proposal.backend
    calibration = proposal.calibration
    if calibration is not None:
        metadata["raw_score"] = getattr(calibration, "raw", None)
        metadata["calibrated_confidence"] = getattr(calibration, "point", None)
        metadata["calibration_version"] = getattr(calibration, "calibration_version", None)
        metadata["features_hash"] = getattr(calibration, "features_hash", None)
    model_evidence = _model_evidence(proposal)
    if model_evidence:
        metadata["model_evidence"] = model_evidence
    return metadata


def _model_evidence(proposal: SemanticProposal) -> dict[str, Any]:
    """Model provenance for model-assisted proposals (empty for deterministic)."""
    out: dict[str, Any] = {}
    for e in proposal.evidence:
        if e.kind != "embedding":
            continue
        for token in e.detail.split():
            if "=" in token:
                key, _, value = token.partition("=")
                if key in {"model", "sha"}:
                    out["model_id" if key == "model" else "model_sha256"] = value
                elif key in {"cos", "margin"}:
                    try:
                        out[key] = float(value)
                    except ValueError:  # pragma: no cover - defensive
                        continue
        if "candidates" in out:  # pragma: no cover - single evidence per proposal
            break
    for e in proposal.evidence:
        if e.kind == "embedding_candidates":
            out["candidates"] = e.detail
    return out


def is_memory_replay(proposal: SemanticProposal) -> bool:
    """``True`` iff *proposal* was retrieved from cleaning memory.

    A side-channel via :class:`~freshdata.semantic.types.SemanticEvidence`
    rather than a new field, so the frozen ``SemanticProposal``/
    ``SemanticPolicyDecision`` types stay unchanged.
    """
    return any(e.kind == "memory_replay" for e in proposal.evidence)


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #


def _iter_action_like(decisions: Any):
    if hasattr(decisions, "actions"):
        yield from decisions.actions
    elif isinstance(decisions, list):
        yield from decisions
    # DataFrames (and anything else) carry no semantic metadata to learn from.


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def learn_semantic_repairs(decisions: Any) -> list[dict[str, Any]]:
    """Extract accepted semantic value repairs from prior clean decisions.

    Accepts a :class:`~freshdata.CleanReport`, a list of
    :class:`~freshdata.Action`, or a list of dicts — the same shapes
    :func:`freshdata.learn_cleaning_memory` already handles. Only decisions
    with ``step == "semantic"``, an approved-like ``status``, and a
    ``model_id`` starting with ``"semantic:"`` are learned; skipped/rejected
    decisions (including the ``identifier_like`` veto and ``unsafe_ambiguous``
    conflict records, which are never approved) are never learned.
    """
    records: list[dict[str, Any]] = []
    for action in _iter_action_like(decisions):
        if _field(action, "step") != "semantic":
            continue
        status = str(_field(action, "status", "")).lower()
        model_id = str(_field(action, "model_id", ""))
        if status not in _LEARNABLE_STATUSES or not model_id.startswith("semantic:"):
            continue
        metadata = _field(action, "metadata", {}) or {}
        issue_type = metadata.get("issue_type")
        if not issue_type or metadata.get("raw_value") is None:
            continue  # nothing concrete to replay (e.g. an identifier veto record)
        column = _field(action, "column")
        is_sensitive_date = (
            issue_type == "date_phrase"
            and isinstance(column, str)
            and _SENSITIVE_DATE_COLUMN_RE.search(column)
        )
        if is_sensitive_date:
            continue  # never persist raw dates-of-birth into reusable memory
        records.append({
            "column": column,
            "issue_type": issue_type,
            "raw_value": metadata.get("raw_value"),
            "proposed_value": metadata.get("proposed_value"),
            "proposed_type": metadata.get("proposed_type", "str"),
            "expert": metadata.get("expert"),
            "confidence": float(_field(action, "confidence", 0.0)),
            "risk": _field(action, "risk", "low"),
            "status": "approved",
            "memory_key": metadata.get("memory_key"),
            "column_signature": metadata.get("column_signature", {}),
            "value_signature": metadata.get("value_signature", {}),
        })
    return records


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

_EXPERTS_BY_ISSUE = {e.issue_type: e for e in VALUE_EXPERTS}


def _reconstruct_value(stored: object, proposed_type: str) -> object:
    if proposed_type == "datetime" and stored is not None:
        try:
            return pd.Timestamp(stored)
        except (ValueError, TypeError):
            return None
    return stored


def _similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def semantic_memory_proposals(
    df: pd.DataFrame, ctx: SemanticContext, memory: Any
) -> SemanticProposalSet:
    """Retrieve compatible semantic repairs from *memory* as candidate proposals.

    Conservative by design: nothing is retrieved unless ``memory.match(df).ok``,
    and a repair's column must still pass its expert's ``applies(info)`` check
    against the *current* data (not stale assumptions). Exact normalized-value
    matches keep the learned confidence; fuzzy matches (similarity >= 0.92) are
    capped at ``min(learned_confidence, similarity)``, so drift never inflates
    confidence. Risk is recomputed from that (possibly reduced) confidence by
    the normal scoring path, same as any deterministic proposal.
    """
    out = SemanticProposalSet()
    match = memory.match(df)
    if not match.ok:
        return out

    value_patterns = getattr(memory, "value_patterns", None) or {}
    repairs = value_patterns.get("semantic_repairs", [])
    if not repairs:
        return out

    for repair in repairs:
        column = repair.get("column")
        issue_type = repair.get("issue_type")
        if not column or column not in ctx.columns:
            continue
        info = ctx.columns[column]
        expert = _EXPERTS_BY_ISSUE.get(issue_type)
        if expert is None or not expert.applies(info):
            continue  # column is no longer eligible for this kind of repair

        proposed_type = repair.get("proposed_type", "str")
        proposed = _reconstruct_value(repair.get("proposed_value"), proposed_type)
        if proposed is None and proposed_type == "datetime":
            continue

        learned_raw = repair.get("raw_value")
        learned_norm = _normalize_text(learned_raw)
        learned_confidence = float(repair.get("confidence", 0.0))

        try:
            counts = df[column].value_counts(dropna=True)
        except TypeError:
            continue
        for raw, count in counts.items():
            if not isinstance(raw, str):
                continue
            similarity = _similarity(learned_norm, _normalize_text(raw))
            if similarity < SIMILARITY_THRESHOLD:
                continue
            retrieved_confidence = (
                learned_confidence if similarity >= 0.999 else min(learned_confidence, similarity)
            )
            evidence = (
                SemanticEvidence(
                    "memory_replay",
                    f"replayed approved semantic repair from cleaning memory "
                    f"{memory.dataset_id!r}",
                    0.0,
                ),
                SemanticEvidence(
                    "pattern",
                    f"matches learned value {learned_raw!r} (similarity {similarity:.2f})",
                    0.0,
                ),
            )
            out.add(
                make_proposal(
                    column=column,
                    raw_value=raw,
                    proposed_value=proposed,
                    issue_type=issue_type,
                    expert=str(repair.get("expert") or issue_type),
                    base_confidence=retrieved_confidence,
                    evidence=evidence,
                    count=int(count),
                    rationale=(
                        f"Replayed approved semantic repair from cleaning memory "
                        f"{memory.dataset_id!r}"
                    ),
                    info=info,
                    backend="memory",
                )
            )
    return out
