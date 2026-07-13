"""Semantic-evidence helpers for the Peel normalizer (spec §7).

Semantic decisions ride into a :class:`CleanReport` as ``Action``\\s with
``step == "semantic"`` and a ``metadata`` dict carrying the proposal payload
(``raw_value``, ``proposed_value``, signed ``evidence`` entries, ``backend``,
optional ``calibrated_confidence``/``model_evidence``/``memory_key``). This
module turns those into display-neutral rows and a plain-language confidence
phrase, keeping every raw field for the audit layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .view import confidence_phrase

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..report import Action, CleanReport


def is_semantic(action: Action) -> bool:
    return action.step == "semantic" or "evidence" in (action.metadata or {})


def _decision(action: Action) -> str:
    """One of ``applied`` / ``review`` / ``ambiguous`` from the Action status."""
    if action.status == "suggested":
        return "review"
    if action.status == "skipped":
        return "ambiguous"
    return "applied"


def _source(action: Action) -> str:
    """Plain-language provenance: rules / cleaning memory / model."""
    meta = action.metadata or {}
    if meta.get("model_evidence"):
        return "model"
    if action.memory_influenced or meta.get("memory_key"):
        return "cleaning memory"
    return "rules"


def _confidence(action: Action) -> tuple[float, bool]:
    """Effective confidence and whether it was independently calibrated."""
    meta = action.metadata or {}
    calibrated = "calibrated_confidence" in meta
    value = meta.get("calibrated_confidence", action.confidence)
    return float(value or 0.0), calibrated


def confidence_label(action: Action) -> str:
    """``"moderate evidence (0.84)"`` — number and plain phrase together."""
    value, calibrated = _confidence(action)
    phrase = confidence_phrase(value, ambiguous=_decision(action) == "ambiguous")
    prefix = "" if calibrated else "~"
    return f"{phrase} ({prefix}{value:.2f})"


def attention_text(action: Action) -> str:
    """Glance-layer sentence for a semantic review/ambiguous item."""
    meta = action.metadata or {}
    raw = meta.get("raw_value")
    proposed = meta.get("proposed_value")
    if _decision(action) == "ambiguous":
        change = f"'{raw}' → no change made" if raw is not None else action.description
    elif raw is not None and proposed is not None:
        change = f"'{raw}' → '{proposed}'"
    else:
        change = action.description
    return f"{change} — {confidence_label(action)}"


def _evidence_summary(action: Action) -> str:
    """Signed evidence as one readable line, strongest first.

    Raw ``{kind, detail, weight}`` entries stay intact in the action metadata
    (audit layer); this is the inspect-layer digest.
    """
    signals = list((action.metadata or {}).get("evidence", ()))
    signals.sort(key=lambda s: -abs(float(s.get("weight", 0.0))))
    parts = []
    for signal in signals:
        weight = float(signal.get("weight", 0.0))
        detail = str(signal.get("detail", signal.get("kind", "")))
        parts.append(f"{weight:+.2f} {detail}")
    return "; ".join(parts)


def _max_evidence_weight(action: Action) -> float:
    signals = (action.metadata or {}).get("evidence", ())
    weights = [abs(float(s.get("weight", 0.0))) for s in signals]
    return max(weights) if weights else 0.0


def semantic_rows(rep: CleanReport) -> list[dict[str, Any]]:
    """Semantic proposals as display rows, review/ambiguous before applied."""
    order = {"ambiguous": 0, "review": 1, "applied": 2}
    rows = []
    for action in rep.actions:
        if not is_semantic(action):
            continue
        meta = action.metadata or {}
        decision = _decision(action)
        rows.append(
            {
                "decision": decision,
                "column": action.column or "",
                "change": attention_text(action).split(" — ")[0],
                "confidence": confidence_label(action),
                "source": _source(action),
                "reversible": action.reversible,
                "reason": action.rationale,
                "backend": meta.get("backend", ""),
                "evidence": _evidence_summary(action),
                "_order": (order[decision], -_max_evidence_weight(action), action.column or ""),
            }
        )
    rows.sort(key=lambda r: r["_order"])
    for row in rows:
        del row["_order"]
    return rows


def coverage_note(rep: CleanReport) -> str:
    """Plain-language line naming which evidence sources actually ran."""
    sources = {_source(a) for a in rep.actions if is_semantic(a)}
    if not sources:
        return ""
    ran = ", ".join(s for s in ("rules", "cleaning memory", "model") if s in sources)
    skipped = [
        e.get("fallback_step") or e.get("backend")
        for e in rep.fallback_events
        if "semantic" in str(e.get("fallback_step", "")).lower()
        or "model" in str(e.get("backend", "")).lower()
    ]
    note = f"checked with {ran}"
    if skipped:
        note += (
            "; FreshData continued without the optional model "
            f"({', '.join(str(s) for s in skipped)})"
        )
    return note


def count_by_decision(rep: CleanReport) -> dict[str, int]:
    counts = {"applied": 0, "review": 0, "ambiguous": 0}
    for action in rep.actions:
        if is_semantic(action):
            counts[_decision(action)] += 1
    return counts
