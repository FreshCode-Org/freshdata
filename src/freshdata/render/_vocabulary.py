"""Peel's plain-language vocabulary (spec §14).

Display layers 1-2 speak user language; the audit layer keeps exact technical
terms. This module is the single place display wording is defined so every
renderer stays consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..report import Action

#: step name → plain-language verb phrase, used when summarizing per column.
_STEP_PHRASES = {
    "missing": "filled missing values",
    "impute": "filled missing values",
    "outliers": "extreme values adjusted",
    "duplicates": "removed duplicate rows",
    "fix_dtypes": "fixed value types",
    "whitespace": "trimmed whitespace",
    "semantic": "standardized values",
}

#: audit/internal term → display phrase (spec §14.2), applied to prose.
TERMS = {
    "backend abstained": "no safe match was found",
    "fallback event": "FreshData continued without the optional engine/model",
    "calibration unavailable": "confidence could not be independently adjusted",
    "constraint violation": "this value breaks a rule you defined",
    "residual values": "values not resolved by earlier checks",
    "not materialized": "result kept in the engine",
}


def plain_step(action: Action) -> str:
    """A short plain-language phrase for *action*, for per-column summaries.

    Falls back to the action's own description — descriptions are already
    human sentences; the map only replaces the jargon-heavy step families.
    """
    phrase = _STEP_PHRASES.get(action.step)
    if phrase is None:
        return action.description
    if action.count:
        return f"{phrase} ({action.count:,})"
    return phrase
