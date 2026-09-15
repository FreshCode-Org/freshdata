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


#: Action statuses that record a decision *not* to change data.
_NOT_APPLIED = frozenset({"skipped", "suggested"})


def changed_values(action: Action) -> bool:
    """``True`` when *action* actually changed cells or rows.

    Informational notes (``count == 0``, e.g. "preserved 3 missing value(s)")
    and actions that were only suggested or deliberately skipped are not
    changes.
    """
    return action.count > 0 and action.status not in _NOT_APPLIED


def plain_step(action: Action) -> str:
    """A short plain-language phrase for *action*, for per-column summaries.

    Falls back to the action's own description — descriptions are already
    human sentences; the map only replaces the jargon-heavy step families.
    Actions that changed nothing keep their description, so a preserved gap
    is never reworded as a fill.
    """
    phrase = _STEP_PHRASES.get(action.step)
    if phrase is None or not changed_values(action):
        return action.description
    return f"{phrase} ({action.count:,})"
