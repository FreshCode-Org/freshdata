"""Route each column to the semantic experts eligible for it.

Routing is purely about *semantic suitability* (does the column look numeric,
boolean, monetary, ...). It deliberately does not consider column protection
(target/id/preserve) — that is the policy gate's job, so that protected columns
still produce auditable "skipped" decisions when a repair would have applied.
"""

from __future__ import annotations

from .experts import IDENTIFIER_EXPERT, VALUE_EXPERTS, SemanticExpert
from .types import SemanticColumnInfo


def route(info: SemanticColumnInfo) -> list[SemanticExpert]:
    """Return the experts whose ``applies(info)`` is True for this column.

    The identifier-protection veto expert is always considered last so its
    protective records sort after any value proposals for the same column.
    """
    experts: list[SemanticExpert] = [e for e in VALUE_EXPERTS if e.applies(info)]
    if IDENTIFIER_EXPERT.applies(info):
        experts.append(IDENTIFIER_EXPERT)
    return experts
