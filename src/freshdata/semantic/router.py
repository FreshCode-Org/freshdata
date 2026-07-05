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

    Built-in value experts run first, then any *active plugin experts* (each
    isolated so a plugin bug or bad output cannot break the clean — see
    :mod:`freshdata.plugins`). The identifier-protection veto expert is always
    considered last so its protective records sort after any value proposals
    for the same column: a plugin may propose, but can never out-rank the
    protected-column veto or the policy gate.
    """
    from ..plugins import active_experts  # noqa: PLC0415 - avoid import cycle

    experts: list[SemanticExpert] = [e for e in VALUE_EXPERTS if e.applies(info)]
    experts.extend(e for e in active_experts() if e.applies(info))
    if IDENTIFIER_EXPERT.applies(info):
        experts.append(IDENTIFIER_EXPERT)
    return experts
