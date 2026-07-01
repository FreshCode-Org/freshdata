"""The Semantic Cleaning Layer for FreshData.

An explainable, deterministic, **offline** stage that profiles semantic data
issues (spelled-out numbers, currency strings, unit suffixes, boolean/category
synonyms), generates *scored proposals*, gates them through policy, and applies
only the safe ones — recording every applied/suggested/skipped decision in the
``CleanReport``. An LLM never mutates the DataFrame.

The layer is off by default; it engages only when ``CleanConfig.semantic_mode``
is ``"assist"``, ``"review"``, or ``"auto"``. Importing this package pulls in
nothing beyond pandas and the standard library.

Extension points (intentionally not implemented in v1):

- ``router``/``experts`` accept new candidate generators (retrieval-backed
  corrections, private-LLM candidates) behind the same ``applies``/``propose``
  interface and ``semantic_backends`` config.
- ``policy`` is the single place to add semantic-memory replay and privacy
  redaction (``semantic_privacy_policy``) before any future external inference.
- proposals project cleanly to OpenLineage / quality-ops events.
"""

from __future__ import annotations

from .apply import run_semantic
from .profiler import plan_semantic, profile_semantic_issues
from .types import (
    SemanticColumnInfo,
    SemanticContext,
    SemanticEvidence,
    SemanticIssue,
    SemanticPolicyDecision,
    SemanticProposal,
    SemanticProposalSet,
)

__all__ = [
    "SemanticColumnInfo",
    "SemanticContext",
    "SemanticEvidence",
    "SemanticIssue",
    "SemanticPolicyDecision",
    "SemanticProposal",
    "SemanticProposalSet",
    "plan_semantic",
    "profile_semantic_issues",
    "run_semantic",
]
