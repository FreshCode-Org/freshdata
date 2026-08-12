"""Deterministic natural-language context compiler (tier 0).

``fd.clean(df, context="CustomerID is unique. Never modify revenue.")`` is
powered by this package: prose is parsed into intents, column phrases are
resolved against the real schema, and the result is a typed, JSON-serializable
:class:`ContextPolicy` that lowers into the existing :class:`~freshdata.CleanConfig`
machinery. Fully offline, model-free, and dependency-free — unresolved or
unparsed sentences are always surfaced, never guessed at.

See ``ARCHITECTURE.md`` for how this package fits into the overall cleaning flow.
"""

from .compiler import apply_policy_to_config, compile_context, resolve_policy
from .parser import ParseResult, parse_context, split_sentences
from .resolve import Resolution, resolve_reference
from .types import (
    ColumnConstraint,
    ContextPolicy,
    IntentCandidate,
    PolicyError,
    PolicyIssue,
    Provenance,
    Thresholds,
    UnparsedSentence,
    UnresolvedRef,
)

__all__ = [
    "ColumnConstraint",
    "ContextPolicy",
    "IntentCandidate",
    "ParseResult",
    "PolicyError",
    "PolicyIssue",
    "Provenance",
    "Resolution",
    "Thresholds",
    "UnparsedSentence",
    "UnresolvedRef",
    "apply_policy_to_config",
    "compile_context",
    "parse_context",
    "resolve_policy",
    "resolve_reference",
    "split_sentences",
]
