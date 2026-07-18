"""Typed value objects for the Semantic Cleaning Layer.

These are plain, dependency-light dataclasses: importing this module pulls in
nothing beyond the standard library. The richer behaviour (building context,
detecting issues, scoring, policy gating, applying) lives in sibling modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

#: Risk levels an action may carry.
RISKS = frozenset({"low", "medium", "high"})

#: Statuses a recorded semantic decision may have.
STATUSES = frozenset({"automatic", "suggested", "skipped", "approved"})

#: Recognised semantic issue types.
ISSUE_TYPES = frozenset(
    {
        "spelled_number",
        "category_synonym",
        "boolean_synonym",
        "unit_suffix",
        "currency_string",
        "date_phrase",
        "identifier_like",
        "unsafe_ambiguous",
        "email_format",
        "phone_format",
        "reference_value",
        "format_alignment",
        "encoding_repair",
        "numeric_format",
    }
)


@dataclass(frozen=True)
class SemanticEvidence:
    """One explainable reason behind a proposal or veto.

    *kind* is a short machine-readable tag (e.g. ``"column_role"``,
    ``"value_share"``, ``"pattern"``, ``"context_hint"``); *detail* is a human
    sentence; *weight* is the (signed) contribution to confidence in [-1, 1].
    """

    kind: str
    detail: str
    weight: float = 0.0


@dataclass(frozen=True)
class SemanticColumnInfo:
    """Everything the semantic experts need to know about one column.

    Built once per column by :mod:`freshdata.semantic.context` from the engine's
    role inference, cheap value-shape statistics, and any user-supplied hints.
    """

    name: str
    role: str  # engine role: id/target/datetime/boolean/numeric/text/categorical
    n_nonnull: int
    nunique: int | None
    high_cardinality: bool
    preserve: bool
    free_text: bool
    numeric_like: bool
    boolean_like: bool
    money_like: bool
    unit_like: bool
    identifier_like: bool
    dominant_unit: str | None = None
    date_like: bool = False
    # Merged user hints (``semantic_context``):
    semantic_type: str | None = None
    unit: str | None = None
    allowed_values: tuple[object, ...] = ()
    mutable: bool | None = None
    #: ISO region hint for locale-aware experts (e.g. ``"IN"`` from a
    #: ``LOCALE_FORMAT(phone, region="IN")`` constraint).
    region: str | None = None
    #: Email-shaped column: explicit ``semantic_type="email"`` hint, or an
    #: email-suggesting name whose values are dominated by ``@`` addresses.
    email_like: bool = False
    #: Phone-shaped column: only ever set from an explicit
    #: ``semantic_type="phone"`` hint (deterministic; never guessed from data).
    phone_like: bool = False
    #: Explicit day/month order for ambiguous numeric dates (``None`` = unset).
    dayfirst: bool | None = None
    #: Dataset-level ``semantic_context["currencies"]`` (accepted ISO codes),
    #: mirrored onto every column: a parsed currency outside this set is
    #: surfaced for review instead of auto-applied.
    allowed_currencies: tuple[str, ...] = ()
    #: Dataset-level ``semantic_context["reference_date"]``, mirrored onto every
    #: column so date experts (which only see column info) can resolve phrases
    #: like ``"today"`` without silently using the real wall-clock date.
    reference_date: str | None = None
    #: Declared via ``sensitive_columns``: value experts must not repair or
    #: quote values here — anomalies are review-routed without disclosure.
    sensitive: bool = False


@dataclass(frozen=True)
class SemanticContext:
    """Dataset-level semantic context assembled from config + column metadata."""

    dataset: str | None
    columns: Mapping[str, SemanticColumnInfo]
    auto_threshold: float
    review_threshold: float
    max_distinct_values: int
    sample_size: int
    privacy_policy: str
    mode: str
    id_columns: frozenset[str]
    preserve_columns: frozenset[str]
    target_column: str | None

    def info(self, column: str) -> SemanticColumnInfo | None:
        return self.columns.get(column)


@dataclass(frozen=True)
class SemanticIssue:
    """A value in a column that an expert flagged as semantically suspect."""

    column: str
    issue_type: str
    raw_value: object
    count: int
    expert: str


@dataclass(frozen=True)
class SemanticProposal:
    """A scored, explainable proposal to rewrite ``raw_value`` in *column*."""

    column: str
    raw_value: object
    proposed_value: object
    issue_type: str
    expert: str
    confidence: float
    risk: str
    rationale: str
    evidence: tuple[SemanticEvidence, ...] = ()
    count: int = 0
    status: str = "suggested"
    reversible: bool = True
    human_review: bool = False
    #: Which named backend generated the proposal ("deterministic" | "memory"
    #: | "embedding"). Provenance only — the policy gate treats all sources
    #: uniformly.
    backend: str = "deterministic"
    #: Calibrated confidence record (:class:`~freshdata.semantic.scoring.ActionConfidence`)
    #: attached by the calibration pass; None until calibrated.
    calibration: object | None = None
    #: Structured provenance for learned-profile proposals (profile id,
    #: support, learned precision, transform family).  Merged into the action
    #: metadata when the proposal is applied; None for other backends.
    provenance: Mapping[str, object] | None = None


@dataclass
class SemanticProposalSet:
    """A mutable collection of proposals with simple grouping helpers."""

    proposals: list[SemanticProposal] = field(default_factory=list)

    def add(self, proposal: SemanticProposal) -> None:
        self.proposals.append(proposal)

    def extend(self, proposals: list[SemanticProposal]) -> None:
        self.proposals.extend(proposals)

    def by_column(self) -> dict[str, list[SemanticProposal]]:
        grouped: dict[str, list[SemanticProposal]] = {}
        for p in self.proposals:
            grouped.setdefault(p.column, []).append(p)
        return grouped

    def counts_by_column(self) -> dict[str, int]:
        """Number of affected cells proposed per column."""
        counts: dict[str, int] = {}
        for p in self.proposals:
            counts[p.column] = counts.get(p.column, 0) + p.count
        return counts

    def __iter__(self):
        return iter(self.proposals)

    def __len__(self) -> int:
        return len(self.proposals)


@dataclass(frozen=True)
class SemanticPolicyDecision:
    """The policy gate's verdict on a single proposal."""

    proposal: SemanticProposal
    action: str  # "apply" | "suggest" | "skip"
    status: str  # final recorded status: automatic | suggested | skipped
    risk: str
    reason: str
    human_review: bool
