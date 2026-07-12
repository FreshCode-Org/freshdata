"""Peel view model — the display-neutral middle layer of freshdata's output system.

Every report object is *normalized* (see :mod:`freshdata.render.normalize`) into a
:class:`PeelView`, and every renderer (plain text, rich terminal, notebook HTML)
consumes only views. Normalization is pure and deterministic: the same report
always produces the same view, so renderers never re-derive or re-rank anything.

The vocabulary here is the single source of truth for Peel's shared grammar:

* result **statuses** (``CLEAN``/``CHANGED``/``REVIEW``/``BLOCKED``/``PARTIAL``/
  ``SKIPPED``/``FAILED``) — always rendered as text labels, never color alone;
* finding **severities** (``error``/``warning``/``review``/``info``) with one
  shared ranking used by every attention list;
* the **confidence ladder** mapping scores to plain-language phrases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

#: Result statuses, in display order. A view may carry several (e.g. CHANGED+REVIEW).
STATUSES = ("CLEAN", "CHANGED", "REVIEW", "BLOCKED", "PARTIAL", "SKIPPED", "FAILED")

#: Finding severities, most urgent first.
SEVERITIES = ("error", "warning", "review", "info")

#: Attention-domain ranks (spec §5.3): privacy/safety outrank everything,
#: cosmetic consistency comes last. Normalizers tag each item with one.
DOMAINS = ("privacy", "corruption", "policy", "reliability", "cosmetic")

_SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITIES)}
_DOMAIN_RANK = {name: i for i, name in enumerate(DOMAINS)}


def confidence_phrase(confidence: float, *, ambiguous: bool = False) -> str:
    """The plain-language rung of the confidence ladder for *confidence*.

    ``ambiguous=True`` forces the bottom rung regardless of score (a near-tie
    between candidates is ambiguous even when the winner scored well).
    """
    if ambiguous or confidence < 0.60:
        return "ambiguous — no change made" if ambiguous else "uncertain"
    if confidence >= 0.95:
        return "strong evidence"
    if confidence >= 0.80:
        return "moderate evidence"
    return "uncertain"


@dataclass(frozen=True)
class Metric:
    """One glance-layer number, optionally as a before/after pair."""

    label: str
    value: str
    before: str | None = None
    after: str | None = None


@dataclass(frozen=True)
class AttentionItem:
    """One ranked finding in the attention list.

    ``id`` is stable within a run (``W1``, ``R2``, ``D1``, ...) so terminal
    output, notebook output, and programmatic access all name the same thing.
    """

    id: str
    severity: str  # one of SEVERITIES
    subject: str  # column / frame / finding target ("" for table-level)
    text: str  # plain-language, one sentence
    domain: str = "reliability"  # one of DOMAINS
    count: int = 0  # affected cells/rows (0 = not applicable)
    detail: dict[str, Any] = field(default_factory=dict)  # audit-layer payload

    def sort_key(self) -> tuple[int, int, int, str, str, str]:
        """Shared comparator (spec §5.3): domain, severity, -count, name order,
        then the stable id so ranking is total (independent of input order)."""
        return (
            _DOMAIN_RANK.get(self.domain, len(DOMAINS)),
            _SEVERITY_RANK.get(self.severity, len(SEVERITIES)),
            -self.count,
            self.subject,
            self.text,
            self.id,
        )


@dataclass(frozen=True)
class Section:
    """One inspect-layer group. ``body`` is a thunk so detail costs nothing
    until a renderer actually expands it (spec §16)."""

    key: str  # machine name, e.g. "columns", "actions", "audit"
    title: str  # human name, e.g. "Column changes"
    body: Callable[[], list[dict[str, Any]]]  # lazy rows, JSON-friendly
    count: int = 0  # advertised size without evaluating body

    def rows(self) -> list[dict[str, Any]]:
        return self.body()


@dataclass(frozen=True)
class PeelView:
    """The display-neutral shape of one report (spec §12.2).

    ``audit_ref`` is the original report object — layer 3 is never a rendering.
    """

    kind: str  # "clean_report", "parse", "copilot", ...
    status: tuple[str, ...]  # subset of STATUSES, display order
    headline: str  # one-sentence impact
    metrics: tuple[Metric, ...]
    attention: tuple[AttentionItem, ...]  # already ranked
    next_step: str | None  # one runnable snippet, or None
    sections: tuple[Section, ...]
    audit_ref: Any = None
    banner: str | None = None  # PARTIAL/experimental banner text, if any

    def __post_init__(self) -> None:
        unknown = [s for s in self.status if s not in STATUSES]
        if unknown:
            raise ValueError(f"unknown status labels: {unknown}")

    @property
    def status_label(self) -> str:
        """The combined text label, e.g. ``"CHANGED · REVIEW"``."""
        return " · ".join(self.status)


def rank_attention(items: list[AttentionItem]) -> tuple[AttentionItem, ...]:
    """Sort *items* by the shared comparator. Deterministic; stable for ties."""
    return tuple(sorted(items, key=AttentionItem.sort_key))
