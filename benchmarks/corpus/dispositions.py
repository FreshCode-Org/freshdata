"""The disposition vocabulary shared by every FreshData evaluation harness.

FreshData deliberately has no single ``disposition`` enum in ``src/``: the
library speaks several narrower vocabularies, each correct for its own layer
(``fieldcheck.ACTIONS``, the semantic gate's ``apply/suggest/skip``, the plan's
``auto/suggest/skip/blocked``, ``findings`` severities).  This module does not
add a new public contract to the library.  It defines the vocabulary the
*evaluation* harnesses score against, plus explicit mappings from the
library's own words onto it, so that a gold label means exactly one thing.

Why six values and not the four TruthBench and Gauntlet already use:
``benchmarks/gauntlet/metrics.REVIEW_ACTIONS`` folds ``quarantine``,
``manual_review`` and ``reject`` together, so no gold corpus can currently say
"this row must be *rejected*, not merely queued for a human".  Those are
materially different outcomes for a caller -- a rejected row is not in the
accepted frame at all, a quarantined one is recoverable from the quarantine
sink, and a reviewed one is still in the accepted frame pending a decision.

Back-compatibility rule: a corpus case labelled :data:`Disposition.REVIEW`
stays satisfied by any of the three review-family outcomes, which is exactly
what ``REVIEW_ACTIONS`` means today.  Only a case that *specifically* demands
``QUARANTINE`` or ``REJECT`` is scored strictly.  Existing four-value fixtures
therefore keep their current meaning and keep passing.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Disposition",
    "REVIEW_FAMILY",
    "MUTATING",
    "from_field_action",
    "satisfies",
]


class Disposition(str, Enum):
    """What a knowledgeable human says should happen to one cell or row."""

    #: Valid data, possibly unusual. Must survive byte-identical.
    PRESERVE = "preserve"
    #: A safe deterministic repair exists and should be applied.
    REPAIR = "repair"
    #: Must be surfaced to the user but never auto-changed.
    FLAG = "flag"
    #: Ambiguous. Must reach a human; must not be auto-repaired or dropped.
    REVIEW = "review"
    #: Must be removed from the accepted frame but stay recoverable.
    QUARANTINE = "quarantine"
    #: Must be refused outright; not recoverable from the accepted output.
    REJECT = "reject"


#: The three outcomes that all satisfy a plain ``REVIEW`` label.
REVIEW_FAMILY = frozenset({Disposition.REVIEW, Disposition.QUARANTINE, Disposition.REJECT})

#: Dispositions under which the cell's value is allowed to change.
MUTATING = frozenset({Disposition.REPAIR})


#: ``fieldcheck.ACTIONS`` -> disposition.  ``normalize`` is a repair;
#: ``accept_with_warning`` is a flag; ``replace_with_null`` destroys the value
#: without preserving it, so it scores as a quarantine only when the report
#: also records the original (checked by the harness, not by this table).
_FIELD_ACTION_TO_DISPOSITION = {
    "accept": Disposition.PRESERVE,
    "accept_with_warning": Disposition.FLAG,
    "normalize": Disposition.REPAIR,
    "replace_with_null": Disposition.QUARANTINE,
    "quarantine": Disposition.QUARANTINE,
    "manual_review": Disposition.REVIEW,
    "reject": Disposition.REJECT,
}


def from_field_action(action: str) -> Disposition:
    """Map a ``fieldcheck`` remediation action onto a disposition.

    Raises ``KeyError`` with the offending action rather than guessing, so a
    new action added to ``fieldcheck.ACTIONS`` fails loudly here instead of
    being silently scored as something it is not.
    """
    try:
        return _FIELD_ACTION_TO_DISPOSITION[action]
    except KeyError:
        raise KeyError(
            f"no disposition mapping for fieldcheck action {action!r}; "
            f"known actions: {sorted(_FIELD_ACTION_TO_DISPOSITION)}"
        ) from None


def satisfies(expected: Disposition, observed: Disposition) -> bool:
    """Does ``observed`` satisfy a gold label of ``expected``?

    Exact match always satisfies.  A plain ``REVIEW`` expectation is satisfied
    by any review-family outcome, preserving today's ``REVIEW_ACTIONS``
    semantics.  Nothing else widens: a ``PRESERVE`` label is never satisfied by
    a repair, and a ``REJECT`` label is never satisfied by a mere review.
    """
    if expected is observed:
        return True
    if expected is Disposition.REVIEW:
        return observed in REVIEW_FAMILY
    return False
