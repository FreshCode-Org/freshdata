"""Properties of the semantic policy gate (`policy.decide`) that had no test.

`decide()` is the single place where a proposed repair becomes ``apply``,
``suggest`` or ``skip``. Mutation testing `src/freshdata/semantic/policy.py`
left a cluster of survivors inside that gate: the identifier carve-out, the
review-threshold boundary, and every ``human_review`` flag the gate hands to
the audit trail. A surviving mutant there is not cosmetic -- it silently
widens what the gate is willing to mutate, or drops the "a human must look at
this" marker from a skipped row.

Each test names the mutation it defeats, so if an assertion is ever weakened
the reason it existed is on the page. Companion file:
``tests/test_policy_guard_mutants.py`` (payload-preservation and the
target/preserve branches of ``_protection``); the cases here do not repeat it.
"""

from __future__ import annotations

import pytest

from freshdata.config import CleanConfig
from freshdata.semantic.policy import _protection, decide
from freshdata.semantic.types import (
    SemanticColumnInfo,
    SemanticContext,
    SemanticProposal,
)

IDENTIFIER_VETO = "identifier column is protected (set mutable=True in semantic_context to allow)"

CFG = CleanConfig(verbose=False)


def _proposal(
    column="code",
    raw="A-1",
    proposed="A1",
    issue_type="format_alignment",
    *,
    confidence=0.99,
    risk="low",
    rationale="t",
):
    return SemanticProposal(
        column=column,
        raw_value=raw,
        proposed_value=proposed,
        issue_type=issue_type,
        expert="t",
        confidence=confidence,
        risk=risk,
        rationale=rationale,
    )


def _info(**kw) -> SemanticColumnInfo:
    base = {
        "name": "code",
        "role": "categorical",
        "n_nonnull": 3,
        "nunique": 3,
        "high_cardinality": False,
        "preserve": False,
        "free_text": False,
        "numeric_like": False,
        "boolean_like": False,
        "money_like": False,
        "unit_like": False,
        "identifier_like": False,
    }
    base.update(kw)
    return SemanticColumnInfo(**base)


def _ctx(
    columns=None,
    *,
    id_columns=(),
    preserve_columns=(),
    target=None,
    mode="auto",
    auto_threshold=0.95,
    review_threshold=0.70,
):
    return SemanticContext(
        dataset=None,
        columns=columns or {},
        auto_threshold=auto_threshold,
        review_threshold=review_threshold,
        max_distinct_values=50,
        sample_size=1000,
        privacy_policy="none",
        mode=mode,
        id_columns=frozenset(id_columns),
        preserve_columns=frozenset(preserve_columns),
        target_column=target,
    )


# -- identifier protection may not be narrowed to the explicit list ---------


def test_a_detected_identifier_is_protected_without_being_listed():
    """Kills policy bool#8 (`or` -> `and`) in the ``is_id`` expression.

    This is the "ID-protection removed" class. With `and`, a column only
    counted as an identifier when it was BOTH listed in ``id_columns`` AND
    detected as identifier-like -- so a key column the profiler recognised on
    its own, which the user never had to enumerate, lost its veto entirely and
    became mutable by any expert.
    """
    ctx = _ctx(columns={"code": _info(identifier_like=True)}, id_columns=())
    proposal = _proposal(issue_type="category_synonym", raw="A-1", proposed="B-2")
    assert _protection(proposal, ctx) == IDENTIFIER_VETO
    decision = decide(proposal, CFG, ctx)
    assert (decision.action, decision.status) == ("skip", "skipped")
    assert decision.reason == IDENTIFIER_VETO


def test_a_listed_identifier_is_protected_without_any_column_info():
    """Also kills policy bool#8 from the other side.

    ``id_columns`` is the user's explicit declaration; it must hold even for a
    column the profiler built no :class:`SemanticColumnInfo` for.
    """
    ctx = _ctx(columns={}, id_columns=("code",))
    proposal = _proposal(issue_type="category_synonym", raw="A-1", proposed="B-2")
    assert _protection(proposal, ctx) == IDENTIFIER_VETO
    assert decide(proposal, CFG, ctx).action == "skip"


def test_an_identifier_is_opted_in_only_by_mutable_being_exactly_true():
    """Kills policy bool#10 (`and` -> `or`) in the ``mutable is True`` opt-in.

    With `or`, the first operand (``info is not None``) short-circuits the
    whole test, so merely HAVING column info was read as "the user opted this
    identifier in" and every profiled identifier lost its veto. Only an
    explicit ``mutable=True`` hint may open the column.
    """
    ctx = _ctx(columns={"code": _info(identifier_like=True, mutable=None)}, id_columns=("code",))
    proposal = _proposal(issue_type="category_synonym", raw="A-1", proposed="B-2")
    assert _protection(proposal, ctx) == IDENTIFIER_VETO
    assert decide(proposal, CFG, ctx).action == "skip"


def test_an_identifier_with_mutable_true_is_let_through():
    """The opt-in must still work: ``mutable=True`` clears the identifier veto."""
    ctx = _ctx(columns={"code": _info(identifier_like=True, mutable=True)}, id_columns=("code",))
    proposal = _proposal(issue_type="category_synonym", raw="A-1", proposed="B-2")
    assert _protection(proposal, ctx) is None
    assert decide(proposal, CFG, ctx).action == "apply"


# -- the format-alignment carve-out needs BOTH conditions -------------------


def test_a_format_alignment_that_changes_the_payload_is_still_vetoed():
    """Kills policy bool#11 (`and` -> `or`), issue-type side.

    With `or`, merely being labelled ``format_alignment`` was enough to pass
    the identifier carve-out, whatever the values were -- an expert could
    rewrite ``A-1`` to ``A-2`` inside a protected key column. The carve-out
    exists only for repairs whose payload is verified unchanged.
    """
    ctx = _ctx(columns={"code": _info(identifier_like=True)}, id_columns=("code",))
    proposal = _proposal(issue_type="format_alignment", raw="A-1", proposed="A-2")
    assert _protection(proposal, ctx) == IDENTIFIER_VETO
    assert decide(proposal, CFG, ctx).action == "skip"


def test_a_payload_preserving_repair_of_another_issue_type_is_still_vetoed():
    """Kills policy bool#11 (`and` -> `or`), payload side.

    Being payload-preserving is necessary but not sufficient: the carve-out is
    scoped to ``format_alignment``. With `or`, any other expert's
    payload-preserving rewrite (e.g. an encoding repair) slipped into a
    protected identifier column as well.
    """
    ctx = _ctx(columns={"code": _info(identifier_like=True)}, id_columns=("code",))
    proposal = _proposal(issue_type="encoding_repair", raw="A-1", proposed="A1")
    assert _protection(proposal, ctx) == IDENTIFIER_VETO
    assert decide(proposal, CFG, ctx).action == "skip"


def test_a_verified_format_alignment_still_passes_the_carve_out():
    """Positive control for the carve-out both bool#11 mutants widen."""
    ctx = _ctx(columns={"code": _info(identifier_like=True)}, id_columns=("code",))
    proposal = _proposal(issue_type="format_alignment", raw="A-1", proposed="A1")
    assert _protection(proposal, ctx) is None
    assert decide(proposal, CFG, ctx).action == "apply"


# -- the review threshold is a floor, not a ceiling -------------------------


def test_confidence_exactly_at_the_review_threshold_is_not_skipped():
    """Kills policy cmp#7 (`<` -> `<=`) on the confidence floor.

    The gate's own boundary: ``review_threshold`` is the lowest confidence
    still worth a human's attention, so a proposal sitting exactly on it must
    survive into the mode logic. With `<=` the boundary moved and the proposal
    was dropped to a skip -- silently narrowing what a review queue ever sees.
    """
    ctx = _ctx(mode="auto", review_threshold=0.70, auto_threshold=0.95)
    decision = decide(_proposal(confidence=0.70), CFG, ctx)
    assert (decision.action, decision.status) == ("suggest", "suggested")
    assert decision.reason == "held for review (mode=auto)"


def test_confidence_exactly_at_a_coincident_auto_threshold_is_applied():
    """Also kills cmp#7, at the sharpest point.

    When both thresholds coincide, a proposal exactly on the line is an
    automatic apply; the mutant turns that same proposal into a skip.
    """
    ctx = _ctx(mode="auto", review_threshold=0.80, auto_threshold=0.80)
    decision = decide(_proposal(confidence=0.80), CFG, ctx)
    assert (decision.action, decision.status) == ("apply", "automatic")


def test_confidence_just_below_the_review_threshold_is_skipped():
    """The floor must still bite one step below the boundary."""
    ctx = _ctx(mode="auto", review_threshold=0.70)
    decision = decide(_proposal(confidence=0.69), CFG, ctx)
    assert decision.action == "skip"
    assert decision.reason == "confidence 0.69 below review threshold 0.70"


# -- every branch hands the audit trail the right human_review flag ---------


def test_the_protective_veto_expert_routes_to_a_human():
    """Kills policy const_bool#4 (`human_review=True` -> `False`).

    An ``identifier_like`` veto is the protective expert refusing a mutation;
    the row is skipped but a human is meant to see it, so the flag that puts
    it in front of one may not be dropped.
    """
    decision = decide(
        _proposal(issue_type="identifier_like", rationale="looks like a key"),
        CFG,
        _ctx(),
    )
    assert (decision.action, decision.status) == ("skip", "skipped")
    assert decision.reason == "looks like a key"
    assert decision.human_review is True


def test_a_protected_column_skip_routes_to_a_human():
    """Kills policy const_bool#5 (`human_review=True` -> `False`).

    A proposal that hit a column protection was worth making; the skip is the
    policy's choice, not a verdict on the signal, so it stays reviewable.
    """
    ctx = _ctx(columns={"code": _info(preserve=True)})
    decision = decide(_proposal(issue_type="category_synonym"), CFG, ctx)
    assert decision.reason == "column is in preserve_columns"
    assert decision.human_review is True


def test_a_below_threshold_skip_is_not_routed_to_a_human():
    """Kills policy const_bool#6 (`human_review=False` -> `True`).

    The low-confidence skip is recorded for audit only. Flagging it would
    flood a review queue with exactly the signals the floor exists to filter.
    """
    decision = decide(_proposal(confidence=0.10), CFG, _ctx(review_threshold=0.70))
    assert decision.action == "skip"
    assert decision.human_review is False


def test_assist_mode_suggestions_are_routed_to_a_human():
    """Kills policy const_bool#7 (`human_review=True` -> `False`).

    assist mode never mutates; its entire output is material for a person.
    """
    decision = decide(_proposal(), CFG, _ctx(mode="assist"))
    assert (decision.action, decision.status) == ("suggest", "suggested")
    assert decision.reason == "assist mode records suggestions only"
    assert decision.human_review is True


def test_a_review_mode_automatic_apply_is_not_routed_to_a_human():
    """Kills policy const_bool#8 (`human_review=False` -> `True`).

    A deterministic low-risk repair applied in review mode is done; marking it
    for review would contradict the automatic status recorded beside it.
    """
    decision = decide(_proposal(confidence=0.99, risk="low"), CFG, _ctx(mode="review"))
    assert (decision.action, decision.status) == ("apply", "automatic")
    assert decision.reason == "deterministic low-risk repair"
    assert decision.human_review is False


@pytest.mark.parametrize(("confidence", "risk"), [(0.80, "low"), (0.99, "medium"), (0.99, "high")])
def test_a_review_mode_hold_is_routed_to_a_human(confidence, risk):
    """Kills policy const_bool#9 (`human_review=True` -> `False`).

    Everything review mode declines to apply -- not confident enough, or not
    low risk -- is precisely what the human is there to decide.
    """
    decision = decide(_proposal(confidence=confidence, risk=risk), CFG, _ctx(mode="review"))
    assert (decision.action, decision.status) == ("suggest", "suggested")
    assert decision.reason == "held for review (mode=review)"
    assert decision.human_review is True


def test_an_auto_mode_automatic_apply_is_not_routed_to_a_human():
    """Kills policy const_bool#10 (`human_review=False` -> `True`)."""
    decision = decide(_proposal(confidence=0.99, risk="medium"), CFG, _ctx(mode="auto"))
    assert (decision.action, decision.status) == ("apply", "automatic")
    assert decision.reason == "high-confidence low-risk repair"
    assert decision.human_review is False


def test_a_disabled_semantic_layer_skips_without_routing_to_a_human():
    """Kills policy const_bool#12 (`human_review=False` -> `True`).

    With the layer off (or an unknown mode) the gate does nothing at all; it
    must not manufacture review work out of a feature the user disabled.
    """
    for mode in ("off", "", "not-a-mode"):
        decision = decide(_proposal(), CFG, _ctx(mode=mode))
        assert (decision.action, decision.status) == ("skip", "skipped")
        assert decision.reason == "semantic layer disabled"
        assert decision.human_review is False


def test_an_unsafe_ambiguous_proposal_is_reported_as_high_risk():
    """The risk override the audit trail reads, independent of the mode path.

    NOTE -- this pins *actual* behaviour, not desired behaviour. The module
    docstring promises the gate will "never auto-apply high risk", but
    ``decide`` gates auto mode on ``proposal.risk`` while ``_decision``
    rewrites ``unsafe_ambiguous`` to ``risk="high"`` afterwards. A proposal
    constructed with ``issue_type="unsafe_ambiguous"`` and ``risk="low"`` is
    therefore applied automatically *and* recorded as high risk. Every
    in-tree producer of ``unsafe_ambiguous`` routes through
    ``scoring.make_proposal``/``risk_for``, which already returns "high", so
    the gap is unreachable from the pipeline today; it is reported as a
    suspected defect rather than fixed here (test-only change).
    """
    decision = decide(
        _proposal(issue_type="unsafe_ambiguous", risk="low", confidence=0.99),
        CFG,
        _ctx(mode="auto"),
    )
    assert decision.risk == "high"
    assert decision.action == "apply"  # suspected defect: see docstring


def test_an_unsafe_ambiguous_proposal_scored_high_risk_is_held():
    """How the pipeline actually reaches the gate: risk already "high"."""
    decision = decide(
        _proposal(issue_type="unsafe_ambiguous", risk="high", confidence=0.99),
        CFG,
        _ctx(mode="auto"),
    )
    assert (decision.action, decision.risk) == ("suggest", "high")
    assert decision.human_review is True
