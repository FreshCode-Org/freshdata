"""The shared trap corpus must stay honest and stay wired to the library.

These tests do not exercise ``fd.clean``; they guard the corpus itself, which
every later semantic test depends on. A corpus that silently drifts away from
the library's vocabulary would make every downstream measurement meaningless.
"""

from __future__ import annotations

import pytest
from benchmarks.corpus import (
    REVIEW_FAMILY,
    TRAPS,
    UNSET,
    Disposition,
    all_cases,
    by_token,
    from_field_action,
    from_gauntlet,
    from_truthbench,
    satisfies,
)
from benchmarks.truthbench.models import Disposition as TBDisposition

from freshdata.fieldcheck import ACTIONS

# -- the corpus is internally consistent -----------------------------------


def test_corpus_is_non_trivial():
    """A corpus that shrinks to nothing would make every later test vacuous."""
    assert len(TRAPS) >= 80
    assert len({c.family for c in TRAPS}) >= 40
    assert len({c.role for c in TRAPS}) >= 20


def test_every_repair_case_carries_a_gold_value():
    for case in TRAPS:
        if case.expected is Disposition.REPAIR:
            assert case.repaired is not UNSET, f"{case.family}/{case.role}"


def test_every_undefined_expectation_names_the_missing_decision():
    """An unknown expectation must be a recorded specification gap, not a shrug."""
    gaps = [c for c in TRAPS if c.expected is None]
    assert gaps, "the corpus should retain the genuinely undecided cases"
    for case in gaps:
        assert case.spec_gap, f"{case.family}/{case.role} has no spec_gap"
        assert len(case.spec_gap) > 40, (
            f"{case.family}/{case.role}: spec_gap must say which decision is "
            "missing, not merely that one is"
        )


def test_non_repair_cases_do_not_carry_a_repair_value():
    for case in TRAPS:
        if case.expected is not Disposition.REPAIR:
            assert case.repaired is UNSET, f"{case.family}/{case.role}"


# -- the corpus's reason to exist ------------------------------------------


def test_the_same_token_resolves_differently_in_different_roles():
    """This is the whole point: context, not the token, decides the outcome.

    A cleaner that keys decisions on the value alone passes every single-role
    test and fails here.
    """
    multi_role = {}
    for case in TRAPS:
        multi_role.setdefault(case.token, set()).add(
            case.expected.value if case.expected else "spec-gap"
        )
    disagreeing = {t: v for t, v in multi_role.items() if len(v) > 1}
    assert len(disagreeing) >= 8, (
        "the corpus must contain many tokens whose correct disposition depends "
        f"on the field; found only {len(disagreeing)}"
    )
    # The canonical examples, spelled out so a regression is legible.
    assert {c.expected for c in by_token("007")} == {
        Disposition.PRESERVE,
        Disposition.REPAIR,
    }
    assert {c.repaired for c in by_token("M") if c.expected is Disposition.REPAIR} == {
        "Male",
        "Medium",
        "Married",
    }


def test_spelled_number_with_a_noun_is_never_a_bare_repair():
    """'twenty apples' must not become 20 -- the headline false positive."""
    (case,) = [c for c in TRAPS if c.token == "twenty apples"]
    assert case.expected is not Disposition.REPAIR
    assert case.expected in REVIEW_FAMILY


# -- the vocabulary stays bound to the library ------------------------------


def test_every_fieldcheck_action_has_a_disposition():
    """If ``fieldcheck.ACTIONS`` grows, this fails instead of mis-scoring.

    Silently scoring an unmapped action as something else is exactly the class
    of defect the corpus exists to prevent.
    """
    for action in ACTIONS:
        assert isinstance(from_field_action(action), Disposition), action


def test_unknown_field_action_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="no disposition mapping"):
        from_field_action("teleport")


@pytest.mark.parametrize(
    ("expected", "observed", "ok"),
    [
        (Disposition.REVIEW, Disposition.QUARANTINE, True),
        (Disposition.REVIEW, Disposition.REJECT, True),
        (Disposition.REVIEW, Disposition.REVIEW, True),
        # A specific demand is not satisfied by a weaker outcome.
        (Disposition.REJECT, Disposition.REVIEW, False),
        (Disposition.QUARANTINE, Disposition.REVIEW, False),
        # Preservation never widens: a repair is a corruption here.
        (Disposition.PRESERVE, Disposition.REPAIR, False),
        (Disposition.PRESERVE, Disposition.FLAG, False),
        (Disposition.REPAIR, Disposition.PRESERVE, False),
    ],
)
def test_satisfies_widens_only_the_review_family(expected, observed, ok):
    assert satisfies(expected, observed) is ok


def test_truthbench_dispositions_are_a_subset_of_ours():
    """Back-compat: the four-value harnesses keep their exact meaning."""
    ours = {d.value for d in Disposition}
    theirs = {d.value for d in TBDisposition}
    assert theirs <= ours
    assert ours - theirs == {"quarantine", "reject"}


# -- the adapters actually reach the existing corpora -----------------------


def test_adapters_bind_to_the_real_harnesses():
    """Guards against the adapters silently degrading to an empty tuple."""
    gauntlet = from_gauntlet()
    truthbench = from_truthbench()
    assert len(gauntlet) >= 50, "gauntlet corpus did not load"
    assert len(truthbench) >= 500, "truthbench corpus did not load"
    assert all_cases() == TRAPS + gauntlet + truthbench
    for case in gauntlet + truthbench:
        assert isinstance(case.expected, Disposition)
