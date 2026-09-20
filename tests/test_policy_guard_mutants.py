"""Properties the semantic decision core had no test for.

Found by mutation testing the two modules where a surviving mutant is most
alarming: the policy gate that decides whether a repair is applied, and the
guard that enforces protected-column byte-identity. Nineteen targeted mutants
(comparison flips, boolean-operator swaps, constant flips) produced a **63.2%
kill rate**, and seven survived the entire 7120-test suite -- four of them in
the identifier carve-out and column protection.

Each test below kills a specific confirmed survivor. The mutation it defeats is
named, so if the assertion is ever weakened the reason is on the page.
"""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.guard import _cell_equal, snapshot_protected
from freshdata.semantic.policy import _payload_preserving, _protection
from freshdata.semantic.types import SemanticColumnInfo, SemanticContext, SemanticProposal


def _proposal(column="code", raw="A-1", proposed="A1", issue_type="format_alignment"):
    return SemanticProposal(
        column=column, raw_value=raw, proposed_value=proposed,
        issue_type=issue_type, expert="t", confidence=0.99, risk="low", rationale="t",
    )


def _info(**kw) -> SemanticColumnInfo:
    base = {
        "name": "code", "role": "categorical", "n_nonnull": 3, "nunique": 3,
        "high_cardinality": False, "preserve": False, "free_text": False,
        "numeric_like": False, "boolean_like": False, "money_like": False,
        "unit_like": False, "identifier_like": False,
    }
    base.update(kw)
    return SemanticColumnInfo(**base)


def _ctx(columns=None, *, id_columns=(), preserve_columns=(), target=None):
    return SemanticContext(
        dataset=None, columns=columns or {}, auto_threshold=0.95, review_threshold=0.70,
        max_distinct_values=50, sample_size=1000, privacy_policy="none", mode="auto",
        id_columns=frozenset(id_columns), preserve_columns=frozenset(preserve_columns),
        target_column=target,
    )


# -- the identifier carve-out cannot smuggle a content change ---------------


def test_a_payload_changing_alignment_is_not_payload_preserving():
    """Kills policy bool#4 (`and` -> `or`).

    With `or`, any non-empty raw payload satisfied the check and a proposal
    that *changed* the identifying content would pass the identifier
    carve-out -- exactly what the docstring says it prevents.
    """
    assert _payload_preserving(_proposal(raw="A-1", proposed="A-2")) is False
    assert _payload_preserving(_proposal(raw="007", proposed="7")) is False


def test_a_genuinely_payload_preserving_alignment_still_passes():
    """The carve-out must keep working; separators and composition may change."""
    assert _payload_preserving(_proposal(raw="A-1", proposed="A1")) is True


@pytest.mark.parametrize(
    ("raw", "proposed"), [(7, "7"), ("7", 7), (None, "7"), (7.0, 7.0)]
)
def test_a_non_string_pair_is_never_payload_preserving(raw, proposed):
    """Kills policy bool#1 (`or` -> `and`) and const_bool#2 (`False` -> `True`).

    Both mutants let a non-string pair reach the payload comparison, where the
    early return exists precisely to stop it.
    """
    assert _payload_preserving(_proposal(raw=raw, proposed=proposed)) is False


def test_an_empty_payload_is_not_preserving():
    """`bool(raw_payload)` guards the all-punctuation case."""
    assert _payload_preserving(_proposal(raw="---", proposed="___")) is False


# -- column protection needs only one reason, not both ----------------------


def test_the_target_column_is_protected_by_name_alone():
    """Kills policy bool#8 (`or` -> `and`).

    A column named as `target_column` must be protected even when its inferred
    role is not "target"; requiring both dropped the protection.
    """
    ctx = _ctx(columns={"code": _info(role="categorical")}, target="code")
    assert _protection(_proposal(), ctx) == "target column is never modified"


def test_the_target_column_is_protected_by_inferred_role_alone():
    ctx = _ctx(columns={"code": _info(role="target")}, target=None)
    assert _protection(_proposal(), ctx) == "target column is never modified"


def test_preserve_columns_protects_by_name_alone():
    """Kills policy bool#12 (`or` -> `and`).

    A column listed in `preserve_columns` must be protected even when the
    column info carries no `preserve` flag.
    """
    ctx = _ctx(columns={"code": _info(preserve=False)}, preserve_columns=("code",))
    assert _protection(_proposal(), ctx) == "column is in preserve_columns"


def test_preserve_columns_protects_by_column_info_alone():
    ctx = _ctx(columns={"code": _info(preserve=True)}, preserve_columns=())
    assert _protection(_proposal(), ctx) == "column is in preserve_columns"


# -- the protected-column snapshot must be independent ----------------------


def test_the_protected_snapshot_does_not_share_data_with_the_frame():
    """Kills guard const_bool#4 (`deep=True` -> `deep=False`).

    The snapshot is what `verify_protected` compares against. If it shared the
    frame's buffer, an in-place mutation would change the snapshot too and the
    violation it exists to catch would go unnoticed.
    """
    df = pd.DataFrame({"keep": [1, 2, 3], "other": [4, 5, 6]})
    snapshot = snapshot_protected(df, ["keep"])
    df.loc[0, "keep"] = 999
    assert snapshot["keep"].tolist() == [1, 2, 3], "snapshot tracked an in-place edit"


# -- documented behaviour of the missing-aware comparison -------------------


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [(None, None, True), (float("nan"), float("nan"), True),
     (float("nan"), None, True), (float("nan"), 5, False), (5, 5, True), (5, 6, False)],
)
def test_cell_equality_treats_two_missing_values_as_equal(left, right, equal):
    """Two missing values compare equal; a missing and a present one do not.

    **Correction.** This docstring previously claimed that flipping
    ``if left_na or right_na`` to ``and`` was an *equivalent mutant*, "verified
    across 289 input pairs ... zero behavioural differences". **That claim was
    wrong**, and the mutant is killed by
    ``test_a_permissive_equality_object_cannot_impersonate_a_missing_value``
    in ``tests/test_guard_byte_identity_mutants.py``.

    The 289-pair pool contained no cell whose ``__eq__`` returns True against
    anything (``unittest.mock.ANY`` and wildcard/matcher objects do). For such
    a cell the fall-through ``bool(left == right)`` returns **True**, not
    False, so the mutant reports a missing value and a present one as equal --
    and the guard misses a protected-column violation. A 1296-pair
    differential found 18 such disagreements, in both argument orders.

    The lesson is about the method, not the operator: an equivalence claim is
    only as strong as the input pool it was checked over, and a pool built
    from ordinary scalars cannot rule out exotic ``__eq__``. Prefer a proof
    that the branch is unreachable over a differential that merely found no
    counterexample.
    """
    assert _cell_equal(left, right) is equal
