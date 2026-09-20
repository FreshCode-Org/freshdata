"""Memory is evidence, not authority (#FD2-006).

A repair learned on one dataset was replayed onto another whose column meant
something different, and applied automatically:

    A: segment = M/F        -> learns  M -> male
    B: segment = S/M/L      -> becomes S/male/L/male

The conflict machinery did exist and did work, but only fired on
*disagreement*: ``_conflict_proposal`` turns a deterministic-vs-memory clash
into a high-risk ``unsafe_ambiguous`` record. On B the ``CategorySynonymExpert``
correctly **abstains** -- the column is not gender-like -- and an abstention is
not a disagreement, so the replayed repair passed through carrying the full
confidence it had earned on a different column.

The stored signatures would not have caught it: both columns profile as
``role="categorical", semantic_type=None, free_text=False``.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.semantic.apply import _CONTEXT_DEPENDENT_ISSUES

CLEAN = {"verbose": False, "return_report": True}


def _frame(values) -> pd.DataFrame:
    return pd.DataFrame(
        {"id": [f"r{i}" for i in range(len(values))], "segment": list(values)}
    )


@pytest.fixture
def gender_memory():
    """Memory learned where ``segment`` genuinely holds genders."""
    learn_df = _frame(["M", "F"] * 6)
    _, report = fd.clean(learn_df, semantic_mode="auto", **CLEAN)
    return fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="hr")


def _semantic(report):
    return [a for a in report.actions if a.step == "semantic"]


def test_a_gender_memory_does_not_rewrite_clothing_sizes(gender_memory):
    """The headline trap: S/M/L must survive a memory that learned M -> male."""
    sizes = _frame(["S", "M", "L", "M"] * 3)
    out, report = fd.clean(sizes, semantic_mode="auto", memory=gender_memory, **CLEAN)

    assert out["segment"].tolist() == sizes["segment"].tolist()
    assert not [a for a in _semantic(report) if a.status == "automatic"]


def test_the_held_repair_is_routed_to_a_human_with_its_reason(gender_memory):
    """Holding it silently would only trade one failure for another."""
    sizes = _frame(["S", "M", "L", "M"] * 3)
    _, report = fd.clean(sizes, semantic_mode="auto", memory=gender_memory, **CLEAN)

    held = [a for a in _semantic(report) if a.metadata.get("backend") == "memory"]
    assert len(held) == 1
    action = held[0]
    assert action.status == "suggested"
    assert action.risk == "high"
    assert action.human_review is True
    assert "corroborate" in action.rationale
    assert action.metadata["raw_value"] == "M"
    assert action.metadata["proposed_value"] == "male"


def test_replay_onto_the_same_kind_of_column_still_works(gender_memory):
    """The guard must not disable the feature it is guarding.

    A second genuinely gender-shaped frame is exactly what memory is for, and
    there the deterministic expert corroborates the repair.
    """
    genders = _frame(["M", "F", "M", "F"] * 3)
    out, report = fd.clean(genders, semantic_mode="auto", memory=gender_memory, **CLEAN)

    assert out["segment"].tolist() == ["male", "female"] * 6
    assert [a for a in _semantic(report) if a.status == "automatic"]


def test_provenance_survives_the_demotion(gender_memory):
    """A held replay is precisely the case a reviewer must be able to trace."""
    sizes = _frame(["S", "M", "L", "M"] * 3)
    _, report = fd.clean(sizes, semantic_mode="auto", memory=gender_memory, **CLEAN)

    action = next(a for a in _semantic(report) if a.metadata.get("backend") == "memory")
    assert action.memory_influenced is True
    assert action.model_id.endswith(":memory")
    kinds = [e["kind"] for e in action.metadata["evidence"]]
    assert "memory_replay" in kinds
    assert "memory_uncorroborated" in kinds, "the reason for holding must be recorded"


def test_a_context_free_repair_is_not_demoted(gender_memory):
    """Only context-dependent issue types need corroboration.

    An encoding or format repair means the same thing in every column, so
    demoting it would make memory useless without making anything safer.
    """
    assert "category_synonym" in _CONTEXT_DEPENDENT_ISSUES
    assert "boolean_synonym" in _CONTEXT_DEPENDENT_ISSUES
    for context_free in ("encoding_repair", "format_alignment", "numeric_format"):
        assert context_free not in _CONTEXT_DEPENDENT_ISSUES
