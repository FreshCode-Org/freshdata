"""Regression tests for accuracy-sensitive semantic repairs.

- #252: a fuzzy memory match must never auto-apply to a different value.
- #253: ``'45%'`` in a rate column of fractions is not rewritten as ``45.0``.
- #254: shape alignment never re-splits digits across groups (``'1.10'``).
- #300: memory replay gates each learned repair by its own expert.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.semantic.canonical import NumericFormatExpert, ShapeAlignmentExpert
from freshdata.semantic.context import build_semantic_context
from freshdata.semantic.experts import VALUE_EXPERTS
from freshdata.semantic.memory import _replay_expert, semantic_memory_proposals


def _semantic(report: fd.CleanReport) -> list[fd.Action]:
    return [a for a in report.actions if a.step == "semantic"]


def _clean(df: pd.DataFrame, **kwargs):
    return fd.clean(df, semantic_mode="auto", return_report=True, verbose=False, **kwargs)


def _proposals(expert, df: pd.DataFrame):
    column = df.columns[0]
    ctx = build_semantic_context(df, CleanConfig(semantic_mode="auto"))
    info = ctx.columns[column]
    assert expert.applies(info)
    return expert.propose(df[column], info)


# --------------------------------------------------------------------------- #
# #252: fuzzy memory replay is review-only
# --------------------------------------------------------------------------- #

_LEARN_EMAILS = ["alice.johnson@EXAMPLE.COM", "bob@x.org", "carl@y.net", "dan@z.io"]
#: One character away from the learned raw value, and already a valid email.
_DRIFTED_EMAILS = ["alice.johnsen@example.com", "bob@x.org", "carl@y.net", "dan@z.io"]


@pytest.fixture
def email_memory():
    df = pd.DataFrame({"email": _LEARN_EMAILS})
    _, report = _clean(df)
    return fd.learn_cleaning_memory(df, decisions=report, dataset_id="crm")


def test_fuzzy_memory_match_does_not_rewrite_a_different_valid_value(email_memory) -> None:
    df = pd.DataFrame({"email": _DRIFTED_EMAILS})
    out, report = _clean(df, memory=email_memory)

    assert out["email"][0] == "alice.johnsen@example.com"
    replayed = [a for a in _semantic(report) if a.memory_influenced]
    assert replayed, "the fuzzy match should still be surfaced as a suggestion"
    assert all(a.status == "suggested" and a.human_review for a in replayed)
    assert all(a.confidence < 0.95 for a in replayed)


def test_fuzzy_memory_cap_follows_the_configured_auto_threshold(email_memory) -> None:
    df = pd.DataFrame({"email": _DRIFTED_EMAILS})
    ctx = build_semantic_context(
        df, CleanConfig(semantic_mode="auto", semantic_auto_threshold=0.9)
    )
    proposals = list(semantic_memory_proposals(df, ctx, email_memory))
    assert proposals
    assert all(p.confidence < 0.9 for p in proposals)


def test_exact_memory_match_still_auto_applies(email_memory) -> None:
    out, report = _clean(pd.DataFrame({"email": _LEARN_EMAILS}), memory=email_memory)

    assert out["email"][0] == "alice.johnson@example.com"
    replayed = [a for a in _semantic(report) if a.memory_influenced]
    assert replayed and all(a.status == "automatic" for a in replayed)


# --------------------------------------------------------------------------- #
# #253: percent stragglers respect the column's scale
# --------------------------------------------------------------------------- #


def test_percent_in_fraction_rate_column_is_not_auto_applied_as_45() -> None:
    df = pd.DataFrame({"conversion_rate": ["0.12", "0.30", "0.25", "45%", "0.5", "0.41"]})
    out, report = _clean(df)

    assert 45.0 not in pd.to_numeric(out["conversion_rate"], errors="coerce").tolist()
    actions = [a for a in _semantic(report) if a.metadata.get("raw_value") == "45%"]
    assert len(actions) == 1
    action = actions[0]
    assert action.status == "suggested" and action.human_review
    assert action.metadata["proposed_value"] == pytest.approx(0.45)


def test_percent_in_fraction_column_proposes_value_over_100() -> None:
    df = pd.DataFrame({"win_ratio": ["0.1", "0.2", "0.9", "30%"]})
    (proposal,) = _proposals(NumericFormatExpert(), df)
    assert proposal.proposed_value == pytest.approx(0.30)
    assert proposal.confidence < 0.95


def test_rate_column_without_scale_evidence_is_held_for_review() -> None:
    # Only two plain numbers: too few to tell fractions from percents.
    df = pd.DataFrame({"growth_rate": ["12", "30", "45%"]})
    (proposal,) = _proposals(NumericFormatExpert(), df)
    assert proposal.proposed_value == 45.0
    assert proposal.confidence < 0.95


def test_percent_scale_rate_column_still_auto_applies() -> None:
    df = pd.DataFrame({"tax_rate": ["12", "30", "25", "45%", "50", "41"]})
    out, report = _clean(df)

    assert out["tax_rate"].tolist() == [12.0, 30.0, 25.0, 45.0, 50.0, 41.0]
    actions = [a for a in _semantic(report) if a.metadata.get("raw_value") == "45%"]
    assert actions and all(a.status == "automatic" for a in actions)


def test_strong_percent_name_keeps_the_percent_number() -> None:
    # TruthBench edu-07 shape: score_percent on a 0-100 scale.
    df = pd.DataFrame({"score_percent": ["95", "90", "87.5", "82", "95%"]})
    out, report = _clean(df)

    assert out["score_percent"].tolist()[-1] == 95.0
    actions = [a for a in _semantic(report) if a.metadata.get("raw_value") == "95%"]
    assert actions and all(a.status == "automatic" for a in actions)


# --------------------------------------------------------------------------- #
# #254: shape alignment never regroups digits
# --------------------------------------------------------------------------- #


def test_version_with_different_grouping_is_left_unchanged() -> None:
    values = ["1.2.3", "2.0.1", "3.4.5", "1.0.0", "2.2.2", "4.1.0", "1.10"]
    out, report = _clean(pd.DataFrame({"version": values}))

    assert out["version"].tolist() == values
    assert not [a for a in _semantic(report) if a.metadata.get("raw_value") == "1.10"]


@pytest.mark.parametrize("raw", ["1.10", "12.3", "1.1.0.0"])
def test_regrouping_candidates_are_not_proposed(raw: str) -> None:
    df = pd.DataFrame({"version": ["1.2.3", "2.0.1", "3.4.5", "1.0.0", "2.2.2", "4.1.0", raw]})
    proposals = _proposals(ShapeAlignmentExpert(), df)
    assert [p for p in proposals if p.raw_value == raw] == []


def test_separator_drift_with_matching_groups_still_auto_applies() -> None:
    # TruthBench crm-04 shape: "555 0101" among "555-0101".
    values = ["555-0101", "555-0102", "555-0103", "555-0104", "555-0105", "555 0106"]
    out, report = _clean(pd.DataFrame({"code": values}))

    assert out["code"].tolist()[-1] == "555-0106"
    actions = [a for a in _semantic(report) if a.metadata.get("raw_value") == "555 0106"]
    assert actions and all(a.status == "automatic" for a in actions)


def test_unseparated_value_split_into_template_groups_is_review_only() -> None:
    df = pd.DataFrame({"code": ["555-0101", "555-0102", "555-0103", "555-0104", "5550105"]})
    out, report = _clean(df)

    assert out["code"].tolist()[-1] == "5550105"
    (proposal,) = [
        p for p in _proposals(ShapeAlignmentExpert(), df) if p.raw_value == "5550105"
    ]
    assert proposal.proposed_value == "555-0105"
    assert proposal.confidence < 0.95


# --------------------------------------------------------------------------- #
# #300: replay is gated by the repair's own expert
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("expert", VALUE_EXPERTS, ids=lambda e: e.name)
def test_replay_expert_resolves_by_stored_expert_name(expert) -> None:
    assert _replay_expert({"expert": expert.name, "issue_type": expert.issue_type}) is expert


def test_replay_expert_falls_back_to_issue_type_only_when_unambiguous() -> None:
    by_issue: dict[str, list] = {}
    for expert in VALUE_EXPERTS:
        by_issue.setdefault(expert.issue_type, []).append(expert)
    for issue_type, experts in by_issue.items():
        resolved = _replay_expert({"expert": None, "issue_type": issue_type})
        if len(experts) == 1:
            assert resolved is experts[0]
        else:
            assert resolved is None
    assert _replay_expert({"expert": "no_such_expert", "issue_type": "format_alignment"}) is None


_REPLAY_FRAMES = {
    "unicode_nfc": {"city": ["Jose\u0301", "Lima", "Quito", "Bogota"]},
    "mojibake": {"venue": ["CafÃ©", "Bar", "Pub", "Inn"]},
    "shape_alignment": {
        "code": ["555-0101", "555-0102", "555-0103", "555-0104", "555-0105", "555 0106"]
    },
    "numeric_format": {"score_percent": ["95", "90", "87.5", "82", "95%"]},
    "time_canonical": {"start_time": ["09:00", "10:30", "11:15", "12:00", "24:00"]},
}


@pytest.mark.parametrize("expert_name", sorted(_REPLAY_FRAMES))
def test_learned_repair_replays_on_the_identical_frame(expert_name: str) -> None:
    data = _REPLAY_FRAMES[expert_name]
    df = pd.DataFrame(data)
    _, report = _clean(df)
    memory = fd.learn_cleaning_memory(df, decisions=report, dataset_id="d")
    learned = memory.value_patterns["semantic_repairs"]
    assert [r["expert"] for r in learned] == [expert_name]

    ctx = build_semantic_context(df, CleanConfig(semantic_mode="auto"))
    assert len(semantic_memory_proposals(df, ctx, memory)) == 1

    _, replay_report = _clean(pd.DataFrame(data), memory=memory)
    actions = _semantic(replay_report)
    assert actions
    assert all(a.memory_influenced for a in actions)
    assert all(a.metadata.get("expert") == expert_name for a in actions)
