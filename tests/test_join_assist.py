"""Tests for the dirty-join assistant (reviewable fuzzy join suggestions)."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd


def _frames():
    left = pd.DataFrame({
        "company_name": ["Acme Inc", "Globex LLC", "Initech"],
        "country": ["US", "US", "US"],
    })
    right = pd.DataFrame({
        "company_name": ["Acme, Inc.", "Globex", "Initech", "Acme Incorporated"],
        "country": ["US", "US", "US", "US"],
    })
    return left, right


def test_exact_match_scores_one() -> None:
    left, right = _frames()
    rep = fd.suggest_join_keys(left, right, on=["company_name"], exact_within=["country"])
    exact = [c for c in rep.candidates if c.left_index == 2 and c.right_index == 2]
    assert exact and exact[0].score == 1.0
    assert exact[0].status == "match"


def test_fuzzy_candidates_ranked() -> None:
    left, right = _frames()
    rep = fd.suggest_join_keys(left, right, on=["company_name"], threshold=0.85)
    scores = [c.score for c in rep.candidates]
    assert scores == sorted(scores, reverse=True)
    assert any(0 < s < 1 for s in scores)


def test_ambiguous_not_silently_accepted() -> None:
    left = pd.DataFrame({"name": ["Acme"]})
    right = pd.DataFrame({"name": ["Acme", "Acme"]})  # identical duplicates → ambiguous
    rep = fd.suggest_join_keys(left, right, on=["name"], threshold=0.85)
    matches = rep.matches
    # At most one confident match; the rest are flagged ambiguous, never silent.
    assert len(matches) <= 1
    assert len(rep.ambiguous) >= 1


def test_reproducible_output() -> None:
    left, right = _frames()
    a = fd.suggest_join_keys(left, right, on=["company_name"]).to_dict()
    b = fd.suggest_join_keys(left, right, on=["company_name"]).to_dict()
    assert a == b


def test_report_surfaces() -> None:
    left, right = _frames()
    rep = fd.suggest_join_keys(left, right, on=["company_name"], exact_within=["country"])
    assert list(rep.to_frame().columns)[:3] == ["left_index", "right_index", "score"]
    assert rep.exact_keys[0]["column"] == "company_name"
    assert "<div class=\"fd-report\"" in rep.to_html()


def test_invalid_on_columns() -> None:
    left, right = _frames()
    with pytest.raises(ValueError, match="on"):
        fd.suggest_join_keys(left, right, on=["nonexistent"])
