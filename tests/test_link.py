"""Tests for the two-frame entity-resolution wrapper (``fd.link``)."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise.entity_resolution import EntityResolutionReport


@pytest.fixture
def left() -> pd.DataFrame:
    return pd.DataFrame(
        {"name": ["Alice Smith", "Bob Jones", "Carol White"], "city": ["NYC", "LA", "SF"]}
    )


@pytest.fixture
def right() -> pd.DataFrame:
    return pd.DataFrame(
        {"name": ["alice smith", "Bob Jonas", "Dan Black"], "city": ["NYC", "LA", "SF"]}
    )


def test_exact_matches_only_identical_keys(left):
    other = pd.DataFrame({"name": ["Alice Smith", "Zoe Gray"], "city": ["NYC", "DC"]})
    rep = fd.link(left, other, keys=["name", "city"], strategy="exact")
    assert isinstance(rep, EntityResolutionReport)
    assert rep.n_matches == 1  # only the identical Alice Smith / NYC row


def test_exact_no_false_match_on_case_difference(left, right):
    rep = fd.link(left, right, keys=["name", "city"], strategy="exact")
    assert rep.n_matches == 0  # 'Alice Smith' != 'alice smith' under exact


def test_fuzzy_links_near_duplicates(left, right):
    rep = fd.link(
        left,
        right,
        keys=["name"],
        strategy="fuzzy",
        threshold=0.8,
        blocking="l.city = r.city",
    )
    assert rep.n_candidate_pairs == 3
    assert rep.n_matches >= 1
    # every pair carries an explanation
    assert all(p.explanation for p in rep.pairs)


def test_external_strategy_formats_adapter_pairs(left, right):
    def adapter(lf, rf, keys):
        out = []
        for i, ln in enumerate(lf["name"]):
            for j, rn in enumerate(rf["name"]):
                if ln.lower() == rn.lower():
                    out.append({"left_index": i, "right_index": j, "score": 1.0})
                elif ln.split()[0].lower() == rn.split()[0].lower():
                    out.append(
                        {
                            "left_index": i,
                            "right_index": j,
                            "score": 0.7,
                            "reason": "first-name match",
                        }
                    )
        return out

    rep = fd.link(left, right, keys=["name"], strategy="external", adapter=adapter)
    assert rep.n_matches == 1  # alice exact
    assert rep.n_possible_matches == 1  # Bob first-name (0.7 in [0.65, 0.85))
    assert rep.backend == "external"
    # explanations carry the adapter's reason
    poss = [p for p in rep.pairs if p.decision == "possible_match"][0]
    assert "first-name match" in poss.explanation[0].rationale


def test_external_requires_adapter(left, right):
    with pytest.raises(ValueError, match="adapter"):
        fd.link(left, right, keys=["name"], strategy="external")


def test_invalid_strategy(left, right):
    with pytest.raises(ValueError, match="strategy must be"):
        fd.link(left, right, keys=["name"], strategy="nope")


def test_missing_key_raises(left, right):
    with pytest.raises(KeyError, match="missing key"):
        fd.link(left, right, keys=["email"], strategy="exact")


def test_empty_keys_raises(left, right):
    with pytest.raises(ValueError, match="at least one key"):
        fd.link(left, right, keys=[], strategy="exact")


def test_return_linked_returns_tuple(left, right):
    result = fd.link(
        left,
        right,
        keys=["name"],
        strategy="fuzzy",
        blocking="l.city = r.city",
        return_linked=True,
    )
    assert isinstance(result, tuple)
    linked, rep = result
    assert isinstance(rep, EntityResolutionReport)


def test_report_exports(left, right):
    rep = fd.link(left, right, keys=["name"], strategy="fuzzy", blocking="l.city = r.city")
    assert "entity resolution" in rep.summary()
    assert "n_candidate_pairs" in rep.to_dict()
    assert not rep.to_frame().empty


def test_does_not_mutate_inputs(left, right):
    lbefore, rbefore = left.copy(deep=True), right.copy(deep=True)
    fd.link(left, right, keys=["name"], strategy="fuzzy", blocking="l.city = r.city")
    pd.testing.assert_frame_equal(left, lbefore)
    pd.testing.assert_frame_equal(right, rbefore)
