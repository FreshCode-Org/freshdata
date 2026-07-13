"""ER upgrades: token_set/metaphone comparators, null_policy, mode presets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise import BlockingRule, ComparisonLevel, EntityResolutionConfig
from freshdata.enterprise.entity_resolution import (
    metaphone,
    soundex,
    token_set_similarity,
)

# -- token_set --------------------------------------------------------------


def test_token_set_reorder_invariant():
    assert token_set_similarity("Ann van Dyke", "van Dyke, Ann") == 1.0


def test_token_set_partial_overlap():
    sim = token_set_similarity("ann marie smith", "ann smith")
    assert 0.0 < sim < 1.0
    assert sim == pytest.approx(2 / 3)


def test_token_set_disjoint_and_edges():
    assert token_set_similarity("alice", "bob") == 0.0
    assert token_set_similarity("", "") == 1.0
    assert token_set_similarity("alice", "") == 0.0
    assert token_set_similarity("A B", "b a") == 1.0  # case-insensitive


def test_token_set_range_and_symmetry():
    cases = [("john q smith", "smith john"), ("a b c", "c d e"), ("x", "x")]
    for s1, s2 in cases:
        sim = token_set_similarity(s1, s2)
        assert 0.0 <= sim <= 1.0
        assert sim == token_set_similarity(s2, s1)


# -- metaphone --------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Smith", "Smyth"),
        ("Philip", "Fillip"),
        ("Knight", "Night"),
        ("White", "Wight"),
        ("Shawn", "Shaun"),
    ],
)
def test_metaphone_homophones_agree(a, b):
    assert metaphone(a) == metaphone(b)


@pytest.mark.parametrize(("a", "b"), [("Smith", "Jones"), ("Knight", "Kingsley")])
def test_metaphone_distinct_names_disagree(a, b):
    assert metaphone(a) != metaphone(b)


def test_metaphone_catches_what_soundex_misses():
    # KN-/N- silent-letter pairs: soundex keeps the leading K (K523 vs N230),
    # metaphone models the silent K.
    assert soundex("Knight") != soundex("Night")
    assert metaphone("Knight") == metaphone("Night")


def test_metaphone_edge_inputs():
    assert metaphone("") == ""
    assert metaphone("123") == ""
    assert metaphone("X-ray") == metaphone("xray")
    assert metaphone("a") == "A"


def test_metaphone_deterministic():
    assert metaphone("Katherine") == metaphone("Katherine")


# -- null_policy ------------------------------------------------------------


@pytest.fixture
def sparse_pair_df() -> pd.DataFrame:
    """Two records of the same person; the second is missing email+phone."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["alice garcia", "alice garcia", "zed zephyr"],
            "email": ["alice@x.test", None, "zed@y.test"],
            "phone": ["5551234567", None, "5559990000"],
            "dob": ["1980-02-01", "1980-02-01", "1999-09-09"],
        }
    )


def _config(**overrides) -> EntityResolutionConfig:
    base: dict = {
        "enabled": True,
        "backend": "pandas",
        "unique_id_column": "id",
        "blocking_rules": (BlockingRule("l.dob = r.dob", "same dob"),),
        "comparisons": (
            ComparisonLevel("name", "jaro_winkler", threshold=0.85, weight=2.0),
            ComparisonLevel("email", "jaro_winkler", threshold=0.9, weight=2.0),
            ComparisonLevel("phone", "levenshtein", threshold=0.85, weight=2.0),
            ComparisonLevel("dob", "exact", weight=1.0),
        ),
    }
    base.update(overrides)
    return EntityResolutionConfig(**base)


def test_neutral_scores_sparse_pair_higher(sparse_pair_df):
    _, rep_pen = fd.resolve_entities(sparse_pair_df, config=_config(null_policy="penalize"))
    _, rep_neu = fd.resolve_entities(sparse_pair_df, config=_config(null_policy="neutral"))

    def score(rep):
        [pair] = [p for p in rep.pairs if {p.left_id, p.right_id} == {1, 2}]
        return pair.match_probability

    # penalize: missing email+phone drag the weighted mean down; neutral drops
    # those fields from the denominator, so the surviving evidence dominates.
    assert score(rep_neu) > score(rep_pen)
    assert score(rep_neu) >= 0.85  # name+dob agree perfectly


def test_neutral_explanations_mark_ignored_fields(sparse_pair_df):
    _, rep = fd.resolve_entities(sparse_pair_df, config=_config(null_policy="neutral"))
    [pair] = [p for p in rep.pairs if {p.left_id, p.right_id} == {1, 2}]
    ignored = [e for e in pair.explanation if "ignored" in e.rationale]
    assert {e.field for e in ignored} == {"email", "phone"}
    assert all(e.contribution == 0.0 for e in ignored)


def test_penalize_remains_default_behavior(sparse_pair_df):
    assert EntityResolutionConfig(enabled=True).null_policy == "penalize"
    _, rep = fd.resolve_entities(sparse_pair_df, config=_config())
    [pair] = [p for p in rep.pairs if {p.left_id, p.right_id} == {1, 2}]
    missing_fields = [e for e in pair.explanation if "no support" in e.rationale]
    assert missing_fields  # missing sides still recorded as opposing evidence


def test_all_fields_missing_under_neutral_is_non_match():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": [None, None],
            "email": [None, None],
            "phone": [None, None],
            "dob": ["2000-01-01", "2000-01-01"],
        }
    )
    cfg = _config(
        null_policy="neutral",
        comparisons=(
            ComparisonLevel("name", "jaro_winkler", weight=1.0),
            ComparisonLevel("email", "jaro_winkler", weight=1.0),
        ),
    )
    _, rep = fd.resolve_entities(df, config=cfg)
    assert all(p.decision == "non_match" for p in rep.pairs)


# -- mode presets -----------------------------------------------------------


def test_mode_presets_set_thresholds():
    assert EntityResolutionConfig(mode="precision").match_threshold == 0.92
    assert EntityResolutionConfig(mode="recall").match_threshold == 0.75
    balanced = EntityResolutionConfig()
    assert (balanced.match_threshold, balanced.clerical_review_threshold) == (0.85, 0.65)


def test_mode_conflicts_with_explicit_thresholds():
    with pytest.raises(ValueError, match="mode"):
        EntityResolutionConfig(mode="precision", match_threshold=0.9)
    with pytest.raises(ValueError, match="mode"):
        EntityResolutionConfig(mode="recall", clerical_review_threshold=0.5)


def test_mode_and_null_policy_validation():
    with pytest.raises(ValueError, match="mode"):
        EntityResolutionConfig(mode="turbo")
    with pytest.raises(ValueError, match="null_policy"):
        EntityResolutionConfig(null_policy="ignore")


def test_recall_mode_finds_more_matches_than_precision():
    rng = np.random.default_rng(3)
    names = ["maria silva", "john smith", "wei chen", "amit patel"]
    rows = []
    for i in range(80):
        base = names[i % len(names)]
        noisy = base if rng.random() < 0.5 else base.replace("i", "y", 1)
        rows.append(
            {
                "id": i,
                "name": noisy,
                "email": f"{base.split()[0]}{i % 8}@t.test",
                "phone": None,
                "dob": f"199{i % 4}-01-01",
            }
        )
    df = pd.DataFrame(rows)
    _, rep_p = fd.resolve_entities(
        df, config=_config(mode="precision", null_policy="neutral")
    )
    _, rep_r = fd.resolve_entities(df, config=_config(mode="recall", null_policy="neutral"))
    assert rep_r.n_matches >= rep_p.n_matches


# -- new comparator kinds end-to-end -----------------------------------------


def test_token_set_and_metaphone_kinds_in_pipeline():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["ann van dyke", "van dyke, ann"],
            "last": ["knight", "night"],
            "dob": ["1970-01-01", "1970-01-01"],
        }
    )
    cfg = EntityResolutionConfig(
        enabled=True,
        backend="pandas",
        unique_id_column="id",
        blocking_rules=(BlockingRule("l.dob = r.dob", "same dob"),),
        comparisons=(
            ComparisonLevel("name", "token_set", weight=2.0),
            ComparisonLevel("last", "metaphone", weight=1.0),
        ),
    )
    _, rep = fd.resolve_entities(df, config=cfg)
    [pair] = rep.pairs
    assert pair.decision == "match"
    assert pair.comparison_vector["name"] == 1.0
    assert pair.comparison_vector["last"] == 1.0
