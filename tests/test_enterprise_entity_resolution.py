"""Tests for probabilistic entity resolution (Feature 3)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise.config import BlockingRule, ComparisonLevel, EntityResolutionConfig
from freshdata.enterprise.entity_resolution import (
    EntityResolutionError,
    jaro_winkler,
    levenshtein,
    levenshtein_similarity,
    link_entities,
    resolve_entities,
    soundex,
)

HAS_DUCKDB = pytest.importorskip  # alias for readability


def _people() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": [
                "Jonathan Smith",
                "Jon Smith",
                "Johnny Smith",
                "Alice Brown",
                "Alicia Brown",
                "Robert King",
            ],
            "dob": [
                "1990-01-01",
                "1990-01-01",
                "1990-01-01",
                "1985-05-05",
                "1985-05-05",
                "1970-12-12",
            ],
            "email": [
                "jsmith@x.com",
                "jsmith@x.com",
                "jon@y.com",
                "alice@z.com",
                "alicia@z.com",
                "rking@q.com",
            ],
        }
    )


def _config(backend: str = "pandas", **overrides) -> EntityResolutionConfig:
    base: dict = {
        "unique_id_column": "id",
        "backend": backend,
        "blocking_rules": (BlockingRule(sql="l.dob = r.dob"),),
        "comparisons": (
            ComparisonLevel(column="name", kind="jaro_winkler", threshold=0.85, weight=3.0),
            ComparisonLevel(column="dob", kind="exact", weight=1.0),
            ComparisonLevel(column="email", kind="exact", weight=1.0),
        ),
        "match_threshold": 0.80,
        "clerical_review_threshold": 0.55,
    }
    base.update(overrides)
    return EntityResolutionConfig(**base)


# --- string primitives ----------------------------------------------------


def test_jaro_winkler_basic():
    assert jaro_winkler("martha", "martha") == 1.0
    assert jaro_winkler("", "x") == 0.0
    assert jaro_winkler("martha", "marhta") > 0.9
    assert jaro_winkler("abc", "xyz") < 0.5


def test_levenshtein_basic():
    assert levenshtein("kitten", "kitten") == 0
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein_similarity("abc", "abc") == 1.0


def test_soundex_basic():
    assert soundex("Robert") == soundex("Rupert")
    assert soundex("") == "0000"
    assert soundex("Tymczak")[0] == "T"


# --- resolution -----------------------------------------------------------


@pytest.mark.parametrize("backend", ["pandas", "duckdb"])
def test_duplicate_persons_cluster_correctly(backend):
    pytest.importorskip("duckdb") if backend == "duckdb" else None
    df = _people()
    out, report = resolve_entities(df, config=_config(backend))
    assert report.backend == backend
    clusters = {frozenset(c.record_ids) for c in report.clusters}
    assert frozenset({1, 2}) in clusters  # same dob + same email + near-identical name
    assert "cluster_id" in out.columns


def test_blocking_prevents_cartesian_explosion():
    # 6 rows -> 15 unblocked pairs; dob blocking yields far fewer.
    df = _people()
    out, report = resolve_entities(df, config=_config("pandas"))
    assert report.n_candidate_pairs < 15
    assert report.n_candidate_pairs == 4  # (1,2),(1,3),(2,3) on dob + (4,5)


def test_max_pairs_gate_raises():
    df = _people()
    with pytest.raises(EntityResolutionError, match="max_pairs"):
        resolve_entities(df, config=_config("pandas", max_pairs=1))


def test_no_blocking_rules_is_rejected():
    df = _people()
    with pytest.raises(EntityResolutionError, match="blocking_rules"):
        resolve_entities(df, config=_config("pandas", blocking_rules=()))


def test_exact_email_match_scores_high():
    df = _people()
    _out, report = resolve_entities(df, config=_config("pandas"))
    pair = next(p for p in report.pairs if {p.left_id, p.right_id} == {1, 2})
    assert pair.match_probability >= 0.9
    assert pair.decision == "match"


def test_fuzzy_name_same_dob_scores_high():
    # name-heavy config: fuzzy name + same dob should clear the match bar.
    cfg = _config(
        "pandas",
        comparisons=(
            ComparisonLevel(column="name", kind="jaro_winkler", weight=3.0),
            ComparisonLevel(column="dob", kind="exact", weight=2.0),
        ),
        match_threshold=0.85,
    )
    df = _people()
    _out, report = resolve_entities(df, config=cfg)
    pair = next(p for p in report.pairs if {p.left_id, p.right_id} == {1, 2})
    assert pair.match_probability >= 0.85


def test_different_dob_scores_low():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["John Smith", "John Smith"],
            "dob": ["1990-01-01", "1965-07-07"],
            "email": ["a@x.com", "b@y.com"],
        }
    )
    cfg = _config(
        "pandas",
        blocking_rules=(BlockingRule(sql="lower(left(l.name,4)) = lower(left(r.name,4))"),),
    )
    _out, report = resolve_entities(df, config=cfg)
    pair = report.pairs[0]
    assert pair.comparison_vector["dob"] == 0.0
    assert pair.decision != "match"


def test_connected_components_stable_cluster_ids():
    df = _people()
    out1 = resolve_entities(df, config=_config("pandas"), return_report=False)
    out2 = resolve_entities(df.sample(frac=1, random_state=1), config=_config("pandas"),
                            return_report=False)
    # Map id -> cluster_id; the {1,2} pair should share a cluster id in both runs.
    m1 = dict(zip(out1["id"], out1["cluster_id"]))
    m2 = dict(zip(out2["id"], out2["cluster_id"]))
    assert m1[1] == m1[2]
    assert m2[1] == m2[2]
    assert (m1[1] == m1[3]) == (m2[1] == m2[3])  # structure preserved


def test_canonical_chosen_by_completeness():
    df = pd.DataFrame(
        {
            "id": [10, 11],
            "name": ["Ann Lee", "Ann Lee"],
            "dob": ["2000-02-02", "2000-02-02"],
            "email": [None, "ann@x.com"],  # row 11 is more complete
        }
    )
    cfg = _config("pandas", comparisons=(
        ComparisonLevel(column="name", kind="exact", weight=1.0),
        ComparisonLevel(column="dob", kind="exact", weight=1.0),
    ), match_threshold=0.9)
    _out, report = resolve_entities(df, config=cfg)
    cluster = next(c for c in report.clusters if set(c.record_ids) == {10, 11})
    assert cluster.canonical_record_id == 11


def test_pandas_fallback_blocking_with_substr_and_function():
    df = _people()
    cfg = _config(
        "pandas",
        blocking_rules=(BlockingRule(sql="substr(l.name,1,3) = substr(r.name,1,3)"),),
    )
    _out, report = resolve_entities(df, config=cfg)
    assert report.n_candidate_pairs >= 1


def test_pandas_blocking_rejects_unsupported_sql():
    df = _people()
    cfg = _config("pandas", blocking_rules=(BlockingRule(sql="l.dob < r.dob"),))
    with pytest.raises(EntityResolutionError, match="equality"):
        resolve_entities(df, config=cfg)


def test_duckdb_backend_runs():
    pytest.importorskip("duckdb")
    df = _people()
    cfg = _config(
        "duckdb",
        blocking_rules=(BlockingRule(sql="lower(l.email) = lower(r.email)"),),
    )
    _out, report = resolve_entities(df, config=cfg)
    assert report.backend == "duckdb"
    assert any(set(c.record_ids) == {1, 2} for c in report.clusters)


def test_non_unique_id_rejected():
    df = pd.DataFrame({"id": [1, 1], "name": ["a", "b"], "dob": ["x", "x"]})
    cfg = _config("pandas", comparisons=(ComparisonLevel(column="name", kind="exact"),))
    with pytest.raises(EntityResolutionError, match="unique"):
        resolve_entities(df, config=cfg)


def test_report_is_json_serializable():
    df = _people()
    _out, report = resolve_entities(df, config=_config("pandas"))
    text = json.dumps(report.to_dict(), default=str)
    assert "n_clusters" in text


def test_link_entities_cross_source():
    left = pd.DataFrame(
        {"id": [1, 2], "name": ["Jon Smith", "Ann Lee"], "dob": ["1990-01-01", "2000-02-02"]}
    )
    right = pd.DataFrame(
        {"id": [3, 4], "name": ["Jonathan Smith", "Zed"], "dob": ["1990-01-01", "1911-11-11"]}
    )
    cfg = _config(
        "pandas",
        link_type="link_only",
        blocking_rules=(BlockingRule(sql="l.dob = r.dob"),),
        comparisons=(
            ComparisonLevel(column="name", kind="jaro_winkler", weight=2.0),
            ComparisonLevel(column="dob", kind="exact", weight=1.0),
        ),
        match_threshold=0.75,
    )
    _out, report = link_entities(left, right, config=cfg)
    # ids 1 and 3 share dob and similar names -> linked across sources
    assert any({1, 3} <= set(c.record_ids) for c in report.clusters)


def test_public_api_exposed():
    assert fd.resolve_entities is resolve_entities
    assert fd.link_entities is link_entities


def test_polars_input_supported():
    pl = pytest.importorskip("polars")
    df = pl.from_pandas(_people())
    out = resolve_entities(df, config=_config("pandas"), return_report=False)
    assert isinstance(out, pl.DataFrame)
    assert "cluster_id" in out.columns


def test_scales_blocking_only_touches_candidates():
    # Many records, tight blocking -> candidate pairs stay small.
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": range(500),
            "name": ["Name" + str(i % 250) for i in range(500)],
            "dob": [f"19{rng.integers(50, 99)}-01-01" for _ in range(500)],
            "email": [f"u{i % 250}@x.com" for i in range(500)],
        }
    )
    cfg = _config(
        "pandas",
        blocking_rules=(BlockingRule(sql="lower(l.email) = lower(r.email)"),),
        comparisons=(ComparisonLevel(column="name", kind="exact", weight=1.0),),
        match_threshold=0.9,
        max_pairs=10_000,
    )
    _out, report = resolve_entities(df, config=cfg)
    assert report.n_candidate_pairs == 250  # each email shared by exactly 2 rows
