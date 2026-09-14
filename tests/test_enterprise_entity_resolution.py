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
    _is_missing,
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
    out2 = resolve_entities(
        df.sample(frac=1, random_state=1), config=_config("pandas"), return_report=False
    )
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
    cfg = _config(
        "pandas",
        comparisons=(
            ComparisonLevel(column="name", kind="exact", weight=1.0),
            ComparisonLevel(column="dob", kind="exact", weight=1.0),
        ),
        match_threshold=0.9,
    )
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


# --- regressions: blocking parser, missing values, link metadata -------------


def test_duckdb_backend_accepts_non_equality_blocking_sql():
    # #236: the pandas parser must not reject valid DuckDB SQL on the duckdb backend.
    pytest.importorskip("duckdb")
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["jonathan", "jonathon", "bob"]})
    cfg = EntityResolutionConfig(
        backend="duckdb",
        comparisons=(ComparisonLevel("name", "jaro_winkler"),),
        blocking_rules=(
            BlockingRule("jaro_winkler_similarity(l.name, r.name) > 0.8"),
            BlockingRule("l.id = r.id + 99"),
        ),
    )
    _out, report = resolve_entities(df, config=cfg)
    assert report.backend == "duckdb"
    assert report.n_candidate_pairs == 1
    # Rules the pandas parser cannot read are simply not attributed.
    assert report.pairs[0].blocking_rule_ids == ()


def test_duckdb_backend_still_attributes_equality_rules():
    pytest.importorskip("duckdb")
    cfg = _config(
        "duckdb",
        blocking_rules=(
            BlockingRule("l.dob < r.dob"),
            BlockingRule("lower(l.email) = lower(r.email)"),
        ),
    )
    _out, report = resolve_entities(_people(), config=cfg)
    pair = next(p for p in report.pairs if {p.left_id, p.right_id} == {1, 2})
    assert pair.blocking_rule_ids == ("block_001",)


@pytest.mark.parametrize(
    "sql",
    [
        "l.id <= r.id",
        "l.id >= r.id",
        "l.email != r.email",
        "l.email <> r.email",
        "l.email = r.email OR l.phone = r.phone",
        "l.email = r.email or l.phone = r.phone",
        "l.phone = '555'",
        "l.id = 1",
        "l.id = r.id + 1",
        "l.email between r.email and r.phone",
        "(l.email = r.email)",
        "lower() = lower()",
    ],
)
def test_pandas_blocking_rejects_non_equality_sql(sql):
    # #237: these used to be accepted and silently produce zero candidate pairs.
    df = pd.DataFrame({"id": [1, 2], "email": ["a@x.com", "b@x.com"], "phone": ["555", "555"]})
    cfg = EntityResolutionConfig(
        backend="pandas",
        blocking_rules=(BlockingRule(sql),),
        comparisons=(ComparisonLevel("phone"),),
    )
    with pytest.raises(EntityResolutionError, match="equality|unsupported"):
        resolve_entities(df, config=cfg)


_VALID_EQUALITY_RULES = [
    "l.dob = r.dob",
    "lower(l.email) = lower(r.email)",
    "upper(l.email)=upper(r.email)",
    "l.dob = r.dob and substr(lower(l.name), 1, 3) = substr(lower(r.name), 1, 3)",
    "l.dob = r.dob AND right(l.email, 5) = right(r.email, 5)",
    "lower(left(l.name,4)) = lower(left(r.name,4))",
    "trim(l.dob) = trim(r.dob)",
    "l.dob == r.dob",
    '  l."dob" = r."dob"\n  AND l."email" = r."email"  ',
]


@pytest.mark.parametrize("sql", _VALID_EQUALITY_RULES)
def test_pandas_blocking_equality_rules_match_duckdb(sql):
    # #237: the stricter parser must keep every valid equality rule working.
    pytest.importorskip("duckdb")
    df = _people()
    cfg_p = _config("pandas", blocking_rules=(BlockingRule(sql),))
    cfg_d = _config("duckdb", blocking_rules=(BlockingRule(sql),))
    _o, rp = resolve_entities(df, config=cfg_p)
    _o, rd = resolve_entities(df, config=cfg_d)
    assert rp.n_candidate_pairs > 0
    assert {(p.left_id, p.right_id) for p in rp.pairs} == {
        (p.left_id, p.right_id) for p in rd.pairs
    }


def test_pandas_blocking_quoted_identifiers():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "rock and roll": ["x", "x", "y"],
            'say "hi"': ["a", "a", "a"],
            "a=b": ["k", "k", "k"],
        }
    )
    sql = (
        'l."rock and roll" = r."rock and roll" and l."say ""hi""" = r."say ""hi""" '
        'and l."a=b" = r."a=b"'
    )
    cfg = _config(
        "pandas",
        blocking_rules=(BlockingRule(sql),),
        comparisons=(ComparisonLevel("a=b", "exact"),),
    )
    _out, report = resolve_entities(df, config=cfg)
    assert [(p.left_id, p.right_id) for p in report.pairs] == [(1, 2)]
    assert report.pairs[0].blocking_rule_ids == ("block_000",)


def test_pandas_blocking_warns_on_unknown_column(caplog):
    df = _people()
    cfg = _config("pandas", blocking_rules=(BlockingRule("l.emial = r.emial"),))
    with caplog.at_level("WARNING", logger="freshdata.enterprise.entity_resolution"):
        _out, report = resolve_entities(df, config=cfg)
    assert report.n_candidate_pairs == 0
    assert "emial" in caplog.text


@pytest.mark.parametrize(
    "value", [None, float("nan"), np.nan, pd.NaT, pd.NA, np.datetime64("NaT", "ns")]
)
def test_is_missing_recognises_scalar_missing_values(value):
    assert _is_missing(value)


@pytest.mark.parametrize(
    "value", ["", "NaT", "nan", b"", 0, 0.0, pd.Timestamp("2020-01-01"), [None]]
)
def test_is_missing_keeps_real_values(value):
    assert not _is_missing(value)


def test_missing_datetimes_do_not_agree():
    # #238: two NaT values used to score 1.0 and merge the records.
    df = pd.DataFrame(
        {"id": [1, 2], "zip": ["10001", "10001"], "dob": pd.to_datetime([None, None])}
    )
    cfg = EntityResolutionConfig(
        backend="pandas",
        blocking_rules=(BlockingRule("l.zip = r.zip"),),
        comparisons=(ComparisonLevel("dob", "exact"),),
    )
    frame, report = resolve_entities(df, config=cfg)
    pair = report.pairs[0]
    assert pair.comparison_vector == {"dob": 0.0}
    assert pair.decision == "non_match"
    assert frame["cluster_id"].nunique() == 2
    assert report.clusters == []


@pytest.mark.parametrize("kind", ["exact", "date_distance"])
def test_missing_datetime_on_one_side_is_missing(kind):
    df = pd.DataFrame(
        {"id": [1, 2], "zip": ["1", "1"], "dob": pd.to_datetime(["2020-01-01", None])}
    )
    cfg = EntityResolutionConfig(
        backend="pandas",
        blocking_rules=(BlockingRule("l.zip = r.zip"),),
        comparisons=(ComparisonLevel("dob", kind, threshold=5.0),),
    )
    _frame, report = resolve_entities(df, config=cfg)
    assert report.pairs[0].comparison_vector == {"dob": 0.0}
    assert "missing" in report.pairs[0].explanation[0].rationale


def test_pd_na_values_do_not_agree():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "zip": ["1", "1"],
            "code": pd.array([None, None], dtype="string"),
            "n": pd.array([None, None], dtype="Int64"),
        }
    )
    cfg = EntityResolutionConfig(
        backend="pandas",
        blocking_rules=(BlockingRule("l.zip = r.zip"),),
        comparisons=(ComparisonLevel("code", "exact"), ComparisonLevel("n", "exact")),
    )
    _frame, report = resolve_entities(df, config=cfg)
    assert report.pairs[0].comparison_vector == {"code": 0.0, "n": 0.0}
    assert report.pairs[0].decision == "non_match"


@pytest.mark.parametrize(
    "column",
    [
        pd.to_datetime([None, None]),
        pd.array([None, None], dtype="string"),
        pd.array([None, None], dtype="Int64"),
    ],
)
def test_missing_blocking_keys_do_not_block_together(column):
    # #238: NaT / pd.NA blocking keys must not form candidate pairs.
    df = pd.DataFrame({"id": [1, 2], "key": column, "name": ["a", "a"]})
    cfg = EntityResolutionConfig(
        backend="pandas",
        blocking_rules=(BlockingRule("l.key = r.key"),),
        comparisons=(ComparisonLevel("name", "exact"),),
    )
    _frame, report = resolve_entities(df, config=cfg)
    assert report.n_candidate_pairs == 0


def test_link_entities_records_thresholds_for_review_queue():
    # #271: link_entities must record thresholds like resolve_entities does.
    base = "a" * 30
    left = pd.DataFrame({"id": ["l1", "l2"], "e": ["x", "y"], "n": [base, base]})
    right = pd.DataFrame(
        {"id": ["r1", "r2"], "e": ["x", "y"], "n": ["bb" + base[2:], "bbb" + base[3:]]}
    )
    cfg = EntityResolutionConfig(
        backend="pandas",
        blocking_rules=(BlockingRule("l.e = r.e"),),
        comparisons=(ComparisonLevel("n", "levenshtein"),),
        match_threshold=0.95,
        clerical_review_threshold=0.9,
    )
    _o, linked = link_entities(left, right, config=cfg)
    _o, resolved = resolve_entities(pd.concat([left, right], ignore_index=True), config=cfg)
    assert linked.runtime_metadata == resolved.runtime_metadata
    assert linked.runtime_metadata["match_threshold"] == 0.95
    assert linked.runtime_metadata["clerical_review_threshold"] == 0.9

    def order(rep):
        return [round(i.score, 3) for i in fd.build_review_queue(rep).items]

    assert order(linked) == order(resolved) == [0.933, 0.9]
