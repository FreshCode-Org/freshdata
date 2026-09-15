"""Regression tests for the clerical review loop: export -> reviewer edit ->
load_review_decisions -> apply_review_decisions (#239, #240, #267, #268)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from freshdata.enterprise import (
    BlockingRule,
    ComparisonLevel,
    EntityCluster,
    EntityResolutionConfig,
    MatchPair,
    ReviewDecision,
    apply_review_decisions,
    build_review_queue,
    export_review_queue,
    load_review_decisions,
    resolve_entities,
)
from freshdata.enterprise.entity_resolution import (
    EntityResolutionReport,
    _coerce_id,
    _strip_formula_guard,
)


def _review_config() -> EntityResolutionConfig:
    # match_threshold=0.99 puts near-identical names in the clerical-review band.
    return EntityResolutionConfig(
        enabled=True,
        backend="pandas",
        blocking_rules=(BlockingRule("l.e = r.e"),),
        comparisons=(ComparisonLevel("n", "jaro_winkler"),),
        match_threshold=0.99,
        clerical_review_threshold=0.5,
    )


def _review_report(ids: list[str]) -> EntityResolutionReport:
    df = pd.DataFrame(
        {
            "id": ids,
            "e": ["x", "x", "y", "y"][: len(ids)],
            "n": ["ann", "anne", "bob", "bobb"][: len(ids)],
        }
    )
    return resolve_entities(df, config=_review_config())[1]


def _reviewer_fills_csv(path, decisions) -> None:
    """Simulate a reviewer editing the exported queue as text."""
    q = pd.read_csv(path, dtype=str, keep_default_na=False)
    q["decision"] = decisions
    q.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# #239: CSV round-trip keeps id spelling; blank decision cells are undecided
# --------------------------------------------------------------------------- #


def test_csv_roundtrip_keeps_leading_zero_ids(tmp_path):
    report = _review_report(["007", "008"])
    path = export_review_queue(report, tmp_path / "queue.csv")
    _reviewer_fills_csv(path, "accept")

    decisions = load_review_decisions(path)
    assert [(d.left_id, d.right_id) for d in decisions] == [("007", "008")]

    updated = apply_review_decisions(report, decisions)
    assert updated.feedback_summary["n_applied"] == 1
    assert updated.feedback_summary["n_unmatched"] == 0
    assert updated.n_matches == 1


def test_csv_blank_decision_cells_are_skipped(tmp_path):
    report = _review_report(["a", "b", "c", "d"])
    path = export_review_queue(report, tmp_path / "queue.csv")
    q = pd.read_csv(path)
    assert len(q) == 2
    q["decision"] = ["accept", None]  # the reviewer decided one row only
    q.to_csv(path, index=False)

    decisions = load_review_decisions(path)
    assert len(decisions) == 1
    assert decisions[0].decision == "accept"
    # Blank optional cells come back as None, not "" or NaN.
    assert decisions[0].reviewer is None
    assert decisions[0].decided_at is None
    assert decisions[0].note == ""
    assert apply_review_decisions(report, decisions).feedback_summary["n_applied"] == 1


def test_csv_whitespace_only_decision_is_skipped(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame(
        [
            {"left_id": "1", "right_id": "2", "decision": "  "},
            {"left_id": "3", "right_id": "4", "decision": "reject"},
        ]
    ).to_csv(path, index=False)
    assert [d.decision for d in load_review_decisions(path)] == ["reject"]


def test_jsonl_null_decision_is_skipped(tmp_path):
    path = tmp_path / "d.jsonl"
    rows = [
        {"left_id": 1, "right_id": 2, "decision": None, "note": None},
        {"left_id": 3, "right_id": 4, "decision": "accept", "note": None},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    decisions = load_review_decisions(path)
    assert [(d.left_id, d.decision, d.note) for d in decisions] == [(3, "accept", "")]


def test_parquet_null_decision_is_skipped(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "d.parquet"
    pd.DataFrame(
        {
            "left_id": [1, 3],
            "right_id": [2, 4],
            "decision": [None, "accept"],
            "reviewer": [None, None],
        }
    ).to_parquet(path, index=False)
    decisions = load_review_decisions(path)
    assert len(decisions) == 1
    assert decisions[0].decision == "accept"
    assert decisions[0].reviewer is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (float("nan"), None),
        (np.nan, None),
        (pd.NA, None),
        ("", None),
        ("007", "007"),
        (" ", " "),
        (0, 0),
        (7, 7),
    ],
)
def test_coerce_id(value, expected):
    assert _coerce_id(value) is expected or _coerce_id(value) == expected


# --------------------------------------------------------------------------- #
# #240: the CSV formula guard on id cells is undone on load
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prefix", ["+44", "-", "=", "@", " =", "\t="])
def test_csv_roundtrip_formula_like_ids(tmp_path, prefix):
    ids = [f"{prefix}10", f"{prefix}11"]
    report = _review_report(ids)
    path = export_review_queue(report, tmp_path / "queue.csv")
    # The export-side sanitizer is unchanged: the ids are still guarded on disk.
    assert f"'{ids[0]}" in path.read_text(encoding="utf-8")
    _reviewer_fills_csv(path, "accept")

    decisions = load_review_decisions(path)
    assert {decisions[0].left_id, decisions[0].right_id} == set(ids)
    updated = apply_review_decisions(report, decisions)
    assert updated.feedback_summary["n_applied"] == 1
    assert updated.n_matches == 1


def test_csv_unsanitized_export_roundtrip(tmp_path):
    report = _review_report(["+4410", "+4411"])
    path = export_review_queue(report, tmp_path / "queue.csv", sanitize_formulas=False)
    _reviewer_fills_csv(path, "accept")
    decisions = load_review_decisions(path)
    assert decisions[0].left_id == "+4410"
    assert apply_review_decisions(report, decisions).feedback_summary["n_applied"] == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("'+4410", "+4410"),
        ("'=SUM(A1)", "=SUM(A1)"),
        ("' =1", " =1"),
        ("'abc", "'abc"),  # not a guard: no formula prefix follows
        ("'", "'"),
        ("+4410", "+4410"),
        (7, 7),
        (None, None),
    ],
)
def test_strip_formula_guard(value, expected):
    assert _strip_formula_guard(value) == expected


def test_formula_guard_only_stripped_for_csv(tmp_path):
    # JSONL is never sanitized on export, so a leading ' is part of the id.
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"left_id": "'+1", "right_id": "'+2", "decision": "accept"}))
    decisions = load_review_decisions(path)
    assert (decisions[0].left_id, decisions[0].right_id) == ("'+1", "'+2")


def test_csv_guard_not_stripped_from_free_text(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame(
        [{"left_id": "1", "right_id": "2", "decision": "accept", "note": "'=cmd"}]
    ).to_csv(path, index=False)
    assert load_review_decisions(path)[0].note == "'=cmd"


# --------------------------------------------------------------------------- #
# #267: item_id-only decisions are resolved through the queue, or rejected
# --------------------------------------------------------------------------- #


def _two_record_report() -> EntityResolutionReport:
    df = pd.DataFrame({"id": ["a", "b"], "e": ["x", "x"], "n": ["ann", "anne"]})
    return resolve_entities(df, config=_review_config())[1]


def test_item_id_only_decision_resolved_via_queue():
    report = _two_record_report()
    queue = build_review_queue(report)
    item = queue.items[0]
    out = apply_review_decisions(
        report, [ReviewDecision("accept", item_id=item.item_id)], queue=queue
    )
    assert out.feedback_summary["n_applied"] == 1
    assert out.feedback_summary["n_promoted"] == 1
    assert out.n_matches == 1
    assert [set(c.record_ids) for c in out.clusters] == [{"a", "b"}]


def test_item_id_only_decision_without_queue_raises():
    report = _two_record_report()
    item = build_review_queue(report).items[0]
    with pytest.raises(ValueError, match=r"queue=") as exc:
        apply_review_decisions(report, [ReviewDecision("accept", item_id=item.item_id)])
    assert item.item_id in str(exc.value)


def test_unknown_item_id_raises_even_with_queue():
    report = _two_record_report()
    queue = build_review_queue(report)
    with pytest.raises(ValueError, match="rev_999999"):
        apply_review_decisions(
            report, [ReviewDecision("accept", item_id="rev_999999")], queue=queue
        )


def test_incomplete_pair_without_item_id_raises():
    report = _two_record_report()
    with pytest.raises(ValueError, match="left_id='a'"):
        apply_review_decisions(report, [ReviewDecision("accept", left_id="a")])


def test_pair_decision_wins_over_queue_lookup():
    report = _two_record_report()
    queue = build_review_queue(report)
    out = apply_review_decisions(
        report,
        [ReviewDecision("reject", left_id="b", right_id="a", item_id="rev_999999")],
        queue=queue,
    )
    assert out.feedback_summary["decisions"]["reject"] == 1


def test_n_unmatched_counts_decisions_for_unknown_pairs():
    report = _two_record_report()
    out = apply_review_decisions(
        report,
        [
            ReviewDecision("accept", left_id="a", right_id="b"),
            ReviewDecision("accept", left_id="a", right_id="zzz"),
        ],
    )
    assert out.feedback_summary["n_applied"] == 1
    assert out.feedback_summary["n_unmatched"] == 1


def test_recalibrate_uses_item_id_resolved_decisions():
    config = _review_config()
    report = _two_record_report()
    queue = build_review_queue(report)
    out = apply_review_decisions(
        report,
        [ReviewDecision("accept", item_id=queue.items[0].item_id)],
        config=config,
        recalibrate=True,
        queue=queue,
    )
    assert out.feedback_summary["recalibrated_weights"]["n"] != config.comparisons[0].weight


# --------------------------------------------------------------------------- #
# #268: applying decisions keeps cluster identity
# --------------------------------------------------------------------------- #


def test_apply_no_decisions_is_identity_on_clusters():
    df = pd.DataFrame({"id": [1, 2, 3, 4], "e": ["a", "b", "b", "c"], "x": [1, None, 5, 1]})
    cfg = EntityResolutionConfig(
        enabled=True,
        backend="pandas",
        blocking_rules=(BlockingRule("l.e = r.e"),),
        comparisons=(ComparisonLevel("e"),),
    )
    frame, report = resolve_entities(df, config=cfg)
    after = apply_review_decisions(report, [])

    assert [c.to_dict() for c in after.clusters] == [c.to_dict() for c in report.clusters]
    assert [(c.cluster_id, c.canonical_record_id) for c in after.clusters] == [("er_000001", 3)]
    assert set(frame["cluster_id"]) >= {c.cluster_id for c in after.clusters}
    assert after.n_clusters == report.n_clusters
    # The returned clusters are copies; the input report is not shared.
    assert all(a is not b for a, b in zip(after.clusters, report.clusters))


def _pair(left, right, decision, score=0.9) -> MatchPair:
    return MatchPair(left, right, score, score, {"f": score}, decision)


def _manual_report(pairs, clusters, n_records=8) -> EntityResolutionReport:
    return EntityResolutionReport(
        n_records=n_records,
        n_candidate_pairs=len(pairs),
        n_matches=sum(p.decision == "match" for p in pairs),
        n_possible_matches=sum(p.decision == "possible_match" for p in pairs),
        n_clusters=len(clusters),
        backend="pandas",
        pairs=pairs,
        clusters=clusters,
    )


def _merge_report() -> EntityResolutionReport:
    # Clusters {1,2} (canonical 2) and {3,4} (canonical 4); (2,3) links them;
    # (5,6) and (7,8) are singletons in the review band.
    pairs = [
        _pair(1, 2, "match", 0.99),
        _pair(3, 4, "match", 0.97),
        _pair(2, 3, "possible_match", 0.7),
        _pair(5, 6, "possible_match", 0.7),
        _pair(7, 8, "possible_match", 0.7),
    ]
    clusters = [
        EntityCluster("er_000000", (1, 2), 2, 2, 0.99),
        EntityCluster("er_000002", (3, 4), 2, 4, 0.97),
    ]
    return _manual_report(pairs, clusters)


def test_redundant_accept_keeps_cluster_unchanged():
    report = _merge_report()
    out = apply_review_decisions(report, [ReviewDecision("accept", left_id=2, right_id=1)])
    assert [c.to_dict() for c in out.clusters] == [c.to_dict() for c in report.clusters]


def test_merge_keeps_lowest_original_id_and_its_canonical():
    report = _merge_report()
    out = apply_review_decisions(report, [ReviewDecision("accept", left_id=2, right_id=3)])
    assert len(out.clusters) == 1
    merged = out.clusters[0]
    assert merged.cluster_id == "er_000000"
    assert merged.canonical_record_id == 2
    assert set(merged.record_ids) == {1, 2, 3, 4}
    assert merged.size == 4


def test_new_cluster_ids_do_not_collide_across_rounds():
    report = _merge_report()
    first = apply_review_decisions(report, [ReviewDecision("accept", left_id=5, right_id=6)])
    ids_first = [c.cluster_id for c in first.clusters]
    assert ids_first == ["er_000000", "er_000002", "er_000008"]  # n_records=8

    second = apply_review_decisions(first, [ReviewDecision("accept", left_id=7, right_id=8)])
    ids_second = [c.cluster_id for c in second.clusters]
    assert ids_second == ["er_000000", "er_000002", "er_000008", "er_000009"]
    assert len(set(ids_second)) == len(ids_second)
    new = next(c for c in second.clusters if c.cluster_id == "er_000009")
    assert new.canonical_record_id == 7


def _chain_report() -> EntityResolutionReport:
    pairs = [_pair(1, 2, "match"), _pair(2, 3, "match")]
    return _manual_report(pairs, [EntityCluster("er_000000", (1, 2, 3), 3, 3, 0.9)], 3)


def test_split_keeps_id_with_the_canonical_record():
    reject = ReviewDecision("reject", left_id=1, right_id=2)
    out = apply_review_decisions(_chain_report(), [reject])
    assert [(c.cluster_id, c.record_ids, c.canonical_record_id) for c in out.clusters] == [
        ("er_000000", (2, 3), 3)
    ]


def test_split_away_from_canonical_gets_fresh_id():
    reject = ReviewDecision("reject", left_id=2, right_id=3)
    out = apply_review_decisions(_chain_report(), [reject])
    assert [(c.cluster_id, c.record_ids, c.canonical_record_id) for c in out.clusters] == [
        ("er_000003", (1, 2), 1)
    ]


def test_accept_on_resolved_report_keeps_untouched_cluster_ids():
    # End to end: (1,2) cluster; (3,4) in review band. Accepting (3,4) must not
    # renumber the existing cluster or change its canonical record.
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alice smith", "alice smith", "bob jones", "robert jones", "carol lee"],
            "email": ["a@x.com", "a@x.com", "bob@y.com", "bob@y.com", "carol@z.com"],
        }
    )
    cfg = EntityResolutionConfig(
        enabled=True,
        backend="pandas",
        blocking_rules=(BlockingRule("lower(l.email) = lower(r.email)"),),
        comparisons=(
            ComparisonLevel("name", "jaro_winkler", threshold=0.99, weight=2.0),
            ComparisonLevel("email", "exact", weight=1.0),
        ),
        match_threshold=0.95,
        clerical_review_threshold=0.3,
    )
    frame, report = resolve_entities(df, config=cfg)
    assert [set(c.record_ids) for c in report.clusters] == [{1, 2}]
    before = report.clusters[0]

    out = apply_review_decisions(report, [ReviewDecision("accept", left_id=3, right_id=4)])
    kept = next(c for c in out.clusters if set(c.record_ids) == {1, 2})
    assert kept.to_dict() == before.to_dict()
    new = next(c for c in out.clusters if set(c.record_ids) == {3, 4})
    assert new.cluster_id not in set(frame["cluster_id"])
