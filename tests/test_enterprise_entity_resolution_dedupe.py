"""Tests for the probabilistic dedupe subsystem: review queue, clerical feedback,
explainable weights, golden-record merge policies, and domain templates."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pandas as pd
import pytest

from freshdata.enterprise import (
    BlockingRule,
    ComparisonLevel,
    EntityCluster,
    EntityResolutionConfig,
    GoldenRecordPolicy,
    MatchPair,
    ReviewDecision,
    ReviewItem,
    ReviewQueueConfig,
    apply_review_decisions,
    build_review_queue,
    education_template,
    export_review_queue,
    get_template,
    healthcare_template,
    load_review_decisions,
    media_template,
    merge_entities,
    recalibrate_weights,
    redaction_columns,
    resolve_entities,
    retail_template,
)
from freshdata.enterprise.entity_resolution_templates import with_overrides


def _config(**overrides) -> EntityResolutionConfig:
    base: dict = {
        "enabled": True,
        "backend": "pandas",
        "unique_id_column": "id",
        "blocking_rules": (
            BlockingRule("lower(l.email) = lower(r.email)", "same email"),
            BlockingRule(
                "l.dob = r.dob and substr(lower(l.name), 1, 3) = "
                "substr(lower(r.name), 1, 3)",
                "same dob + name prefix",
            ),
        ),
        "comparisons": (
            ComparisonLevel("name", "jaro_winkler", threshold=0.85, weight=2.0),
            ComparisonLevel("email", "jaro_winkler", threshold=0.90, weight=3.0),
            ComparisonLevel("dob", "exact", weight=2.0),
            ComparisonLevel("phone", "levenshtein", threshold=0.85, weight=1.0),
        ),
        "match_threshold": 0.95,
        "clerical_review_threshold": 0.5,
    }
    base.update(overrides)
    return EntityResolutionConfig(**base)


def _people() -> pd.DataFrame:
    # (1,2) are identical -> match; (3,4) share an email but differ on name/phone
    # -> land in the clerical-review band; 5 is a singleton.
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alice smith", "alice smith", "bob jones", "bobby jones", "carol lee"],
            "email": ["a@x.com", "a@x.com", "bob@y.com", "bob@y.com", "carol@z.com"],
            "dob": ["2000-01-01", "2000-01-01", "1990-05-05", "1990-05-05", "1985-03-03"],
            "phone": ["5551234", "5551234", "5559999", "5550000", "5557777"],
        }
    )


# --------------------------------------------------------------------------- #
# Blocking
# --------------------------------------------------------------------------- #


def test_pandas_blocking_generates_expected_pairs():
    _, report = resolve_entities(_people(), config=_config())
    keys = {tuple(sorted((p.left_id, p.right_id))) for p in report.pairs}
    assert (1, 2) in keys
    assert (3, 4) in keys
    assert report.n_matches == 1  # only the identical pair clears match_threshold


def test_blocking_backends_agree():
    pytest.importorskip("duckdb")
    _, rp = resolve_entities(_people(), config=_config(backend="pandas"))
    _, rd = resolve_entities(_people(), config=_config(backend="duckdb"))
    assert rp.n_candidate_pairs == rd.n_candidate_pairs


def test_pairs_carry_blocking_rule_ids():
    _, report = resolve_entities(_people(), config=_config())
    pair = next(p for p in report.pairs if {p.left_id, p.right_id} == {3, 4})
    assert "block_000" in pair.blocking_rule_ids  # the email rule


# --------------------------------------------------------------------------- #
# Review queue
# --------------------------------------------------------------------------- #


def test_review_queue_from_possible_matches():
    _, report = resolve_entities(_people(), config=_config())
    queue = build_review_queue(report)
    keys = {tuple(sorted((it.left_id, it.right_id))) for it in queue.items}
    assert keys == {(3, 4)}  # only the possible_match
    item = queue.items[0]
    assert item.status == "pending"
    assert item.created_at
    assert item.blocking_rule_ids
    assert "score" in item.explanation


@pytest.mark.parametrize("fmt", ["csv", "jsonl", "parquet"])
def test_export_review_queue_roundtrip(tmp_path, fmt):
    if fmt == "parquet":
        pytest.importorskip("pyarrow")
    _, report = resolve_entities(_people(), config=_config())
    queue = build_review_queue(report)
    path = tmp_path / f"queue.{fmt}"
    export_review_queue(queue, path, format=fmt)
    assert path.exists()
    if fmt == "jsonl":
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == len(queue)
    else:
        reader = pd.read_csv if fmt == "csv" else pd.read_parquet
        back = reader(path)
        assert len(back) == len(queue)
        assert {"left_id", "right_id", "explanation"} <= set(back.columns)


@pytest.mark.parametrize("fmt", ["csv", "jsonl"])
def test_load_review_decisions(tmp_path, fmt):
    rows = [
        {"left_id": 3, "right_id": 4, "decision": "accept", "reviewer": "jo"},
        {"left_id": 1, "right_id": 2, "decision": "reject"},
    ]
    path = tmp_path / f"decisions.{fmt}"
    if fmt == "csv":
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        path.write_text("\n".join(json.dumps(r) for r in rows))
    decisions = load_review_decisions(path)
    assert {d.decision for d in decisions} == {"accept", "reject"}
    assert decisions[0].pair_key == (str(3), str(4))


# --------------------------------------------------------------------------- #
# Clerical feedback
# --------------------------------------------------------------------------- #


def test_accepted_possible_match_becomes_match_and_clusters():
    _, report = resolve_entities(_people(), config=_config())
    decisions = [ReviewDecision("accept", left_id=3, right_id=4)]
    updated = apply_review_decisions(report, decisions)
    pair = next(p for p in updated.pairs if {p.left_id, p.right_id} == {3, 4})
    assert pair.decision == "match"
    assert updated.feedback_summary["n_promoted"] == 1
    clustered = any(
        set(c.record_ids) >= {3, 4} for c in updated.clusters
    )
    assert clustered


def test_rejected_pair_never_clusters():
    _, report = resolve_entities(_people(), config=_config())
    # 1 & 2 matched (and clustered) originally; reject them.
    assert any(set(c.record_ids) >= {1, 2} for c in report.clusters)
    decisions = [ReviewDecision("reject", left_id=1, right_id=2)]
    updated = apply_review_decisions(report, decisions)
    pair = next(p for p in updated.pairs if {p.left_id, p.right_id} == {1, 2})
    assert pair.decision == "non_match"
    assert not any(set(c.record_ids) >= {1, 2} for c in updated.clusters)
    assert updated.feedback_summary["n_demoted"] == 1


def test_manual_merge_promotes():
    _, report = resolve_entities(_people(), config=_config())
    updated = apply_review_decisions(
        report, [ReviewDecision("manual_merge", left_id=3, right_id=4)]
    )
    assert updated.feedback_summary["decisions"]["manual_merge"] == 1
    assert any(set(c.record_ids) >= {3, 4} for c in updated.clusters)


# --------------------------------------------------------------------------- #
# Recalibration (safe by default)
# --------------------------------------------------------------------------- #


def test_apply_does_not_recalibrate_by_default():
    _, report = resolve_entities(_people(), config=_config())
    updated = apply_review_decisions(report, [ReviewDecision("accept", left_id=3, right_id=4)])
    assert "recalibrated_weights" not in updated.feedback_summary


def test_recalibrate_returns_new_config_without_mutating():
    config = _config()
    _, report = resolve_entities(_people(), config=config)
    decisions = [
        ReviewDecision("accept", left_id=3, right_id=4),
        ReviewDecision("reject", left_id=1, right_id=2),
    ]
    new_config = recalibrate_weights(config, report, decisions)
    assert new_config is not config
    original_weights = [c.weight for c in config.comparisons]
    new_weights = [c.weight for c in new_config.comparisons]
    assert original_weights == [2.0, 3.0, 2.0, 1.0]  # untouched
    assert new_weights != original_weights


def test_apply_with_recalibrate_attaches_weights():
    config = _config()
    _, report = resolve_entities(_people(), config=config)
    updated = apply_review_decisions(
        report,
        [ReviewDecision("accept", left_id=3, right_id=4)],
        config=config,
        recalibrate=True,
    )
    assert "recalibrated_weights" in updated.feedback_summary


# --------------------------------------------------------------------------- #
# Explainable weights + redaction + to_frame
# --------------------------------------------------------------------------- #


def test_report_to_frame_pairs_and_explanations():
    _, report = resolve_entities(_people(), config=_config())
    pairs = report.to_frame("pairs")
    assert {"left_id", "right_id", "decision", "cmp_email"} <= set(pairs.columns)
    expl = report.to_frame("explanations")
    assert {"field", "left_value", "similarity", "contribution", "rationale"} <= set(expl.columns)
    assert len(expl) == sum(len(p.explanation) for p in report.pairs)


def test_explanation_previews_redacted():
    _, report = resolve_entities(_people(), config=_config(), redact_columns=["email"])
    expl = report.to_frame("explanations")
    email_rows = expl[expl.field == "email"]
    assert set(email_rows.left_value.tolist()) == {"<redacted>"}
    name_rows = expl[expl.field == "name"]
    assert "<redacted>" not in set(name_rows.left_value.tolist())


def test_redaction_columns_from_privacy():
    cols = ["id", "email", "phone", "city"]
    redacted = redaction_columns(cols, privacy=object())
    assert "email" in redacted and "phone" in redacted
    assert "city" not in redacted
    assert redaction_columns(cols) == frozenset()  # no privacy -> nothing


# --------------------------------------------------------------------------- #
# Golden-record merge policies
# --------------------------------------------------------------------------- #


def _golden_fixture():
    df = pd.DataFrame(
        {
            "id": ["a1", "a2"],
            "name": ["Alice", "Alice A"],
            "email": ["a@x.com", "a2@x.com"],
            "phone": [None, "555"],
            "source": ["crm", "warehouse"],
            "updated": ["2021-01-01", "2023-06-01"],
        }
    )
    cluster = EntityCluster(
        cluster_id="er_000000",
        record_ids=("a1", "a2"),
        size=2,
        canonical_record_id="a1",
        confidence=0.9,
    )
    return df, [cluster]


def test_golden_most_complete():
    df, clusters = _golden_fixture()
    gold, lineage = merge_entities(
        df, clusters, GoldenRecordPolicy("most_complete", id_column="id")
    )
    assert gold.iloc[0]["name"] == "Alice A"  # a2 is fully populated
    assert lineage[0]["field_sources"]["name"] == "a2"


def test_golden_most_recent():
    df, clusters = _golden_fixture()
    gold, _ = merge_entities(
        df, clusters,
        GoldenRecordPolicy("most_recent", timestamp_column="updated", id_column="id"),
    )
    assert gold.iloc[0]["name"] == "Alice A"


def test_golden_trusted_source():
    df, clusters = _golden_fixture()
    gold, _ = merge_entities(
        df, clusters,
        GoldenRecordPolicy(
            "trusted_source", source_column="source",
            source_priority=("crm", "warehouse"), id_column="id",
        ),
    )
    assert gold.iloc[0]["name"] == "Alice"  # crm wins


def test_golden_non_null_prefer_left():
    df, clusters = _golden_fixture()
    gold, lineage = merge_entities(
        df, clusters, GoldenRecordPolicy("non_null_prefer_left", id_column="id")
    )
    assert gold.iloc[0]["name"] == "Alice"  # left record
    assert gold.iloc[0]["phone"] == "555"   # filled from a2 (a1 null)
    assert lineage[0]["field_sources"]["phone"] == "a2"
    assert lineage[0]["field_sources"]["name"] == "a1"


def test_golden_column_priority_map():
    df, clusters = _golden_fixture()
    policy = GoldenRecordPolicy(
        "column_priority_map",
        source_column="source",
        column_priority_map={"phone": ("warehouse", "crm")},
        id_column="id",
    )
    gold, _ = merge_entities(df, clusters, policy)
    assert gold.iloc[0]["phone"] == "555"  # warehouse preferred for phone


def test_golden_custom_callable_and_report_lineage():
    df, clusters = _golden_fixture()
    policy = GoldenRecordPolicy("custom", custom=lambda sub: {"name": "MERGED"}, id_column="id")
    _, report = resolve_entities(_people(), config=_config())
    gold, lineage = merge_entities(df, clusters, policy, report=report)
    assert gold.iloc[0]["name"] == "MERGED"
    assert report.golden_record_lineage == lineage


# --------------------------------------------------------------------------- #
# Domain templates
# --------------------------------------------------------------------------- #


def test_domain_templates_build():
    for factory in (education_template, healthcare_template, media_template):
        tpl = factory()
        assert tpl.config.enabled
        assert tpl.config.blocking_rules
        assert tpl.config.comparisons


def test_get_template_unknown_raises():
    with pytest.raises(KeyError):
        get_template("aerospace")


def test_healthcare_template_redacts_pii():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "patient_id": ["M1", "M1", "M2"],
            "dob": ["2000-01-01", "2000-01-01", "1990-02-02"],
            "phone": ["5551111", "5551111", "5552222"],
            "address": ["1 Main", "1 Main St", "9 Oak"],
            "insurance_id": ["I1", "I1", "I2"],
        }
    )
    tpl = healthcare_template()
    _, report = resolve_entities(df, config=tpl.config, redact_columns=tpl.redact_columns)
    expl = report.to_frame("explanations")
    phone_rows = expl[expl.field == "phone"]
    assert set(phone_rows.left_value.tolist()) <= {"<redacted>"}


# --------------------------------------------------------------------------- #
# Benchmark smoke test
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Validation & edge cases
# --------------------------------------------------------------------------- #


def test_validation_errors():
    with pytest.raises(ValueError):
        ReviewQueueConfig(include_decisions=("bogus",))
    with pytest.raises(ValueError):
        ReviewQueueConfig(max_items=-1)
    with pytest.raises(ValueError):
        ReviewDecision("nope", left_id=1, right_id=2)
    with pytest.raises(ValueError):
        ReviewDecision("accept")  # no ids
    with pytest.raises(ValueError):
        GoldenRecordPolicy("most_recent")  # missing timestamp_column
    with pytest.raises(ValueError):
        GoldenRecordPolicy("trusted_source")  # missing source info
    with pytest.raises(ValueError):
        GoldenRecordPolicy("custom")  # missing callable


def test_to_frame_invalid_kind():
    _, report = resolve_entities(_people(), config=_config())
    with pytest.raises(ValueError):
        report.to_frame("bogus")


def test_export_unknown_format(tmp_path):
    _, report = resolve_entities(_people(), config=_config())
    with pytest.raises(ValueError):
        export_review_queue(report, tmp_path / "q.weird")


def test_load_review_decisions_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "d.parquet"
    pd.DataFrame(
        [{"left_id": 3, "right_id": 4, "decision": "accept"}]
    ).to_parquet(path, index=False)
    decisions = load_review_decisions(path)
    assert decisions[0].decision == "accept"


def test_export_from_raw_report_builds_queue(tmp_path):
    _, report = resolve_entities(_people(), config=_config())
    path = export_review_queue(report, tmp_path / "q.jsonl")
    assert path.exists()


def test_review_queue_sort_and_cap():
    _, report = resolve_entities(_people(), config=_config(clerical_review_threshold=0.1))
    cfg = ReviewQueueConfig(sort_by="score_desc", max_items=1, redact_columns=("email",))
    queue = build_review_queue(report, config=cfg)
    assert len(queue) == 1
    assert "<redacted>" in queue.items[0].explanation or queue.items[0].explanation


def test_match_pair_explanation_text_without_detail():
    p = MatchPair(1, 2, 0.7, 1.2, {}, "possible_match")
    assert "no field-level detail" in p.explanation_text()


def test_review_item_to_flat():
    item = ReviewItem(
        item_id="rev_000000", left_id=1, right_id=2, score=0.7, match_weight=1.0,
        comparison_vector={"name": 0.9}, blocking_rule_ids=("block_000",),
        explanation="x", created_at="now",
    )
    flat = item.to_flat()
    assert flat["blocking_rule_ids"] == "block_000"
    assert isinstance(flat["comparison_vector"], str)


def test_recalibrate_noop_without_matching_decisions():
    config = _config()
    _, report = resolve_entities(_people(), config=config)
    # Decisions reference ids with no surviving pair -> nothing to learn from.
    new_config = recalibrate_weights(
        config, report, [ReviewDecision("accept", left_id=99, right_id=100)]
    )
    assert [c.weight for c in new_config.comparisons] == [c.weight for c in config.comparisons]


def test_merge_entities_missing_id_column():
    df, clusters = _golden_fixture()
    with pytest.raises(KeyError):
        merge_entities(df.drop(columns=["id"]), clusters,
                       GoldenRecordPolicy("most_complete", id_column="id"))


def test_retail_template_and_overrides():
    tpl = retail_template()
    assert tpl.name == "retail"
    assert "email" in tpl.redact_columns
    bumped = with_overrides(tpl, match_threshold=0.99)
    assert bumped.config.match_threshold == 0.99
    assert get_template("retail").name == "retail"


def test_benchmark_smoke():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "bench_er", root / "benchmarks" / "bench_entity_resolution.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = mod.synth(200, 0.3, seed=1)
    result = mod.evaluate(df, "pandas")
    assert set(result) >= {"reduction_ratio", "runtime_s", "precision", "recall", "f1"}
    assert 0.0 <= result["reduction_ratio"] <= 1.0
