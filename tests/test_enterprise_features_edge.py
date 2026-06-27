"""Edge-case and serialization coverage for the three new enterprise features."""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.enterprise.config import (
    BlockingRule,
    ComparisonLevel,
    DriftConfig,
    EntityResolutionConfig,
    MaskingRule,
    PIIDetectionConfig,
)
from freshdata.enterprise.contracts import (
    ColumnContract,
    DataContract,
    build_baseline,
    compare_to_baseline,
    load_baseline,
    save_baseline,
)
from freshdata.enterprise.entity_resolution import (
    EntityResolutionError,
    _make_expr,
    _split_args,
    jaro_winkler,
    resolve_entities,
    soundex,
)
from freshdata.enterprise.privacy import (
    KAnonymityReport,
    PIIScanReport,
    anonymize,
    check_k_anonymity,
    detect_in_text,
    detect_pii,
)

# --------------------------------------------------------------------------
# contracts edge cases
# --------------------------------------------------------------------------


def test_baseline_round_trip_with_contract(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    contract = DataContract(
        name="c",
        version="3.1.0",
        columns=(
            ColumnContract(name="x", dtype="int64", min_value=0, description="key"),
            ColumnContract(name="y", allowed_values=("a", "b", "c")),
        ),
        metadata={"owner": "data-eng"},
    )
    base = build_baseline(df, name="d", contract=contract, trust_score=88.5)
    path = tmp_path / "b.json"
    save_baseline(base, path)
    loaded = load_baseline(path)
    assert loaded.contract is not None
    assert loaded.contract.version == "3.1.0"
    assert loaded.contract.column("x").min_value == 0
    assert loaded.contract.column("missing") is None
    assert loaded.trust_score == 88.5


def test_drift_report_summary_and_dtype_warn():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    base = build_baseline(df, name="d")
    df2 = df.copy()
    df2["x"] = df2["x"].astype(str)
    contract = DataContract(
        name="c",
        columns=(ColumnContract(name="x", dtype="int64"),),
        fail_on_dtype_change=False,  # -> warning, not error
    )
    report = compare_to_baseline(df2, base, contract=contract, trust_score=95.0)
    text = report.summary()
    assert "drift report" in text
    assert "trust score: 95.0" in text
    assert any(f.check_id == "contract.dtype" and f.status == "warned" for f in report.findings)


def test_missing_required_warns_when_not_failing():
    df = pd.DataFrame({"a": [1]})
    base = build_baseline(df, name="d")
    contract = DataContract(
        name="c",
        columns=(ColumnContract(name="b", required=True),),
        fail_on_missing_required=False,
    )
    report = compare_to_baseline(df, base, contract=contract)
    assert any(
        f.check_id == "contract.missing_required" and f.status == "warned"
        for f in report.findings
    )


def test_contract_max_missing_ratio_and_cardinality():
    df = pd.DataFrame({"a": [1, None, None, None], "b": list(range(4))})
    base = build_baseline(df, name="d")
    contract = DataContract(
        name="c",
        columns=(
            ColumnContract(name="a", max_missing_ratio=0.1),
            ColumnContract(name="b", max_cardinality=2),
        ),
    )
    report = compare_to_baseline(df, base, contract=contract)
    ids = {f.check_id for f in report.errors}
    assert "contract.max_missing_ratio" in ids
    assert "contract.max_cardinality" in ids


def test_row_count_and_uniqueness_drift():
    base = build_baseline(pd.DataFrame({"a": list(range(100))}), name="d")
    # Far fewer rows + collapsed uniqueness.
    df2 = pd.DataFrame({"a": [1] * 10})
    report = compare_to_baseline(df2, base)
    ids = {f.check_id for f in report.findings}
    assert "stats.row_count" in ids
    assert "stats.uniqueness" in ids


def test_datetime_range_drift():
    base_df = pd.DataFrame({"ts": pd.date_range("2021-01-01", periods=40, freq="D")})
    base = build_baseline(base_df, name="d")
    df2 = pd.DataFrame({"ts": pd.date_range("2019-01-01", periods=40, freq="D")})
    report = compare_to_baseline(df2, base)
    assert any(f.check_id == "drift.datetime_range" for f in report.findings)


def test_distribution_skipped_below_min_samples():
    base = build_baseline(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), name="d")
    cfg = DriftConfig(min_samples_for_distribution=1000)
    report = compare_to_baseline(pd.DataFrame({"x": [9.0, 9.0, 9.0]}), base, drift_config=cfg)
    assert not report.distribution_drift


# --------------------------------------------------------------------------
# privacy edge cases
# --------------------------------------------------------------------------


def test_scan_report_serialization_and_grouping():
    df = pd.DataFrame({"notes": ["email a@b.com", "ssn 123-45-6789"]})
    report = detect_pii(df)
    assert isinstance(report, PIIScanReport)
    d = report.to_dict()
    assert d["n_entities"] == len(report.entities)
    assert "notes" in report.by_column()
    assert "EMAIL" in report.entity_types


def test_detect_custom_pattern():
    cfg = PIIDetectionConfig(
        custom_patterns=({"name": "BADGE", "regex": r"BDG-\d{4}", "score": 0.9},),
    )
    hits = detect_in_text("badge BDG-1234 here", config=cfg)
    assert any(e.entity_type == "BADGE" for e in hits)


def test_anonymize_drop_strategy_records_event():
    df = pd.DataFrame({"ssn": ["123-45-6789", "987-65-4321"], "keep": [1, 2]})
    rule = MaskingRule(name="d", columns=("ssn",), strategy="drop", entity_types=("SSN",))
    out, report = anonymize(df, rules=(rule,))
    assert "ssn" not in out.columns
    assert any(e.strategy == "drop" for e in report.events)


def test_anonymize_partial_and_regex_scrub_and_hash():
    df = pd.DataFrame(
        {
            "card": ["4111111111111111"],
            "free": ["call 555-12-3456"],
            "tok": ["value"],
        }
    )
    rules = (
        MaskingRule(name="p", columns=("card",), strategy="partial", visible=4),
        MaskingRule(name="r", columns=("free",), strategy="regex_scrub", scrub_patterns=("ssn",)),
        MaskingRule(name="t", columns=("tok",), strategy="tokenize"),  # no key -> default salt
    )
    out, report = anonymize(df, rules=rules)
    assert out["card"].iloc[0].endswith("1111")
    assert "555-12-3456" not in out["free"].iloc[0]
    assert out["tok"].iloc[0].startswith("tok_")
    assert report.to_json()  # serializable
    assert "anonymized" in report.summary()


def test_surrogate_email_preserves_domain():
    df = pd.DataFrame({"email": ["john.doe@company.com"]})
    rule = MaskingRule(
        name="s", columns=("email",), strategy="surrogate", preserve_format=True
    )
    out, _report = anonymize(df, rules=(rule,))
    assert out["email"].iloc[0].endswith("@company.com")
    assert not out["email"].iloc[0].startswith("john.doe")


def test_masking_event_to_dict_has_tags():
    df = pd.DataFrame({"email": ["a@b.com"]})
    rule = MaskingRule(name="e", columns=("email",), strategy="redact", entity_types=("EMAIL",))
    _out, report = anonymize(df, rules=(rule,))
    d = report.events[0].to_dict()
    assert d["hipaa_tag"] == "HIPAA:email"
    assert d["gdpr_tag"] == "GDPR:contact_data"


def test_k_anonymity_report_helpers_and_empty():
    report = check_k_anonymity(pd.DataFrame({"a": []}), ["a"], k=5)
    assert report.ok  # empty frame trivially ok
    full = check_k_anonymity(pd.DataFrame({"a": [1, 1, 2]}), ["a"], k=2)
    assert isinstance(full, KAnonymityReport)
    assert "k-anonymity" in full.summary()
    assert full.to_dict()["ok"] is False


def test_anonymize_detection_skips_non_text_columns():
    df = pd.DataFrame({"n": [1, 2, 3], "t": ["ssn 123-45-6789", "x", "y"]})
    out, report = anonymize(df, detection_config=PIIDetectionConfig())
    assert list(out["n"]) == [1, 2, 3]
    assert report.entities_found >= 1


# --------------------------------------------------------------------------
# entity-resolution edge cases
# --------------------------------------------------------------------------


def test_split_args_handles_nested():
    assert _split_args("lower(l.a), 1, 2") == ["lower(l.a)", "1", "2"]


def test_make_expr_variants():
    rec = {"name": "Robert", "code": "ABCDEF"}
    assert _make_expr("upper(l.name)")(rec) == "ROBERT"
    assert _make_expr("trim(r.name)")(rec) == "Robert"
    assert _make_expr("left(l.code, 2)")(rec) == "AB"
    assert _make_expr("right(l.code, 2)")(rec) == "EF"
    assert _make_expr("substr(l.code, 2, 3)")(rec) == "BCD"
    assert _make_expr("substr(l.code, 3)")(rec) == "CDEF"


def test_make_expr_unsupported_function():
    with pytest.raises(EntityResolutionError, match="unsupported"):
        _make_expr("md5(l.x)")


def test_numeric_and_date_and_phonetic_comparisons():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "amount": [100.0, 103.0],
            "when": ["2020-01-01", "2020-01-04"],
            "name": ["Robert", "Rupert"],
            "block": ["k", "k"],
        }
    )
    cfg = EntityResolutionConfig(
        unique_id_column="id",
        backend="pandas",
        blocking_rules=(BlockingRule(sql="l.block = r.block"),),
        comparisons=(
            ComparisonLevel(column="amount", kind="numeric_distance", threshold=5.0, weight=1.0),
            ComparisonLevel(column="when", kind="date_distance", threshold=7.0, weight=1.0),
            ComparisonLevel(column="name", kind="phonetic", weight=1.0),
        ),
        match_threshold=0.6,
        clerical_review_threshold=0.3,
    )
    _out, report = resolve_entities(df, config=cfg)
    pair = report.pairs[0]
    assert pair.comparison_vector["amount"] > 0  # within tolerance
    assert pair.comparison_vector["name"] == 1.0  # Robert ~ Rupert soundex


def test_custom_sql_comparison_is_skipped_in_python_scoring():
    df = pd.DataFrame({"id": [1, 2], "k": ["a", "a"], "name": ["x", "x"]})
    cfg = EntityResolutionConfig(
        unique_id_column="id",
        backend="pandas",
        blocking_rules=(BlockingRule(sql="l.k = r.k"),),
        comparisons=(
            ComparisonLevel(column="name", kind="exact", weight=1.0),
            ComparisonLevel(column="name", kind="custom_sql", sql="l.name = r.name", weight=5.0),
        ),
        match_threshold=0.9,
    )
    _out, report = resolve_entities(df, config=cfg)
    # custom_sql contributes nothing; exact name match still drives the score.
    assert "name" in report.pairs[0].comparison_vector


def test_resolve_requires_comparisons():
    df = pd.DataFrame({"id": [1], "k": ["a"]})
    cfg = EntityResolutionConfig(
        unique_id_column="id",
        backend="pandas",
        blocking_rules=(BlockingRule(sql="l.k = r.k"),),
        comparisons=(),
    )
    with pytest.raises(EntityResolutionError, match="ComparisonLevel"):
        resolve_entities(df, config=cfg)


def test_resolve_missing_id_column():
    df = pd.DataFrame({"k": ["a", "a"]})
    cfg = EntityResolutionConfig(
        unique_id_column="id",
        backend="pandas",
        blocking_rules=(BlockingRule(sql="l.k = r.k"),),
        comparisons=(ComparisonLevel(column="k", kind="exact"),),
    )
    with pytest.raises(KeyError):
        resolve_entities(df, config=cfg)


def test_jaro_winkler_and_soundex_edges():
    assert jaro_winkler("a", "b") == 0.0
    assert soundex("123") == "0000"
