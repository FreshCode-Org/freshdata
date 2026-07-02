"""Edge and serialization branches not exercised by the mainline tests."""

from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
from freshdata.context import (
    ContextPolicy,
    PolicyError,
    PolicyIssue,
    UnresolvedRef,
    apply_policy_to_config,
    compile_context,
    parse_context,
    resolve_policy,
    resolve_reference,
)
from freshdata.context.compiler import effective_columns
from freshdata.context.normalize import parse_scalar, singular, snake_ref, strip_quotes
from freshdata.context.types import Provenance, UnparsedSentence
from freshdata.context.validate import validate_frame

# -- normalize helpers ----------------------------------------------------------


def test_singular_rules():
    assert singular("emails") == "email"
    assert singular("countries") == "country"
    assert singular("addresses") == "address"
    assert singular("churches") == "church"
    assert singular("status") == "statu"  # cheap stemmer, by design
    assert singular("boss") == "boss"  # -ss never stripped
    assert singular("age") == "age"
    assert singular("id") == "id"  # short tokens untouched


def test_parse_scalar_and_quotes():
    assert parse_scalar("18") == 18
    assert parse_scalar("1.5") == 1.5
    assert parse_scalar("'text'") == "text"
    assert strip_quotes('  "Phone numbers", ') == "Phone numbers"
    assert snake_ref("Salary($)") == "salary"


# -- dataclass serialization ------------------------------------------------------


def test_intent_candidate_and_unparsed_to_dict():
    result = parse_context("CustomerID is unique. Total nonsense here.")
    candidate = result.candidates[0]
    payload = candidate.to_dict()
    assert payload["intent"] == "unique"
    assert payload["column_refs"] == ["CustomerID"]
    assert payload["provenance"]["sentence"] == "CustomerID is unique"
    unparsed = result.unparsed[0]
    assert isinstance(unparsed, UnparsedSentence)
    assert unparsed.to_dict()["sentence"] == "Total nonsense here"


def test_provenance_round_trip():
    prov = Provenance(sentence="Age is unique", tier=0, parse_confidence=1.0)
    assert Provenance.from_dict(prov.to_dict()) == prov


def test_from_json_accepts_pathlib_path(tmp_path):
    policy = compile_context("age is unique.", columns=["age"])
    path = tmp_path / "p.json"
    policy.to_json(path)
    assert ContextPolicy.from_json(Path(path)) == policy


def test_summary_shows_unresolved_issues_and_custom_kinds():
    policy = compile_context(
        "Deduplicate by email and city.\n"
        "Drop rows where age is missing.\n"
        "Allowed status values are a, b.\n"
        "quantity must be at least 1.\n"
        "heart_rate is unique.\n"
        "Gibberish sentence here.\n",
        columns=["email", "city", "age", "status", "quantity"],
    )
    text = policy.summary()
    assert "columns=['email', 'city']" in text
    assert "kind=drop_if" in text
    assert "values=['a', 'b']" in text
    assert "[1, None]" in text
    assert "unresolved references: 1" in text
    assert "heart_rate" in text
    assert "[warning] unparsed_sentence" in text


def test_thresholds_review_never_exceeds_auto():
    policy = compile_context(
        "Missing age should be estimated only if confidence >60%.", columns=["age"]
    )
    cfg = fd.CleanConfig(semantic_auto_threshold=0.9, semantic_review_threshold=0.7)
    thr = policy.thresholds("age", "impute", cfg)
    assert thr.auto == 0.9  # policy can only raise the bar
    assert thr.review <= thr.auto


# -- parser edge branches ----------------------------------------------------------


def test_valid_format_without_known_format_is_unparsed():
    result = parse_context("gizmo must be valid.")
    assert not result.candidates
    assert len(result.unparsed) == 1


def test_map_pairs_without_parseable_pairs_is_unparsed():
    result = parse_context("Map gender values: nothing useful")
    assert not result.candidates
    assert len(result.unparsed) == 1


def test_allowed_values_empty_list_is_unparsed():
    result = parse_context("status must be one of: ,,")
    assert not result.candidates


# -- resolver edge branches ----------------------------------------------------------


def test_duplicate_normalized_labels_are_ambiguous():
    r = resolve_reference("customer name", ["Customer Name", "customer_name"])
    assert r.column is None
    assert "normalize" in r.reason


def test_empty_schema():
    r = resolve_reference("age", [])
    assert r.column is None and r.reason == "schema is empty"


# -- compiler / apply edge branches ---------------------------------------------------


def test_effective_columns_without_normalization():
    cfg = fd.CleanConfig(column_names=False, verbose=False)
    assert effective_columns(None, ["Customer ID"], cfg) == ["Customer ID"]
    assert effective_columns(None, None, cfg) is None


def test_compile_against_dataframe_uses_normalized_labels():
    df = pd.DataFrame({"Customer ID": [1], "Monthly Revenue": [2.0]})
    policy = compile_context("Never modify revenue.", df=df)
    assert policy.constraints[0].column == "monthly_revenue"


def test_apply_policy_rejects_foreign_policy_object():
    cfg = fd.CleanConfig(verbose=False)
    object.__setattr__(cfg, "policy", {"not": "a policy"})  # bypass config validation
    with pytest.raises(TypeError, match="ContextPolicy"):
        apply_policy_to_config(cfg, columns=["age"])


def test_strict_supplied_policy_raises_at_clean_time():
    policy = compile_context("heart_rate is unique.")  # schema-free, defers
    df = pd.DataFrame({"age": [1, 2]})
    with pytest.raises(PolicyError, match="heart_rate"):
        fd.clean(df, policy=policy, strict=True, verbose=False)


def test_non_strict_supplied_policy_surfaces_warning():
    policy = compile_context("heart_rate is unique.")
    df = pd.DataFrame({"age": [1, 2]})
    _, report = fd.clean(df, policy=policy, return_report=True, verbose=False)
    assert any("heart_rate" in w for w in report.warnings)


def test_resolve_policy_dedup_key_member_missing():
    policy = compile_context("Deduplicate by email and heart_rate.")
    resolved = resolve_policy(policy, ["email", "age"])
    assert not resolved.constraints
    assert any(u.ref == "heart_rate" for u in resolved.unresolved)


# -- fd.validate edge branches ----------------------------------------------------------


def test_validate_strict_raises():
    df = pd.DataFrame({"age": [1]})
    with pytest.raises(PolicyError):
        fd.validate(df, context="Gibberish here.", strict=True)


def test_validate_range_one_sided():
    df = pd.DataFrame({"quantity": [0, 5]})
    findings = fd.validate(df, context="quantity must be at least 1.")
    (finding,) = findings.errors
    assert finding.extra["min_value"] == 1
    assert finding.extra["max_value"] is None


def test_validate_rejects_foreign_policy_object():
    cfg = fd.CleanConfig(verbose=False)
    object.__setattr__(cfg, "policy", "nope")
    with pytest.raises(TypeError, match="ContextPolicy"):
        validate_frame(pd.DataFrame({"a": [1]}), cfg)


def test_finding_list_to_dicts_redacts():
    df = pd.DataFrame({"status": ["active", "weird"]})
    findings = fd.validate(df, context="Allowed status values are active, inactive.")
    payload = findings.to_dicts()
    assert payload[0]["observed_value"] == "[redacted]"
    revealed = findings.to_dicts(include_pii=True)
    assert any(d["observed_value"] for d in revealed)


def test_config_strict_type_checked():
    with pytest.raises(TypeError, match="strict"):
        fd.CleanConfig(strict="yes")
    with pytest.raises(TypeError, match="context"):
        fd.CleanConfig(context=42)


def test_compile_dedup_key_with_unresolved_member():
    policy = compile_context(
        "Deduplicate by email and heart_rate.", columns=["email", "age"]
    )
    assert not policy.constraints  # the whole group is withheld
    assert any(u.ref == "heart_rate" for u in policy.unresolved)


def test_apply_policy_without_schema_lowers_resolved_constraints():
    policy = compile_context("Never modify revenue.", columns=["revenue", "age"])
    cfg = fd.CleanConfig(policy=policy, verbose=False)
    lowered = apply_policy_to_config(cfg)  # no df/columns: already resolved
    assert "revenue" in lowered.preserve_columns


def test_protected_column_meta_never_keeps_impute_hint():
    policy = compile_context(
        "Missing age should be estimated only if confidence >95%. Never modify age.",
        columns=["age"],
    )
    lowered = policy.lower(fd.CleanConfig(verbose=False))
    meta = lowered.semantic_context["columns"]["age"]
    assert meta["mutable"] is False
    assert "impute_min_confidence" not in meta


def test_unresolved_ref_and_issue_from_dict_defaults():
    ref = UnresolvedRef.from_dict({"ref": "x"})
    assert (ref.sentence, ref.reason, ref.candidates) == ("", "", ())
    issue = PolicyIssue.from_dict({"kind": "k", "message": "m"})
    assert (issue.severity, issue.sentences, issue.columns) == ("warning", (), ())


def test_validate_skips_constraints_for_absent_columns():
    policy = compile_context("revenue is unique.", columns=["revenue"])
    frame = pd.DataFrame({"age": [1, 2]})  # a different frame entirely
    findings = fd.validate(frame, policy=policy)
    assert not any(f.rule_name == "context.unique" for f in findings)


def test_validate_passes_when_values_all_allowed_and_in_range():
    frame = pd.DataFrame({"status": ["a", "a"], "age": [20, 30], "cust_id": [1, 2]})
    findings = fd.validate(
        frame,
        context=(
            "Allowed status values are a, b. Age must be between 18 and 100. "
            "cust_id is unique."
        ),
    )
    assert not findings.errors
