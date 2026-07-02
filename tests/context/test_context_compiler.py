"""Compiler behaviour: conflicts, strict mode, schema-free compiles, resolution."""

import pytest

from freshdata import PolicyError, compile_context
from freshdata.context import ContextPolicy, resolve_policy

SCHEMA = ["cust_id", "email_addr", "age", "monthly_revenue", "status"]


def test_protection_beats_repair():
    policy = compile_context(
        "Never modify email_addr. email_addr must be valid emails.", columns=SCHEMA
    )
    (issue,) = [i for i in policy.issues if i.kind == "protection_conflict"]
    assert issue.severity == "error"
    assert issue.columns == ("email_addr",)
    assert len(issue.sentences) == 2  # both sides shown
    repair = [c for c in policy.constraints if c.rule == "valid_format"][0]
    assert repair.action == "validate_only"  # demoted, protection wins
    assert policy.is_protected("email_addr")


def test_restated_constraint_last_wins():
    policy = compile_context(
        "Age must be between 18 and 100. Age must be between 21 and 60.", columns=SCHEMA
    )
    ranges = [c for c in policy.constraints if c.rule == "range"]
    assert len(ranges) == 1
    assert ranges[0].params == {"lo": 21, "hi": 60}
    assert any(i.kind == "superseded" for i in policy.issues)


def test_domain_restated_last_wins():
    policy = compile_context(
        "This is a retail dataset. This is a wholesale dataset.", columns=SCHEMA
    )
    assert policy.dataset_domain == "wholesale"
    assert any(i.kind == "domain_overridden" for i in policy.issues)


def test_unresolved_reference_is_surfaced_not_guessed():
    policy = compile_context("blood_pressure must be between 0 and 200.", columns=SCHEMA)
    assert not policy.constraints
    (miss,) = policy.unresolved
    assert miss.ref == "blood_pressure"
    assert miss.candidates  # shortlist offered for disambiguation


def test_compiler_never_invents_columns():
    policy = compile_context("Nonexistent is unique.", columns=SCHEMA)
    assert all(c.column in SCHEMA for c in policy.constraints if c.column)
    assert policy.unresolved


# -- strict mode ---------------------------------------------------------------


def test_strict_unresolved_raises():
    with pytest.raises(PolicyError, match="unresolved column reference"):
        compile_context("blood_pressure is unique.", columns=SCHEMA, strict=True)


def test_strict_unparsed_raises():
    with pytest.raises(PolicyError, match="unparsed_sentence"):
        compile_context("The vibes are immaculate.", columns=SCHEMA, strict=True)


def test_strict_conflict_raises():
    with pytest.raises(PolicyError, match="protection_conflict"):
        compile_context(
            "Never modify email_addr. email_addr must be valid emails.",
            columns=SCHEMA,
            strict=True,
        )


def test_strict_clean_text_compiles():
    policy = compile_context("age is unique.", columns=SCHEMA, strict=True)
    assert policy.strict
    assert policy.constraints[0].column == "age"


# -- schema-free compile + later resolution ------------------------------------


def test_schema_free_compile_defers_resolution():
    policy = compile_context("CustomerID is unique. Never modify revenue.")
    assert all(c.column is None for c in policy.constraints)
    assert not policy.unresolved  # nothing to resolve against yet

    resolved = resolve_policy(policy, SCHEMA)
    by_rule = {c.rule: c for c in resolved.constraints}
    assert by_rule["unique"].column == "cust_id"
    assert by_rule["protected"].column == "monthly_revenue"
    assert resolved.source_text_sha256 == policy.source_text_sha256


def test_resolve_policy_reports_misses():
    policy = compile_context("heart_rate is unique.")
    resolved = resolve_policy(policy, SCHEMA)
    assert not resolved.constraints
    assert resolved.unresolved[0].ref == "heart_rate"


def test_resolve_policy_noop_when_fully_resolved():
    policy = compile_context("age is unique.", columns=SCHEMA)
    assert resolve_policy(policy, SCHEMA) is policy


def test_schema_free_dedup_key_resolves_later():
    policy = compile_context("Deduplicate by email and status.")
    resolved = resolve_policy(policy, SCHEMA)
    (constraint,) = resolved.constraints
    assert constraint.params["columns"] == ["email_addr", "status"]


def test_empty_text_compiles_to_empty_policy():
    policy = compile_context("", columns=SCHEMA)
    assert policy == ContextPolicy(
        constraints=(), unresolved=(), issues=(),
        source_text_sha256=policy.source_text_sha256,
    )
