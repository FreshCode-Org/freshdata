"""ContextPolicy types: round-trips, protection, thresholds, lowering."""

import json

import pytest

from freshdata import CleanConfig, compile_context
from freshdata.context import ContextPolicy

SCHEMA = ["cust_id", "full_name", "email_addr", "mobile", "age", "monthly_revenue", "city"]

ECOMMERCE = """
This is an ecommerce customer dataset.
CustomerID is unique.
Emails must be valid.
Phone numbers are Indian.
Missing Age should be estimated only if confidence >95%.
Never modify revenue values.
"""


@pytest.fixture()
def policy():
    return compile_context(ECOMMERCE, columns=SCHEMA)


def test_json_round_trip(policy, tmp_path):
    path = tmp_path / "policy.json"
    text = policy.to_json(path)
    assert json.loads(text) == policy.to_dict()
    assert ContextPolicy.from_json(path) == policy
    assert ContextPolicy.from_json(text) == policy
    assert ContextPolicy.from_dict(policy.to_dict()) == policy


def test_source_hash_is_stable(policy):
    again = compile_context(ECOMMERCE, columns=SCHEMA)
    assert policy.source_text_sha256 == again.source_text_sha256
    other = compile_context("Age is unique.")
    assert policy.source_text_sha256 != other.source_text_sha256


def test_is_protected(policy):
    assert policy.is_protected("monthly_revenue")
    assert not policy.is_protected("age")
    assert not policy.is_protected("nonexistent")


def test_is_protected_matches_unresolved_by_name():
    schema_free = compile_context("Never modify Monthly Revenue.")
    assert schema_free.constraints[0].column is None
    assert schema_free.is_protected("monthly_revenue")


def test_thresholds(policy):
    cfg = CleanConfig()
    thr = policy.thresholds("age", "impute", cfg)
    assert thr.auto == 0.95
    assert thr.from_policy
    default = policy.thresholds("city", "impute", cfg)
    assert default.auto == cfg.semantic_auto_threshold
    assert not default.from_policy
    # a policy can only raise the bar, never lower it
    high_cfg = CleanConfig(semantic_auto_threshold=0.99)
    assert policy.thresholds("age", "impute", high_cfg).auto == 0.99


def test_summary_mentions_every_constraint(policy):
    text = policy.summary()
    for token in ("unique", "valid_format", "locale_format", "impute_missing", "protected"):
        assert token in text
    assert "ecommerce_customer" in text


# -- lowering ------------------------------------------------------------------


def test_lower_is_pure_and_complete(policy):
    cfg = CleanConfig(preserve_columns=("city",), verbose=False)
    lowered = policy.lower(cfg)
    assert cfg.preserve_columns == ("city",)  # original untouched
    assert lowered.preserve_columns == ("city", "monthly_revenue")
    assert "cust_id" in lowered.id_columns
    cols = lowered.semantic_context["columns"]
    assert cols["email_addr"]["semantic_type"] == "email"
    assert cols["mobile"] == {"semantic_type": "phone", "region": "IN"}
    assert cols["age"]["impute_min_confidence"] == 0.95
    assert cols["monthly_revenue"]["mutable"] is False
    assert cols["cust_id"]["unique"] is True
    assert lowered.semantic_context["dataset"] == "ecommerce_customer"
    assert lowered.policy is policy
    assert lowered.context is None


def test_lower_merges_existing_semantic_context(policy):
    cfg = CleanConfig(
        semantic_context={"dataset": "mine", "columns": {"age": {"unit": "years"}}},
        verbose=False,
    )
    lowered = policy.lower(cfg)
    assert lowered.semantic_context["dataset"] == "mine"  # user hint wins
    assert lowered.semantic_context["columns"]["age"] == {
        "unit": "years",
        "impute_min_confidence": 0.95,
    }


def test_lower_is_idempotent(policy):
    cfg = CleanConfig(verbose=False)
    once = policy.lower(cfg)
    twice = policy.lower(once)
    assert twice.preserve_columns == once.preserve_columns
    assert twice.id_columns == once.id_columns
    assert twice.semantic_context == once.semantic_context


def test_lower_allowed_values_and_range():
    policy = compile_context(
        "Allowed status values are active, inactive. Age must be between 18 and 100.",
        columns=["status", "age"],
    )
    lowered = policy.lower(CleanConfig(verbose=False))
    cols = lowered.semantic_context["columns"]
    assert cols["status"]["allowed_values"] == ["active", "inactive"]
    assert cols["age"] == {"min_value": 18, "max_value": 100}


def test_lower_dedup_key_sets_duplicate_subset():
    policy = compile_context("Deduplicate by email and city.", columns=["email", "city", "age"])
    lowered = policy.lower(CleanConfig(verbose=False))
    assert lowered.duplicate_subset == ("email", "city")
    # an explicit user subset always wins
    explicit = policy.lower(CleanConfig(duplicate_subset=("age",), verbose=False))
    assert explicit.duplicate_subset == ("age",)


def test_lower_rejects_non_config():
    with pytest.raises(TypeError):
        compile_context("Age is unique.", columns=["age"]).lower({})
