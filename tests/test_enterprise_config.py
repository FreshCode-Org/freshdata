"""Validation and construction tests for the enterprise config dataclasses."""

import pytest

from freshdata.enterprise import (
    ClusterConfig,
    EnterpriseConfig,
    LineageConfig,
    MaskingRule,
    SemanticValidatorConfig,
    TrustScoreWeights,
)

# -- MaskingRule ----------------------------------------------------------

def test_masking_rule_valid_and_coerces_to_tuples():
    rule = MaskingRule(name="pii", columns=["email", "phone"], strategy="hash")
    assert rule.columns == ("email", "phone")
    assert isinstance(rule.columns, tuple)
    assert rule.strategy == "hash"


def test_masking_rule_pattern_only_is_allowed():
    rule = MaskingRule(name="byname", pattern=r".*_ssn$", strategy="redact")
    assert rule.pattern == r".*_ssn$"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": "x", "columns": ("c",), "strategy": "nope"}, "strategy must be"),
        ({"name": "x"}, "selects nothing"),
        ({"name": "x", "columns": ("c",), "visible": -1}, "visible must be"),
        ({"name": "x", "columns": ("c",), "hash_length": 2}, "hash_length must be"),
        ({"name": "x", "columns": ("c",), "scrub_patterns": ("bogus",)}, "unknown scrub_patterns"),
    ],
)
def test_masking_rule_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MaskingRule(**kwargs)


# -- ClusterConfig --------------------------------------------------------

def test_cluster_config_defaults_and_coercion():
    cfg = ClusterConfig(columns=["a"])
    assert cfg.method == "fingerprint"
    assert cfg.columns == ("a",)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "nope"}, "method must be"),
        ({"canonical": "nope"}, "canonical must be"),
        ({"ngram_size": 0}, "ngram_size must be"),
        ({"min_cluster_size": 1}, "min_cluster_size must be"),
    ],
)
def test_cluster_config_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ClusterConfig(**kwargs)


# -- TrustScoreWeights ----------------------------------------------------

def test_trust_weights_normalized_sums_to_one():
    weights = TrustScoreWeights(completeness=2, validity=2, uniqueness=2, consistency=2)
    norm = weights.normalized()
    assert pytest.approx(sum(norm.values())) == 1.0
    assert pytest.approx(norm["completeness"]) == 0.25


def test_trust_weights_negative_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        TrustScoreWeights(completeness=-1)


def test_trust_weights_all_zero_rejected():
    with pytest.raises(ValueError, match="at least one"):
        TrustScoreWeights(completeness=0, validity=0, uniqueness=0, consistency=0)


# -- SemanticValidatorConfig ----------------------------------------------

def test_semantic_reference_config_coerces_tuple():
    cfg = SemanticValidatorConfig(name="c", kind="reference", reference=["US", "CA"],
                                  columns=["country"])
    assert cfg.reference == ("US", "CA")
    assert cfg.columns == ("country",)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": "c", "kind": "bogus"}, "kind must be"),
        ({"name": "c", "kind": "reference"}, "needs reference"),
        ({"name": "c", "kind": "regex"}, "needs regex"),
    ],
)
def test_semantic_config_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SemanticValidatorConfig(**kwargs)


# -- EnterpriseConfig -----------------------------------------------------

def test_enterprise_config_defaults():
    ec = EnterpriseConfig()
    assert ec.enable_masking is True
    assert ec.clustering is None
    assert isinstance(ec.trust_weights, TrustScoreWeights)
    assert isinstance(ec.lineage, LineageConfig)


def test_enterprise_config_with_overrides():
    ec = EnterpriseConfig().with_overrides(enable_clustering=True, fail_under_trust=80.0)
    assert ec.enable_clustering is True
    assert ec.fail_under_trust == 80.0


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        ({"masking": ("not a rule",)}, TypeError, "MaskingRule"),
        ({"semantic": ("nope",)}, TypeError, "SemanticValidatorConfig"),
        ({"clustering": "nope"}, TypeError, "ClusterConfig"),
        ({"fail_under_trust": 150.0}, ValueError, "fail_under_trust"),
    ],
)
def test_enterprise_config_invalid(kwargs, exc, match):
    with pytest.raises(exc, match=match):
        EnterpriseConfig(**kwargs)


def test_enterprise_config_is_hashable():
    # Frozen + tuple fields => usable as a dict key / in a set.
    ec = EnterpriseConfig(masking=(MaskingRule(name="p", columns=("e",)),))
    assert hash(ec) == hash(ec)
