"""Tests for F4 masking hardening: token alias, retention, policy provenance."""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.enterprise import MaskingRule, mask_dataframe


def test_token_is_alias_for_tokenize():
    rule = MaskingRule(name="e", columns=("email",), strategy="token")
    assert rule.strategy == "tokenize"


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError, match="strategy must be one of"):
        MaskingRule(name="x", columns=("a",), strategy="bogus")


def test_negative_retention_rejected():
    with pytest.raises(ValueError, match="retention_days"):
        MaskingRule(name="x", columns=("a",), strategy="hash", retention_days=-1)


def test_mask_report_carries_retention_and_provenance():
    df = pd.DataFrame({"ssn": ["111-22-3333", "444-55-6666"], "email": ["a@x.com", "b@y.com"]})
    rules = [
        MaskingRule(name="ssn-hash", columns=("ssn",), strategy="hash",
                    policy_id="HIPAA-164.514", policy_reason="direct identifier",
                    retention_days=30),
        MaskingRule(name="email-redact", columns=("email",), strategy="redact",
                    policy_id="GDPR-Art9", policy_reason="contact PII", retention_days=365),
    ]
    _, report = mask_dataframe(df, rules)

    assert report.retention == {"ssn": 30, "email": 365}
    prov = {p["column"]: p for p in report.policy_provenance}
    assert prov["ssn"]["policy_id"] == "HIPAA-164.514"
    assert prov["ssn"]["reason"] == "direct identifier"
    assert prov["ssn"]["strategy"] == "hash"
    assert prov["email"]["retention_days"] == 365

    d = report.to_dict()
    assert d["retention"]["ssn"] == 30
    assert len(d["policy_provenance"]) == 2


def test_retention_defaults_to_none():
    df = pd.DataFrame({"ssn": ["111-22-3333"]})
    rule = MaskingRule(name="ssn", columns=("ssn",), strategy="hash")
    _, report = mask_dataframe(df, [rule])
    assert report.retention["ssn"] is None
    assert report.policy_provenance[0]["policy_id"] is None
