"""Privacy policy classification: value coverage, rule priority, keys and labels.

* #246: classification reads every distinct value, not the first 200 cells.
* #284: a matching inline rule beats every pack rule.
* #285: key precedence is rule.key_env, rule.key, policy.key_env, policy.key.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from freshdata.enterprise import (
    PIIDetectionConfig,
    PrivacyPolicy,
    PrivacyRule,
    apply_privacy_policy,
    classify_columns,
    load_compliance_pack,
    privacy_policy,
)

KEY = "unit-test-key"


@pytest.fixture
def detect_calls(monkeypatch):
    """Count the detect_in_text calls made during classification."""
    calls: list[int] = []
    real = privacy_policy.detect_in_text

    def counting(text, **kwargs):
        calls.append(len(text))
        return real(text, **kwargs)

    monkeypatch.setattr(privacy_policy, "detect_in_text", counting)
    return calls


def _distinct_words(n: int, width: int) -> list[str]:
    """*n* distinct lowercase words of *width* characters that match no detector."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    words = []
    for i in range(n):
        stem = "".join(letters[(i // 26**k) % 26] for k in range(3))
        words.append(stem + "q" * (width - len(stem)))
    return words


# --------------------------------------------------------------------------
# #246: classify from all distinct values
# --------------------------------------------------------------------------

_PII = "patient email jane.roe@clinic.org ssn 123-45-6789"


@pytest.mark.parametrize("n_before", [10, 250, 10_000])
def test_issue_246_late_pii_is_classified(n_before):
    pol = PrivacyPolicy(packs=(load_compliance_pack("hipaa"),), jurisdiction="US")
    df = pd.DataFrame({"notes": ["follow-up scheduled"] * n_before + [_PII]})
    out, rep = apply_privacy_policy(df, pol)
    assert out["notes"].iloc[-1] == "patient email <EMAIL> ssn <SSN>"
    assert sorted(rep.classifications) == ["notes"]
    assert rep.metadata["classification_values_scanned"] == {"notes": 2}


def test_late_value_regex_match_is_classified():
    rule = PrivacyRule(id="ticket", action="redact", value_regexes=(r"\bTKT-\d{6}\b",))
    df = pd.DataFrame({"ref": [f"note {i}" for i in range(5_000)] + ["TKT-123456"]})
    cls = classify_columns(df, PrivacyPolicy(rules=(rule,)))
    assert cls["ref"].rule is rule
    assert cls["ref"].matched_by == "regex"


def test_late_context_keyword_is_classified():
    rule = PrivacyRule(id="conf", action="quarantine", context=("confidential",))
    df = pd.DataFrame({"memo": [f"memo {i}" for i in range(5_000)] + ["CONFIDENTIAL draft"]})
    out, rep = apply_privacy_policy(df, PrivacyPolicy(rules=(rule,)))
    assert rep.classifications["memo"]["matched_by"] == "context"
    assert set(out["memo"]) == {"<QUARANTINED>"}


def test_late_luhn_value_satisfies_requires_luhn():
    rule = PrivacyRule(id="pan", action="redact", requires_luhn=True, value_regexes=(r"\d{4}",))
    head = ["order 1234 5678 9012 3456"] * 300  # 16 digits, fails Luhn
    policy = PrivacyPolicy(rules=(rule,))
    assert "ref" not in classify_columns(pd.DataFrame({"ref": head}), policy)
    cls = classify_columns(pd.DataFrame({"ref": head + ["4111 1111 1111 1111"]}), policy)
    assert cls["ref"].rule is rule


def test_detect_calls_match_chunk_count(detect_calls):
    values = _distinct_words(640, 999)
    chars = len("\n".join(values))
    cls = classify_columns(pd.DataFrame({"words": values}), PrivacyPolicy())
    assert cls == {}
    assert len(detect_calls) == math.ceil(chars / privacy_policy._CLASSIFY_CHUNK_CHARS)
    assert max(detect_calls) <= privacy_policy._CLASSIFY_CHUNK_CHARS


def test_repeated_values_are_read_once(detect_calls):
    values = _distinct_words(640, 999)
    classify_columns(pd.DataFrame({"words": values * 5}), PrivacyPolicy())
    assert len(detect_calls) == 10
    detect_calls.clear()
    _, rep = apply_privacy_policy(pd.DataFrame({"same": ["repeated"] * 10_000}), PrivacyPolicy())
    assert len(detect_calls) == 1
    assert rep.metadata["classification_values_scanned"] == {"same": 1}


def test_detection_stops_once_every_entity_type_is_found(detect_calls):
    values = ["mail a@b.com", *_distinct_words(200, 999)]
    policy = PrivacyPolicy(detection_config=PIIDetectionConfig(entities=("EMAIL",)))
    cls = classify_columns(pd.DataFrame({"notes": values}), policy)
    assert cls["notes"].classification == "detected PII: EMAIL"
    assert len(detect_calls) == 1


def test_oversized_value_forms_its_own_chunk():
    limit = privacy_policy._CLASSIFY_CHUNK_CHARS
    big = "q" * (limit + 10)
    chunks = privacy_policy._pack_chunks(["a", big, "b", "c"], limit)
    assert chunks == ["a", big, "b\nc"]
    assert privacy_policy._pack_chunks([], limit) == []


def test_values_scanned_ignores_missing_cells():
    df = pd.DataFrame({"s": pd.array(["x", "x", None, "y"], dtype="string"), "n": [1, 1, 2, 2]})
    _, rep = apply_privacy_policy(df, PrivacyPolicy())
    assert rep.metadata["classification_values_scanned"] == {"s": 2, "n": 2}


def test_unhashable_cells_are_classified():
    df = pd.DataFrame({"tags": [["a"], ["a"], ["mail a@b.com"]]})
    _, rep = apply_privacy_policy(df, PrivacyPolicy())
    assert rep.classifications["tags"]["classification"] == "detected PII: EMAIL"
    assert rep.metadata["classification_values_scanned"] == {"tags": 2}


# --------------------------------------------------------------------------
# #284: inline rules take priority over pack rules
# --------------------------------------------------------------------------


def _gdpr_policy(*rules: PrivacyRule, jurisdiction: str = "EU") -> PrivacyPolicy:
    return PrivacyPolicy(
        rules=rules, packs=(load_compliance_pack("gdpr"),), jurisdiction=jurisdiction
    )


def test_issue_284_inline_entity_rule_beats_pack_column_name_rule():
    inline = PrivacyRule(id="inline.email_drop", action="drop", entity_types=("EMAIL",))
    out, rep = apply_privacy_policy(
        pd.DataFrame({"email": ["a@b.com"], "v": [1]}), _gdpr_policy(inline)
    )
    assert rep.classifications["email"]["rule_id"] == "inline.email_drop"
    assert rep.classifications["email"]["matched_by"] == "entity"
    assert list(out.columns) == ["v"]


def test_inline_context_rule_beats_pack_column_name_rule():
    inline = PrivacyRule(id="inline.hold", action="quarantine", context=("contact",))
    df = pd.DataFrame({"email": ["contact a@b.com", "c@d.com"]})
    out, rep = apply_privacy_policy(df, _gdpr_policy(inline))
    assert rep.classifications["email"]["rule_id"] == "inline.hold"
    assert rep.classifications["email"]["matched_by"] == "context"
    assert list(out["email"]) == ["<QUARANTINED>", "<QUARANTINED>"]


def test_specificity_still_applies_within_inline_rules():
    by_context = PrivacyRule(id="by_context", action="redact", context=("contact",))
    by_name = PrivacyRule(id="by_name", action="drop", columns=("email",))
    by_name_too = PrivacyRule(id="by_name_too", action="quarantine", columns=("email",))
    df = pd.DataFrame({"email": ["contact a@b.com"]})
    cls = classify_columns(df, _gdpr_policy(by_context, by_name, by_name_too))
    assert cls["email"].rule is by_name
    assert cls["email"].matched_by == "column-name"


def test_out_of_scope_inline_rule_does_not_override_pack():
    us_only = PrivacyRule(
        id="us.email_drop", action="drop", entity_types=("EMAIL",), jurisdictions=("US",)
    )
    out, rep = apply_privacy_policy(pd.DataFrame({"email": ["a@b.com"]}), _gdpr_policy(us_only))
    assert rep.classifications["email"]["rule_id"] == "gdpr.email"
    assert list(out.columns) == ["email"]


def test_pack_only_policy_is_unchanged():
    df = pd.DataFrame(
        {
            "email": ["a@b.com"],
            "phone_number": ["+44 20 7946 0958"],
            "notes": ["reach me at c@d.com"],
        }
    )
    cls = classify_columns(df, _gdpr_policy())
    assert {c: (v.rule.id, v.matched_by) for c, v in cls.items() if v.rule} == {
        "email": ("gdpr.email", "column-name"),
        "phone_number": ("gdpr.phone", "column-name"),
        "notes": ("gdpr.email", "entity"),
    }


# --------------------------------------------------------------------------
# #285: key precedence
# --------------------------------------------------------------------------

_RULE_ENV = "FRESHDATA_TEST_RULE_KEY"
_POLICY_ENV = "FRESHDATA_TEST_POLICY_KEY"


@pytest.fixture
def clean_key_env(monkeypatch):
    for name in (_RULE_ENV, _POLICY_ENV, "ORG_DEFAULT_KEY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _tokenize(policy: PrivacyPolicy) -> str:
    df = pd.DataFrame({"ssn": ["123-45-6789"]})
    return apply_privacy_policy(df, policy)[0]["ssn"][0]


def test_issue_285_rule_key_beats_policy_key_env(clean_key_env):
    df = pd.DataFrame({"ssn": ["123-45-6789"]})
    rule = PrivacyRule(id="t", action="tokenize", columns=("ssn",), key="rule-specific-key")
    pol = PrivacyPolicy(rules=(rule,), key_env="ORG_DEFAULT_KEY")
    before = apply_privacy_policy(df, pol)[0]["ssn"][0]
    clean_key_env.setenv("ORG_DEFAULT_KEY", "org-wide-key")
    after = apply_privacy_policy(df, pol)[0]["ssn"][0]
    assert before == after == "tok_05817a27d410ecc4"


def test_rule_key_env_beats_rule_key(clean_key_env):
    rule = PrivacyRule(id="t", action="tokenize", key="rule-literal", key_env=_RULE_ENV)
    pol = PrivacyPolicy(rules=(rule,), key="policy-literal", key_env=_POLICY_ENV)
    clean_key_env.setenv(_POLICY_ENV, "policy-env")
    assert privacy_policy._resolve_key(rule, pol) == "rule-literal"
    clean_key_env.setenv(_RULE_ENV, "rule-env")
    assert privacy_policy._resolve_key(rule, pol) == "rule-env"
    clean_key_env.setenv(_RULE_ENV, "")
    assert privacy_policy._resolve_key(rule, pol) == "rule-literal"


def test_rule_key_env_without_rule_key_falls_back_to_policy(clean_key_env):
    rule = PrivacyRule(id="t", action="tokenize", key_env=_RULE_ENV)
    pol = PrivacyPolicy(rules=(rule,), key="policy-literal", key_env=_POLICY_ENV)
    assert privacy_policy._resolve_key(rule, pol) == "policy-literal"
    clean_key_env.setenv(_POLICY_ENV, "policy-env")
    assert privacy_policy._resolve_key(rule, pol) == "policy-env"


@pytest.mark.parametrize("rule", [None, PrivacyRule(id="keyless", action="tokenize")])
def test_keyless_rule_uses_policy_key_env_then_policy_key(clean_key_env, rule):
    pol = PrivacyPolicy(key="policy-literal", key_env=_POLICY_ENV)
    assert privacy_policy._resolve_key(rule, pol) == "policy-literal"
    clean_key_env.setenv(_POLICY_ENV, "policy-env")
    assert privacy_policy._resolve_key(rule, pol) == "policy-env"
    assert privacy_policy._resolve_key(rule, PrivacyPolicy()) is None


def test_keyless_rule_tokens_follow_policy_key(clean_key_env):
    keyless = PrivacyRule(id="t", action="tokenize", columns=("ssn",))
    keyed = PrivacyRule(id="t", action="tokenize", columns=("ssn",), key="policy-env")
    clean_key_env.setenv(_POLICY_ENV, "policy-env")
    via_policy = _tokenize(PrivacyPolicy(rules=(keyless,), key_env=_POLICY_ENV))
    assert via_policy == _tokenize(PrivacyPolicy(rules=(keyed,)))
    assert via_policy.startswith("tok_")


def test_tokenize_without_any_key_still_raises(clean_key_env):
    rule = PrivacyRule(id="t", action="tokenize", columns=("ssn",), key_env=_RULE_ENV)
    pol = PrivacyPolicy(rules=(rule,), key_env=_POLICY_ENV)
    with pytest.raises(ValueError, match="tokenize requires a key"):
        apply_privacy_policy(pd.DataFrame({"ssn": ["123-45-6789"]}), pol)
