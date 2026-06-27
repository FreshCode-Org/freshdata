"""Tests for PII detection + reversible / format-preserving anonymization (Feature 2)."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise.config import MaskingRule, PIIDetectionConfig
from freshdata.enterprise.privacy import (
    InMemoryTokenVault,
    JsonTokenVault,
    anonymize,
    check_k_anonymity,
    detect_in_text,
    detect_pii,
    detokenize_value,
    tokenize_value,
)


def test_regex_detection_basic_entities():
    text = (
        "email john@example.com phone +1 415 555 2671 ssn 123-45-6789 "
        "card 4111111111111111 ip 10.0.0.1 iban GB29NWBK60161331926819"
    )
    found = {e.entity_type for e in detect_in_text(text)}
    assert {"EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IP_ADDRESS", "IBAN"} <= found


def test_credit_card_checksum_boosts_score():
    # 4111111111111111 is Luhn-valid.
    hits = [e for e in detect_in_text("card 4111111111111111") if e.entity_type == "CREDIT_CARD"]
    assert hits and hits[0].source == "checksum"
    assert hits[0].score >= 0.95


def test_context_detection_improves_score():
    no_ctx = detect_in_text("dob 1990-05-02", config=PIIDetectionConfig(use_context=False))
    with_ctx = detect_in_text(
        "patient dob 1990-05-02", config=PIIDetectionConfig(use_context=True)
    )
    assert no_ctx and with_ctx
    assert with_ctx[0].score > no_ctx[0].score
    assert any(t.startswith("context:") for t in with_ctx[0].tags)


def test_free_text_column_with_multiple_entities():
    df = pd.DataFrame(
        {"notes": ["reach me at a@b.com or 123-45-6789", "card 4111 1111 1111 1111"]}
    )
    report = detect_pii(df)
    assert {"EMAIL", "SSN", "CREDIT_CARD"} <= report.entity_types
    assert "notes" in report.columns_scanned


def test_detection_redacts_samples_by_default():
    df = pd.DataFrame({"notes": ["ssn 123-45-6789"]})
    report = detect_pii(df)
    assert all(e.text == "<redacted>" for e in report.entities)
    # opt-in raw
    raw = detect_pii(df, config=PIIDetectionConfig(redact_samples=False))
    assert any("123-45-6789" in e.text for e in raw.entities)


def test_anonymize_redacts_by_default():
    df = pd.DataFrame({"email": ["a@b.com", "c@d.com"]})
    rule = MaskingRule(name="e", columns=("email",), strategy="redact", entity_types=("EMAIL",))
    out, report = anonymize(df, rules=(rule,))
    assert list(out["email"]) == ["***", "***"]
    assert report.cells_changed == 2
    assert "email" in report.columns_changed


def test_report_does_not_leak_raw_pii_by_default():
    df = pd.DataFrame({"email": ["secret@hidden.com"]})
    rule = MaskingRule(name="e", columns=("email",), strategy="hash", entity_types=("EMAIL",))
    out, report = anonymize(df, rules=(rule,))
    text = report.to_json()
    assert "secret@hidden.com" not in text
    assert report.events[0].original_preview == "<redacted>"


def test_audit_include_pii_shows_previews():
    df = pd.DataFrame({"email": ["secret@hidden.com"]})
    rule = MaskingRule(name="e", columns=("email",), strategy="redact", entity_types=("EMAIL",))
    _out, report = anonymize(df, rules=(rule,), audit_include_pii=True)
    assert "secret@hidden.com".startswith(report.events[0].original_preview[:6])


def test_hipaa_gdpr_tags_present():
    df = pd.DataFrame({"ssn": ["123-45-6789"]})
    rule = MaskingRule(name="s", columns=("ssn",), strategy="redact", entity_types=("SSN",))
    _out, report = anonymize(df, rules=(rule,))
    ev = report.events[0]
    assert ev.hipaa_tag == "HIPAA:ssn"
    assert ev.gdpr_tag == "GDPR:national_identifier"
    assert ev.risk_level == "high"


def test_tokenization_deterministic_and_reversible():
    vault = InMemoryTokenVault()
    t1 = tokenize_value("alice@x.com", vault, "KEY")
    t2 = tokenize_value("alice@x.com", vault, "KEY")
    assert t1 == t2 and t1.startswith("tok_")
    assert detokenize_value(t1, vault) == "alice@x.com"


def test_tokenization_requires_key():
    with pytest.raises(ValueError, match="key"):
        tokenize_value("x", InMemoryTokenVault(), "")


def test_detokenize_needs_the_vault():
    vault = InMemoryTokenVault()
    token = tokenize_value("bob@x.com", vault, "KEY")
    with pytest.raises(KeyError):
        detokenize_value(token, InMemoryTokenVault())  # fresh, empty vault


def test_tokenize_strategy_with_json_vault(tmp_path):
    vault_path = tmp_path / "vault.json"
    df = pd.DataFrame({"email": ["a@b.com", "a@b.com", "c@d.com"]})
    rule = MaskingRule(
        name="tok",
        columns=("email",),
        strategy="tokenize",
        reversible=True,
        key="SECRETKEY",
        token_vault_path=str(vault_path),
        entity_types=("EMAIL",),
    )
    out, _report = anonymize(df, rules=(rule,))
    tokens = list(out["email"])
    assert tokens[0] == tokens[1]  # deterministic
    assert tokens[0] != tokens[2]
    assert vault_path.exists()
    # reload vault and reverse
    vault = JsonTokenVault(vault_path)
    assert detokenize_value(tokens[0], vault) == "a@b.com"


def test_reversible_without_key_raises():
    df = pd.DataFrame({"email": ["a@b.com"]})
    rule = MaskingRule(name="t", columns=("email",), strategy="tokenize", reversible=True)
    with pytest.raises(ValueError, match="requires key"):
        anonymize(df, rules=(rule,))


def test_surrogate_preserves_shape():
    df = pd.DataFrame({"ssn": ["123-45-6789"]})
    rule = MaskingRule(
        name="s", columns=("ssn",), strategy="surrogate", preserve_format=True, visible=4
    )
    out, report = anonymize(df, rules=(rule,))
    masked = out["ssn"].iloc[0]
    assert masked != "123-45-6789"
    assert len(masked) == len("123-45-6789")
    assert masked[3] == "-" and masked[6] == "-"  # separators preserved
    assert masked.endswith("6789")  # last 4 preserved
    assert report.metadata["fpe_mode"] == "surrogate_format_preserving_not_crypto_fpe"


def test_fpe_falls_back_to_surrogate_without_crypto():
    df = pd.DataFrame({"acct": ["1234567890"]})
    rule = MaskingRule(name="f", columns=("acct",), strategy="fpe", key="K", preserve_format=True)
    out, report = anonymize(df, rules=(rule,))
    masked = out["acct"].iloc[0]
    assert len(masked) == 10 and masked.isdigit()
    assert report.metadata["fpe_mode"].startswith("surrogate_format_preserving")


def test_detection_driven_anonymization_scrubs_spans():
    df = pd.DataFrame({"notes": ["call me at 123-45-6789 today"]})
    out, report = anonymize(df, detection_config=PIIDetectionConfig())
    assert "123-45-6789" not in out["notes"].iloc[0]
    assert "<SSN>" in out["notes"].iloc[0]
    assert report.entities_found >= 1


def test_k_anonymity_detects_violations():
    df = pd.DataFrame(
        {
            "zip": ["900", "900", "900", "100", "100", "200"],
            "gender": ["F", "F", "F", "M", "M", "F"],
        }
    )
    report = check_k_anonymity(df, ["zip", "gender"], k=3)
    assert not report.ok
    assert report.smallest_class_size == 1
    assert report.rows_violating_k == 3  # (100,M)x2 + (200,F)x1
    assert report.high_risk_groups


def test_k_anonymity_passes_when_groups_large():
    df = pd.DataFrame({"a": ["x"] * 10, "b": ["y"] * 10})
    report = check_k_anonymity(df, ["a", "b"], k=5)
    assert report.ok
    assert report.n_equivalence_classes == 1


def test_k_anonymity_missing_column_raises():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError):
        check_k_anonymity(df, ["nope"], k=2)


def test_no_input_mutation_on_anonymize():
    df = pd.DataFrame({"email": ["a@b.com"]})
    before = df.copy()
    anonymize(df, rules=(MaskingRule(name="e", columns=("email",), strategy="hash"),))
    pd.testing.assert_frame_equal(df, before)


def test_public_api_exposed():
    assert fd.detect_pii is detect_pii
    assert fd.anonymize is anonymize
    assert fd.check_k_anonymity is check_k_anonymity


def test_polars_anonymize_returns_polars():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"email": ["a@b.com", "c@d.com"]})
    rule = MaskingRule(name="e", columns=("email",), strategy="redact")
    out = anonymize(df, rules=(rule,), return_report=False)
    assert isinstance(out, pl.DataFrame)
    assert out["email"].to_list() == ["***", "***"]
