"""Tests for the compliance-grade privacy *policy engine* (jurisdiction-aware).

Covers the four built-in compliance packs (HIPAA, FERPA, PCI, GDPR), jurisdiction
scoping, reversible tokenisation with pluggable vault backends, data minimisation,
the Data-Trust privacy dimension, and the redacted audit report.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from freshdata.enterprise import (
    Action,
    CompliancePack,
    Jurisdiction,
    PrivacyPolicy,
    PrivacyRule,
    SqliteTokenVault,
    apply_privacy_policy,
    available_packs,
    classify_columns,
    detokenize_series,
    load_compliance_pack,
    load_privacy_policy,
    make_vault,
)

KEY = "unit-test-key"


# --------------------------------------------------------------------------
# Pack loading
# --------------------------------------------------------------------------


def test_builtin_packs_load():
    assert set(available_packs()) == {"hipaa", "ferpa", "pci", "gdpr"}
    for name in available_packs():
        pack = load_compliance_pack(name)
        assert isinstance(pack, CompliancePack)
        assert pack.rules, f"{name} has no rules"
        assert pack.default_action in {a.value for a in Action}
        assert pack.allowed_actions


def test_unknown_pack_raises():
    with pytest.raises(ValueError):
        load_compliance_pack("nope")


# --------------------------------------------------------------------------
# HIPAA
# --------------------------------------------------------------------------


def _hipaa_frame():
    return pd.DataFrame(
        {
            "mrn": ["MRN 1234567", "MRN 7654321"],
            "dob": ["1980-01-02", "1975-12-30"],
            "patient_email": ["alice@example.com", "bob@example.org"],
            "city": ["Austin", "Reno"],
        }
    )


def test_hipaa_detects_and_protects_phi():
    df = _hipaa_frame()
    policy = PrivacyPolicy(
        name="clinic", jurisdiction="US", packs=(load_compliance_pack("hipaa"),), key=KEY
    )
    out, report = apply_privacy_policy(df, policy)

    # MRN tokenised (reversible), DOB pseudonymised, email redacted.
    assert all(v.startswith("tok_") for v in out["mrn"])
    assert list(out["dob"]) != list(df["dob"])  # pseudonymised in place
    assert all("@" not in v or v.startswith("<") for v in out["patient_email"])
    assert "hipaa" in report.compliance_pack
    assert report.jurisdiction == "US"

    td = report.trust_dimension
    assert td["sensitive_fields_detected"] >= 3
    assert td["sensitive_fields_touched"] >= 3
    assert td["policy_violations"] == 0
    # the report can prove which policy/action/reversibility applied to MRN
    assert report.trust_dimension["fields"]["mrn"]["reversible"] is True
    assert report.classifications["mrn"]["rule_id"] == "hipaa.mrn"


# --------------------------------------------------------------------------
# FERPA
# --------------------------------------------------------------------------


def test_ferpa_protects_education_record():
    df = pd.DataFrame(
        {
            "student_id": ["S1001", "S1002"],
            "guardian_email": ["mom@home.com", "dad@home.com"],
            "grade": ["A", "B"],
            "homeroom": ["101", "102"],
        }
    )
    policy = PrivacyPolicy(
        name="district", jurisdiction="US", packs=(load_compliance_pack("ferpa"),), key=KEY
    )
    out, report = apply_privacy_policy(df, policy)

    assert all(v.startswith("tok_") for v in out["student_id"])  # tokenised
    assert all("@" not in v for v in out["guardian_email"])  # redacted
    assert list(out["grade"]) != ["A", "B"]  # pseudonymised
    rule_ids = {c["rule_id"] for c in report.classifications.values()}
    assert {"ferpa.student_id", "ferpa.guardian_email", "ferpa.grade"} <= rule_ids


# --------------------------------------------------------------------------
# PCI
# --------------------------------------------------------------------------


def test_pci_card_number_luhn_and_tokenize():
    df = pd.DataFrame(
        {
            "card_number": ["4111111111111111", "4012888888881881"],  # valid Luhn
            "order_id": ["1001", "1002"],
        }
    )
    vault = make_vault("memory")
    policy = PrivacyPolicy(
        name="checkout", jurisdiction="Global", packs=(load_compliance_pack("pci"),), key=KEY
    )
    out, report = apply_privacy_policy(df, policy, vault=vault)

    assert all(v.startswith("tok_") for v in out["card_number"])
    assert list(out["order_id"]) == ["1001", "1002"]  # untouched
    assert "pci" in report.compliance_pack
    # round-trip only with the vault + key
    back = detokenize_series(out["card_number"], vault, KEY)
    assert list(back) == ["4111111111111111", "4012888888881881"]


def test_pci_ignores_non_luhn_numbers():
    # A 16-digit value that fails the Luhn check is not treated as a PAN.
    # valid cards with the final check digit flipped -> fail Luhn
    bad = ["4111111111111110", "4012888888881882"]
    df = pd.DataFrame({"card_number": list(bad)})
    policy = PrivacyPolicy(
        name="c", jurisdiction="Global", packs=(load_compliance_pack("pci"),), key=KEY
    )
    out, _ = apply_privacy_policy(df, policy)
    # column-name still matches pci.pan, but the requires_luhn gate stops the
    # match; with no Luhn-valid candidate the rule does not claim the column.
    assert list(out["card_number"]) == bad


# --------------------------------------------------------------------------
# GDPR + minimisation
# --------------------------------------------------------------------------


def test_gdpr_minimization_drops_unnecessary_columns_when_configured():
    df = pd.DataFrame(
        {
            "email": ["x@a.com", "y@b.com"],
            "marketing_preferences": ["weekly", "never"],
            "country": ["DE", "FR"],
        }
    )
    pack = load_compliance_pack("gdpr")
    # minimisation ON -> the marketing column is dropped
    on = PrivacyPolicy(name="eu", jurisdiction="EU", packs=(pack,), minimize=True, key=KEY)
    out_on, rep_on = apply_privacy_policy(df, on)
    assert "marketing_preferences" not in out_on.columns
    assert "marketing_preferences" in rep_on.metadata["dropped_columns"]

    # minimisation OFF -> the column is kept but flagged unprotected
    off = PrivacyPolicy(name="eu", jurisdiction="EU", packs=(pack,), minimize=False, key=KEY)
    out_off, rep_off = apply_privacy_policy(df, off)
    assert "marketing_preferences" in out_off.columns
    assert "marketing_preferences" in rep_off.trust_dimension["unprotected_sensitive_fields"]


# --------------------------------------------------------------------------
# Jurisdiction changes the outcome
# --------------------------------------------------------------------------


def test_jurisdiction_changes_rule_outcome():
    df = pd.DataFrame({"email": ["a@x.com", "b@y.com"]})
    pack = load_compliance_pack("gdpr")  # EU-scoped rules
    eu = PrivacyPolicy(name="eu", jurisdiction="EU", packs=(pack,), key=KEY)
    us = PrivacyPolicy(name="us", jurisdiction="US", packs=(pack,), key=KEY)

    out_eu, _ = apply_privacy_policy(df, eu)
    out_us, _ = apply_privacy_policy(df, us)

    assert list(out_eu["email"]) != list(df["email"])  # GDPR applies under EU
    assert list(out_us["email"]) == list(df["email"])  # not under US


def test_uk_inherits_eu_rules():
    df = pd.DataFrame({"email": ["a@x.com"]})
    pack = load_compliance_pack("gdpr")
    uk = PrivacyPolicy(name="uk", jurisdiction="UK", packs=(pack,), key=KEY)
    out, _ = apply_privacy_policy(df, uk)
    assert list(out["email"]) != list(df["email"])  # UK GDPR mirrors EU


def test_jurisdiction_override_argument():
    df = pd.DataFrame({"email": ["a@x.com"]})
    policy = PrivacyPolicy(
        name="p", jurisdiction="EU", packs=(load_compliance_pack("gdpr"),), key=KEY
    )
    out, rep = apply_privacy_policy(df, policy, jurisdiction="US")
    assert list(out["email"]) == ["a@x.com"]
    assert rep.jurisdiction == "US"


# --------------------------------------------------------------------------
# Reversible tokenisation requires vault + key
# --------------------------------------------------------------------------


def test_reversible_tokenize_requires_key():
    df = pd.DataFrame({"ssn": ["123-45-6789"]})
    rule = PrivacyRule(id="r", action="tokenize", reversible=True, columns=("ssn",))
    policy = PrivacyPolicy(name="p", rules=(rule,))  # no key anywhere
    with pytest.raises(ValueError):
        apply_privacy_policy(df, policy)


def test_detokenize_requires_key():
    s = pd.Series(["tok_abc"])
    with pytest.raises(ValueError):
        detokenize_series(s, make_vault("memory"), "")


def test_sqlite_vault_round_trip(tmp_path):
    df = pd.DataFrame({"ssn": ["123-45-6789", "987-65-4321"]})
    vault = SqliteTokenVault(tmp_path / "vault.db")
    rule = PrivacyRule(id="ssn", action="tokenize", reversible=True, columns=("ssn",))
    policy = PrivacyPolicy(name="p", rules=(rule,), key=KEY)
    out, report = apply_privacy_policy(df, policy, vault=vault)
    assert all(v.startswith("tok_") for v in out["ssn"])
    assert report.vault_info["backend"] == "sqlite"
    assert report.vault_info["entries"] == 2
    back = detokenize_series(out["ssn"], vault, KEY)
    assert list(back) == ["123-45-6789", "987-65-4321"]


# --------------------------------------------------------------------------
# Audit report never leaks raw PII by default
# --------------------------------------------------------------------------


def test_report_does_not_leak_raw_pii_by_default():
    df = _hipaa_frame()
    policy = PrivacyPolicy(
        name="c", jurisdiction="US", packs=(load_compliance_pack("hipaa"),), key=KEY
    )
    _, report = apply_privacy_policy(df, policy)

    blob = report.to_json()
    assert "alice@example.com" not in blob
    assert "MRN 1234567" not in blob
    assert all(e.original_preview == "<redacted>" for e in report.events)

    frame = report.to_frame()
    assert (frame["original_preview"] == "<redacted>").all()
    # to_json round-trips to valid JSON
    assert isinstance(json.loads(blob), dict)


def test_audit_include_pii_opt_in_shows_previews():
    df = _hipaa_frame()
    policy = PrivacyPolicy(
        name="c", jurisdiction="US", packs=(load_compliance_pack("hipaa"),), key=KEY
    )
    _, report = apply_privacy_policy(df, policy, audit_include_pii=True)
    previews = [e.original_preview for e in report.events if e.original_preview != "<redacted>"]
    assert previews  # at least one real preview shown when explicitly opted in


# --------------------------------------------------------------------------
# Violations & conformance
# --------------------------------------------------------------------------


def test_preserve_without_reason_is_a_violation():
    df = pd.DataFrame({"email": ["a@x.com"]})
    rule = PrivacyRule(id="keep", action="preserve_with_reason", columns=("email",),
                       classification="contact")
    policy = PrivacyPolicy(name="p", rules=(rule,))
    _, report = apply_privacy_policy(df, policy)
    assert any(v["type"] == "preserve_without_reason" for v in report.violations)


def test_preserve_with_reason_is_documented_not_violation():
    df = pd.DataFrame({"email": ["a@x.com"]})
    rule = PrivacyRule(id="keep", action="preserve_with_reason", columns=("email",),
                       classification="contact", legal_basis="Art. 6(1)(a) consent")
    policy = PrivacyPolicy(name="p", rules=(rule,))
    out, report = apply_privacy_policy(df, policy)
    assert list(out["email"]) == ["a@x.com"]  # preserved unchanged
    assert not report.violations
    assert "email" in report.trust_dimension["documented_raw_fields"]


def test_quarantine_action():
    df = pd.DataFrame({"secret": ["x", "y"]})
    rule = PrivacyRule(id="q", action="quarantine", columns=("secret",), classification="s")
    policy = PrivacyPolicy(name="p", rules=(rule,))
    out, report = apply_privacy_policy(df, policy)
    assert list(out["secret"]) == ["<QUARANTINED>", "<QUARANTINED>"]
    assert "secret" in report.metadata["quarantined_columns"]


# --------------------------------------------------------------------------
# Policy (de)serialisation
# --------------------------------------------------------------------------


def test_load_policy_from_json(tmp_path):
    spec = {
        "name": "my-policy",
        "jurisdiction": "EU",
        "minimize": True,
        "packs": ["gdpr"],
        "rules": [
            {"id": "custom.badge", "action": "redact", "column_patterns": ["(?i)badge"]}
        ],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    policy = load_privacy_policy(path)
    assert policy.name == "my-policy"
    assert policy.jurisdiction == "EU"
    assert any(p.name == "gdpr" for p in policy.packs)

    df = pd.DataFrame({"badge": ["B-1", "B-2"], "email": ["a@x.com", "b@y.com"]})
    out, _ = apply_privacy_policy(df, policy)
    assert list(out["badge"]) == ["<REDACTED>", "<REDACTED>"]  # inline rule applied


def test_load_policy_from_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    spec = {"name": "y", "jurisdiction": "US", "packs": ["hipaa"], "key": KEY}
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    policy = load_privacy_policy(path)
    assert policy.jurisdiction == "US"
    assert policy.packs[0].name == "hipaa"


# --------------------------------------------------------------------------
# Classification (read-only)
# --------------------------------------------------------------------------


def test_classify_columns_is_read_only():
    df = _hipaa_frame()
    before = df.copy()
    policy = PrivacyPolicy(name="c", jurisdiction="US", packs=(load_compliance_pack("hipaa"),))
    cls = classify_columns(df, policy)
    assert "mrn" in cls and cls["mrn"].rule.id == "hipaa.mrn"
    pd.testing.assert_frame_equal(df, before)  # unchanged


def test_jurisdiction_and_action_enums_coerce():
    assert Jurisdiction.coerce("eu") is Jurisdiction.EU
    assert Action.coerce("REDACT") is Action.REDACT
    with pytest.raises(ValueError):
        Jurisdiction.coerce("mars")
    with pytest.raises(ValueError):
        Action.coerce("nuke")
