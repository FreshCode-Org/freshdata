"""Keyless tokenize / surrogate / fpe / pseudonymize never use constants from the source.

Without a key, each call uses a random per-call key and emits EphemeralKeyWarning,
so output cannot be recomputed for guessed values. Also #281 part 3: crypto FPE
honours ``visible``.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import sys
import types
import warnings

import pandas as pd
import pytest

from freshdata.enterprise import (
    EphemeralKeyWarning,
    InMemoryTokenVault,
    MaskingRule,
    PrivacyPolicy,
    PrivacyRule,
    anonymize,
    apply_privacy_policy,
    load_compliance_pack,
    privacy,
    privacy_policy,
)

KEY = "unit-test-key"
KEY_ENV = "FRESHDATA_TEST_KEYLESS_KEY"
SURROGATE_MODE = "surrogate_format_preserving_not_crypto_fpe"
LEGACY_TOKEN_SALT = "freshdata-default-token-salt"
LEGACY_SURROGATE_KEY = "freshdata-surrogate"


def _hmac_hex(key: str, value: str, n: int = 16) -> str:
    return hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()[:n]


def _legacy_surrogate(s: str) -> str:
    """The attacker's re-implementation of the pre-2.1.0 keyless surrogate."""
    d = hmac.new(LEGACY_SURROGATE_KEY.encode(), s.encode(), hashlib.sha256).digest()
    return "".join(
        str(d[i % 32] % 10)
        if ch.isdigit()
        else (
            chr((65 if ch.isupper() else 97) + d[i % 32] % 26)
            if ch.isascii() and ch.isalpha()
            else ch
        )
        for i, ch in enumerate(s)
    )


def _call_warned(call, *args, **kwargs):
    """Run *call*, requiring exactly one EphemeralKeyWarning; return (result, message)."""
    with pytest.warns(EphemeralKeyWarning) as record:
        result = call(*args, **kwargs)
    ephemeral = [w for w in record if issubclass(w.category, EphemeralKeyWarning)]
    assert len(ephemeral) == 1
    return result, str(ephemeral[0].message)


def _call_silent(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", EphemeralKeyWarning)
        return call(*args, **kwargs)


@pytest.fixture
def no_pyffx(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyffx", None)


@pytest.fixture
def fake_pyffx(monkeypatch):
    """A stand-in ``pyffx`` recording the digit length of every cipher it builds."""
    lengths: list[int] = []
    module = types.ModuleType("pyffx")

    class Integer:
        def __init__(self, key, length):
            self.length = length
            lengths.append(length)

        def encrypt(self, n):
            return (n * 7 + 3) % (10**self.length)

    module.Integer = Integer
    monkeypatch.setitem(sys.modules, "pyffx", module)
    return lengths


# --------------------------------------------------------------------------
# Advisory PoC
# --------------------------------------------------------------------------


def test_poc_keyless_output_is_not_recoverable_from_public_constants(no_pyffx):
    (tok, _), _ = _call_warned(
        anonymize,
        pd.DataFrame({"ssn": ["123-45-6789"]}),
        rules=(MaskingRule(name="ssn", columns=("ssn",), strategy="tokenize"),),
    )
    (ps, rep), message = _call_warned(
        apply_privacy_policy,
        pd.DataFrame({"phone": ["+1 555 123 4567"]}),
        PrivacyPolicy(packs=(load_compliance_pack("gdpr"),), jurisdiction="EU"),
    )
    assert rep.classifications["phone"]["rule_id"] == "gdpr.phone"
    assert "gdpr.phone" in message

    k = _hmac_hex(LEGACY_TOKEN_SALT, "ssn", 32)
    token = tok["ssn"][0]
    assert token.startswith("tok_")
    ssn_guesses = (f"123-45-{i:04d}" for i in range(10000))
    assert [c for c in ssn_guesses if "tok_" + _hmac_hex(k, c) == token] == []
    pseudonym = ps["phone"][0]
    assert pseudonym != "+1 555 123 4567"
    phone_guesses = (f"+1 555 123 {i:04d}" for i in range(10000))
    assert [c for c in phone_guesses if _legacy_surrogate(c) == pseudonym] == []


# --------------------------------------------------------------------------
# anonymize: random per-call key
# --------------------------------------------------------------------------

_VALUES = ["123-45-6789", "123-45-6789", "987-65-4321"]


@pytest.mark.parametrize("strategy", ["tokenize", "surrogate", "fpe"])
def test_keyless_rule_is_consistent_within_a_call_and_differs_across_calls(strategy, no_pyffx):
    df = pd.DataFrame({"c": _VALUES})
    rule = MaskingRule(name="r", columns=("c",), strategy=strategy)
    (first, report), message = _call_warned(anonymize, df, rules=(rule,))
    (second, _), _ = _call_warned(anonymize, df, rules=(rule,))
    assert first["c"][0] == first["c"][1]
    assert first["c"][0] != first["c"][2]
    assert first["c"][0] != second["c"][0]
    assert all(v not in _VALUES for v in first["c"])
    assert report.metadata["ephemeral_key_rules"] == ["r"]
    assert "['r']" in message and "random per-run key" in message
    assert "key=/key_env=" in message


def test_one_warning_per_call_names_every_keyless_rule(no_pyffx):
    df = pd.DataFrame({"a": ["x1"], "b": ["y2"], "c": ["z3"], "d": ["w4"], "e": ["v5"]})
    rules = (
        MaskingRule(name="tok_a", columns=("a",), strategy="tokenize"),
        MaskingRule(name="sur_b", columns=("b",), strategy="surrogate"),
        MaskingRule(name="fpe_c", columns=("c",), strategy="fpe"),
        MaskingRule(name="keyed_d", columns=("d",), strategy="tokenize", key=KEY),
        MaskingRule(name="hash_e", columns=("e",), strategy="hash"),
        MaskingRule(name="missing", columns=("nope",), strategy="surrogate"),
    )
    (_, report), message = _call_warned(anonymize, df, rules=rules)
    assert report.metadata["ephemeral_key_rules"] == ["tok_a", "sur_b", "fpe_c"]
    assert "keyed_d" not in message and "hash_e" not in message and "missing" not in message
    assert report.to_json()  # serialisable, and no key material recorded
    assert KEY not in report.to_json()


@pytest.mark.parametrize("strategy", ["tokenize", "surrogate", "fpe"])
def test_explicit_key_or_key_env_is_stable_and_silent(strategy, monkeypatch, no_pyffx):
    monkeypatch.setenv(KEY_ENV, KEY)
    df = pd.DataFrame({"c": _VALUES})
    by_key = MaskingRule(name="r", columns=("c",), strategy=strategy, key=KEY)
    by_env = MaskingRule(name="r", columns=("c",), strategy=strategy, key_env=KEY_ENV)
    runs = [_call_silent(anonymize, df, rules=(rule,)) for rule in (by_key, by_key, by_env)]
    outputs = [list(out["c"]) for out, _ in runs]
    assert outputs[0] == outputs[1] == outputs[2]
    assert all("ephemeral_key_rules" not in report.metadata for _, report in runs)


def test_unset_key_env_uses_a_per_call_key(monkeypatch, no_pyffx):
    monkeypatch.delenv(KEY_ENV, raising=False)
    rule = MaskingRule(name="r", columns=("c",), strategy="surrogate", key_env=KEY_ENV)
    (_, report), _ = _call_warned(anonymize, pd.DataFrame({"c": ["555-0100"]}), rules=(rule,))
    assert report.metadata["ephemeral_key_rules"] == ["r"]


@pytest.mark.parametrize("strategy", ["tokenize", "fpe"])
def test_reversible_keyless_rule_still_raises(strategy):
    rule = MaskingRule(name="r", columns=("c",), strategy=strategy, reversible=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", EphemeralKeyWarning)
        with pytest.raises(ValueError, match="requires key"):
            anonymize(pd.DataFrame({"c": ["123-45-6789"]}), rules=(rule,))


def test_keyless_fpe_with_crypto_uses_the_per_call_key(fake_pyffx):
    rule = MaskingRule(name="r", columns=("c",), strategy="fpe")
    (out, report), _ = _call_warned(anonymize, pd.DataFrame({"c": ["123-45-6789"]}), rules=(rule,))
    assert report.metadata["fpe_mode"] == "crypto_fpe"
    assert report.metadata["ephemeral_key_rules"] == ["r"]
    assert [e.reversible for e in report.events] == [False]
    assert out["c"][0] == "864-19-7526"  # the stub ignores the key


def test_no_keyless_code_path_or_constant_remains():
    with pytest.raises(ValueError, match="non-empty key"):
        privacy._surrogate_value("123-45-6789", "")
    with pytest.raises(ValueError, match="non-empty key"):
        privacy._surrogate_value("123-45-6789", None)  # type: ignore[arg-type]
    for strategy in ("tokenize", "surrogate", "fpe"):
        rule = MaskingRule(name="r", columns=("c",), strategy=strategy)
        with pytest.raises(ValueError, match="requires a key"):
            privacy._mask_one("x", rule, None, InMemoryTokenVault())
    for module in (privacy, privacy_policy):
        source = inspect.getsource(module)
        assert LEGACY_TOKEN_SALT not in source
        assert LEGACY_SURROGATE_KEY not in source


# --------------------------------------------------------------------------
# apply_privacy_policy: keyless pseudonymize
# --------------------------------------------------------------------------


def _pseudonymize_policy(**kwargs) -> PrivacyPolicy:
    rules = (
        PrivacyRule(id="pa", action="pseudonymize", columns=("a",)),
        PrivacyRule(id="pb", action="pseudonymize", columns=("b",)),
    )
    return PrivacyPolicy(name="p", rules=rules, **kwargs)


def test_policy_keyless_pseudonymize_uses_one_random_key_per_call(no_pyffx):
    df = pd.DataFrame({"a": ["555-0100-2233"], "b": ["555-0100-2233"]})
    (first, report), message = _call_warned(apply_privacy_policy, df, _pseudonymize_policy())
    (second, _), _ = _call_warned(apply_privacy_policy, df, _pseudonymize_policy())
    assert first["a"][0] == first["b"][0]  # one key for the whole call
    assert first["a"][0] != second["a"][0]
    assert first["a"][0] != "555-0100-2233"
    assert first["a"][0] != _legacy_surrogate("555-0100-2233")
    assert report.metadata["ephemeral_key_rules"] == ["pa", "pb"]
    assert "['pa', 'pb']" in message


def test_policy_keyed_pseudonymize_is_stable_and_silent(monkeypatch, no_pyffx):
    monkeypatch.setenv(KEY_ENV, KEY)
    df = pd.DataFrame({"a": ["555-0100-2233"], "b": ["x"]})
    out1, rep1 = _call_silent(apply_privacy_policy, df, _pseudonymize_policy(key=KEY))
    out2, rep2 = _call_silent(apply_privacy_policy, df, _pseudonymize_policy(key_env=KEY_ENV))
    assert out1["a"][0] == out2["a"][0]
    assert "ephemeral_key_rules" not in rep1.metadata
    assert "ephemeral_key_rules" not in rep2.metadata


def test_policy_keyless_tokenize_still_raises():
    rule = PrivacyRule(id="t", action="tokenize", columns=("ssn",))
    with pytest.raises(ValueError, match="tokenize requires a key"):
        apply_privacy_policy(pd.DataFrame({"ssn": ["123-45-6789"]}), PrivacyPolicy(rules=(rule,)))


@pytest.mark.parametrize(
    ("pack", "jurisdiction", "column", "value", "rule_id"),
    [
        ("gdpr", "EU", "email", "jane.roe@example.com", "gdpr.email"),
        ("gdpr", "EU", "phone", "+1 555 123 4567", "gdpr.phone"),
        ("hipaa", "US", "dob", "1980-01-02", "hipaa.dob"),
        ("ferpa", "US", "grade", "Midterm 93.5 of 100", "ferpa.grade"),
    ],
)
def test_compliance_packs_without_key_do_not_use_the_public_constant(
    pack, jurisdiction, column, value, rule_id, no_pyffx
):
    policy = PrivacyPolicy(packs=(load_compliance_pack(pack),), jurisdiction=jurisdiction)
    df = pd.DataFrame({column: [value]})
    (out, report), message = _call_warned(apply_privacy_policy, df, policy)
    assert report.classifications[column]["rule_id"] == rule_id
    assert report.metadata["ephemeral_key_rules"] == [rule_id]
    assert rule_id in message
    assert out[column][0] not in (value, _legacy_surrogate(value))


# --------------------------------------------------------------------------
# Documented migration recipes reproduce the pre-2.1.0 output
# --------------------------------------------------------------------------


def test_legacy_surrogate_matches_the_advisory_output():
    assert _legacy_surrogate("+1 555 123 4567") == "+1 646 836 3000"


def test_migration_recipes_reproduce_legacy_output(no_pyffx):
    ssn = "123-45-6789"
    df = pd.DataFrame({"ssn": [ssn]})

    legacy_token_key = hmac.new(
        LEGACY_TOKEN_SALT.encode(), b"ssn", hashlib.sha256
    ).hexdigest()[:32]
    tok_rule = MaskingRule(name="ssn", columns=("ssn",), strategy="tokenize", key=legacy_token_key)
    out, _ = _call_silent(anonymize, df, rules=(tok_rule,))
    assert out["ssn"][0] == "tok_" + _hmac_hex(_hmac_hex(LEGACY_TOKEN_SALT, "ssn", 32), ssn)

    sur_rule = MaskingRule(
        name="s", columns=("ssn",), strategy="surrogate", key=LEGACY_SURROGATE_KEY
    )
    out, _ = _call_silent(anonymize, df, rules=(sur_rule,))
    assert out["ssn"][0] == _legacy_surrogate(ssn)

    fpe_rule = MaskingRule(
        name="f", columns=("ssn",), strategy="fpe", key=LEGACY_SURROGATE_KEY,
        preserve_format=True, visible=4,
    )
    out, _ = _call_silent(anonymize, df, rules=(fpe_rule,))
    assert out["ssn"][0] == _legacy_surrogate(ssn)[:7] + ssn[7:]

    policy = PrivacyPolicy(
        packs=(load_compliance_pack("gdpr"),), jurisdiction="EU", key=LEGACY_SURROGATE_KEY
    )
    phone = pd.DataFrame({"phone": ["+1 555 123 4567"]})
    out, _ = _call_silent(apply_privacy_policy, phone, policy)
    assert out["phone"][0] == "+1 646 836 3000"


# --------------------------------------------------------------------------
# #281 part 3: crypto FPE honours visible
# --------------------------------------------------------------------------

_CARD = "4111-1111-1111-1111"


def _fpe_rule(**kwargs) -> MaskingRule:
    return MaskingRule(name="c", columns=("cc",), strategy="fpe", key=KEY, **kwargs)


def test_issue_281_crypto_fpe_keeps_last_visible_characters(fake_pyffx):
    rule = _fpe_rule(preserve_format=True, visible=4)
    out, report = _call_silent(anonymize, pd.DataFrame({"cc": [_CARD]}), rules=(rule,))
    assert out["cc"][0] == "8777-7777-7780-1111"
    assert fake_pyffx == [12]  # only the 12 digits before the visible tail
    assert report.metadata == {"fpe_mode": "crypto_fpe"}


def test_fpe_value_splits_head_and_tail(fake_pyffx):
    assert privacy._fpe_value("123-45-6789", KEY, visible=4) == ("864-18-6789", "crypto_fpe")
    assert fake_pyffx == [5]


@pytest.mark.parametrize("visible", [19, 30])
def test_visible_at_least_length_encrypts_every_digit(fake_pyffx, visible):
    rule = _fpe_rule(preserve_format=True, visible=visible)
    out, _ = _call_silent(anonymize, pd.DataFrame({"cc": [_CARD]}), rules=(rule,))
    assert out["cc"][0] == "8777-7777-7777-7780"
    assert fake_pyffx == [16]


def test_preserve_format_false_ignores_visible(fake_pyffx):
    rule = _fpe_rule(preserve_format=False, visible=4)
    out, _ = _call_silent(anonymize, pd.DataFrame({"cc": [_CARD]}), rules=(rule,))
    assert out["cc"][0] == "8777-7777-7777-7780"
    assert fake_pyffx == [16]


def test_head_without_digits_falls_back_to_surrogate(fake_pyffx):
    rule = _fpe_rule(preserve_format=True, visible=4, reversible=True)
    out, report = _call_silent(anonymize, pd.DataFrame({"cc": ["abcd-1234"]}), rules=(rule,))
    assert out["cc"][0] == privacy._surrogate_value("abcd-1234", KEY, visible=4)
    assert out["cc"][0].endswith("-1234")
    assert fake_pyffx == []
    assert report.metadata == {"fpe_mode": SURROGATE_MODE}
    assert [e.reversible for e in report.events] == [False]


def test_mode_and_reversible_follow_the_mode_actually_used(fake_pyffx):
    rule = _fpe_rule(preserve_format=True, visible=4, reversible=True)
    df = pd.DataFrame({"cc": [_CARD, "abcd-1234"]})
    out, report = _call_silent(anonymize, df, rules=(rule,))
    assert out["cc"][0] == "8777-7777-7780-1111"
    assert report.metadata["fpe_mode"] == "mixed"
    assert report.metadata["fpe_modes"] == {"cc": {"crypto_fpe": 1, SURROGATE_MODE: 1}}
    assert {e.row: e.reversible for e in report.events} == {0: True, 1: False}
