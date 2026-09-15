"""Regression tests for text-quality fixes #316, #317 and #318."""

from __future__ import annotations

import unicodedata

import pandas as pd
import pytest

import freshdata as fd
from freshdata.fieldcheck import FieldSpec, _is_phone, detect_value_type
from freshdata.textclean import TextCleanConfig, clean_text_value
from freshdata.textlint import _script_of

ZWSP = "\u200b"
ACUTE = "\u0301"
JOSE_NFC = "José"

# ---------------------------------------------------------------------------
# #316: clean_text_value output is in the configured form and idempotent
# ---------------------------------------------------------------------------

_TRICKY = [
    "Jose\u200b\u0301",                 # the issue repro: ZWSP between e and acute
    "e\u200c\u0301 and a\u200d\u0308",  # ZWNJ / ZWJ before combining marks
    "\ufeffcafe\u0301",                 # BOM + decomposed accent
    "n\u2060\u0303",                    # word joiner before combining tilde
    "c\x07\u0327a",                     # control char before combining cedilla
    "a\u200e\u030a",                    # bidi mark before combining ring
    "café\u00a0 bar",              # NBSP
    "  Jos\u200bé \u200b",
    "ᄀ\u200bᅡ",               # Hangul jamo split by ZWSP
    "Ａ\u200b\u0301",               # fullwidth letter
    "“quote” — e\u200b\u0301…",
    "",
    "plain ascii",
]


@pytest.mark.parametrize("raw", _TRICKY)
def test_clean_text_value_is_idempotent_and_nfc(raw: str) -> None:
    once = clean_text_value(raw).cleaned
    again = clean_text_value(once)
    assert again.cleaned == once
    assert again.transforms == ()
    assert once == unicodedata.normalize("NFC", once)


@pytest.mark.parametrize("form", ["NFC", "NFD", "NFKC", "NFKD"])
@pytest.mark.parametrize("raw", _TRICKY)
def test_clean_text_value_idempotent_for_every_form(raw: str, form: str) -> None:
    cfg = TextCleanConfig(unicode_form=form)
    once = clean_text_value(raw, cfg).cleaned
    assert clean_text_value(once, cfg).cleaned == once
    assert once == unicodedata.normalize(form, once)


def test_issue_316_repro_composes_and_reports_form_once() -> None:
    result = fd.clean_text_value("Jose" + ZWSP + ACUTE)
    assert result.cleaned == JOSE_NFC
    assert result.transforms == ("strip_zero_width", "unicode_nfc")


def test_renormalization_does_not_duplicate_transform_name() -> None:
    result = clean_text_value("cafe" + ACUTE + " e" + ZWSP + ACUTE)
    assert result.transforms.count("unicode_nfc") == 1
    assert result.cleaned == "café é"


def test_unicode_form_none_leaves_combining_sequence_alone() -> None:
    cfg = TextCleanConfig(unicode_form=None)
    result = clean_text_value("Jose" + ZWSP + ACUTE, cfg)
    assert result.cleaned == "Jose" + ACUTE
    assert result.transforms == ("strip_zero_width",)


# ---------------------------------------------------------------------------
# #317: lint_text_encoding false positives on ordinary text
# ---------------------------------------------------------------------------


def test_issue_317_repro_reports_no_issues() -> None:
    df = pd.DataFrame({
        "ja": ["コーヒー"],   # coffee (katakana + prolonged sound mark)
        "ja2": ["人々"],              # people (kanji + iteration mark)
        "es": ["Nº 5"],
        "pt": ["MANHÃ DE SOL"],
    })
    assert fd.lint_text_encoding(df).issues == []


@pytest.mark.parametrize("value", [
    "コーヒー",
    "人々",
    "スーパー",       # super (katakana)
    "ゝゞ",                   # hiragana iteration marks
    "〆切",                   # ideographic closing mark + kanji
    "Nº 5",
    "1ª edição",
    "2º andar",
    "MANHÃ DE SOL",
    "SÃO PAULO",
    "IRMÃ E IRMÃO",
    "LEÃO",
    "Señor Muñoz",
])
def test_ordinary_text_is_not_flagged(value: str) -> None:
    rep = fd.lint_text_encoding(pd.DataFrame({"x": [value]}))
    assert not [i for i in rep.issues if i.issue_type in ("mixed_script", "mojibake")]


@pytest.mark.parametrize("value", [
    "Ã©cole",               # école
    "cafÃ¨",                # cafè
    "Ã¼ber",                # über
    "seÃ±or",               # señor
    "donâ€™t",         # don't
    "â€œquoted",       # "quoted
    "Â£5",                  # £5
    "SÃ£o Paulo",           # São Paulo
    "Ã\u00a0 la carte",          # à la carte (NBSP form)
])
def test_real_mojibake_is_still_detected(value: str) -> None:
    rep = fd.lint_text_encoding(pd.DataFrame({"x": [value]}))
    assert any(i.issue_type == "mojibake" and i.severity == "high" for i in rep.issues)


@pytest.mark.parametrize("value", [
    "Аlice",                     # Cyrillic A
    "Paypal with Cyrillic а",
    "Hello Ωmega",               # Greek Omega
])
def test_real_mixed_script_is_still_detected(value: str) -> None:
    rep = fd.lint_text_encoding(pd.DataFrame({"x": [value]}))
    assert any(i.issue_type == "mixed_script" for i in rep.issues)


@pytest.mark.parametrize("ch", ["ー", "々", "ª", "º", "ʰ"])
def test_script_neutral_letters(ch: str) -> None:
    assert _script_of(ch) is None


@pytest.mark.parametrize("ch", ["東", "は", "カ", "〆", "ヿ"])
def test_japanese_letters_share_one_script(ch: str) -> None:
    assert _script_of(ch) == "CJK_JP"


def test_latin_and_cyrillic_scripts_differ() -> None:
    assert _script_of("a") == "LATIN"
    assert _script_of("а") == "CYRILLIC"


# ---------------------------------------------------------------------------
# #318: international phone formats and punycode-TLD emails
# ---------------------------------------------------------------------------


def test_issue_318_repro_reports_no_issues() -> None:
    df = pd.DataFrame({
        "p": ["+49 (0) 30 12345678", "+44 (0) 20 7946 0958", "+44 20 7946 0958"],
        "e": ["user@example.xn--p1ai", "a+tag@example.com", "b@example.com"],
    })
    rep = fd.validate_fields(df, {"p": "phone", "e": "email"})
    assert rep.issues == []


@pytest.mark.parametrize("value", [
    "+49 (0) 30 12345678", "+44 (0) 20 7946 0958", "+1 (555) 010-9999",
    "555-0100 123", "+1 415 555 0101", "+86 (0) 10 1234 5678", "+353 (0)1 234 5678",
    "+1 234 567 890 12345",     # 15 digits: the E.164 maximum
])
def test_valid_phones_pass(value: str) -> None:
    assert _is_phone(value)
    assert detect_value_type(value) == "phone"
    rep = fd.validate_fields(pd.DataFrame({"p": [value]}),
                             {"p": FieldSpec(semantic_type="phone")})
    assert rep.issues == []


@pytest.mark.parametrize("value", [
    "12345",                    # too few digits
    "123-45",
    "+1234567890123456",        # 16 digits: over the E.164 maximum
    "+1 (555) 010-9999 ext 4",  # letters
    "(((((((((((())))))",       # no digits
    "+" + "1 " * 16,            # 16 digits spread out
    "(" * 31,                   # too long overall
])
def test_invalid_phones_rejected(value: str) -> None:
    assert not _is_phone(value)
    rep = fd.validate_fields(pd.DataFrame({"p": [value]}),
                             {"p": FieldSpec(semantic_type="phone")})
    assert [i.rule for i in rep.issues] == ["phone_format"]


def test_phone_checks_agree_across_a_mixed_column() -> None:
    values = ["+49 (0) 30 12345678", "12345", "+44 20 7946 0958", "+1234567890123456",
              "+33 (0)1 23 45 67 89", "hello world"]
    rep = fd.validate_fields(pd.DataFrame({"p": values}),
                             {"p": FieldSpec(semantic_type="phone")})
    assert sorted(i.row for i in rep.issues) == [1, 3, 5]


@pytest.mark.parametrize("value", [
    "user@example.xn--p1ai", "user@xn--80ak6aa92e.xn--p1ai", "a@b.co",
    "first.last+tag@sub.example.org", "x@example.xn--fiqs8s",
])
def test_valid_emails_pass(value: str) -> None:
    assert detect_value_type(value) == "email"
    rep = fd.validate_fields(pd.DataFrame({"e": [value]}),
                             {"e": FieldSpec(semantic_type="email")})
    assert rep.issues == []


@pytest.mark.parametrize("value", [
    "a@b.c", "a@b.xn--", "a@b.com1", "x@y", "no-at-sign.com", "a b@c.com",
    "a@b.xn--p1ai!",
])
def test_invalid_emails_rejected(value: str) -> None:
    rep = fd.validate_fields(pd.DataFrame({"e": [value]}),
                             {"e": FieldSpec(semantic_type="email")})
    assert [i.rule for i in rep.issues] == ["email_format"]
