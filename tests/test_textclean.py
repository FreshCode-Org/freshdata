"""Tests for the standalone field-aware text-cleaning pipeline."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import freshdata as fd
from freshdata.textclean import (
    CleanedText,
    TextCleanConfig,
    clean_text,
    clean_text_value,
    config_for_field,
)

# ---------------------------------------------------------------------------
# scalar pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "config", "expected", "expected_transforms"),
    [
        ("  hello   world  ", None, "hello world", ("collapse_whitespace", "strip")),
        ("café", None, "café", ("unicode_nfc",)),
        ("ab\x00c\x07d", None, "abcd", ("strip_control_chars",)),
        ("a\u200bb‌c", None, "abc", ("strip_zero_width",)),
        ("“quoted” — dash…", None, '"quoted" - dash...',
         ("normalize_punctuation",)),
        ("a b", None, "a b", ("normalize_punctuation",)),
        ("<b>Bold</b> &amp; plain", TextCleanConfig(strip_html=True), "Bold & plain",
         ("strip_html", "collapse_whitespace")),
        ("see https://x.co/page now", TextCleanConfig(strip_urls=True), "see now",
         ("strip_urls", "collapse_whitespace")),
        ("LOUD text", TextCleanConfig(case="lower"), "loud text", ("case_lower",)),
        ("sooooo good", TextCleanConfig(max_char_repeat=2), "soo good",
         ("collapse_repeats",)),
        ("abcdef", TextCleanConfig(max_length=3), "abc", ("truncate",)),
    ],
)
def test_scalar_transforms(raw, config, expected, expected_transforms):
    result = clean_text_value(raw, config)
    assert result.cleaned == expected
    assert result.original == raw
    assert result.transforms == expected_transforms
    assert result.changed


def test_clean_value_is_deterministic_and_idempotent():
    raw = "  <p>Nice&nbsp;product’s page</p>  "
    cfg = TextCleanConfig(strip_html=True)
    first = clean_text_value(raw, cfg)
    second = clean_text_value(raw, cfg)
    assert first == second
    again = clean_text_value(first.cleaned, cfg)
    assert again.cleaned == first.cleaned


@pytest.mark.parametrize("value", [None, 12.5, 7, True, float("nan")])
def test_non_strings_pass_through_untouched(value):
    result = clean_text_value(value)
    if isinstance(value, float) and math.isnan(value):
        assert math.isnan(result.cleaned)
    else:
        assert result.cleaned is value
    assert result.transforms == ()
    assert not result.changed


def test_unchanged_string_reports_no_transforms():
    result = clean_text_value("already clean")
    assert result.cleaned == "already clean"
    assert not result.changed


def test_script_and_style_content_dropped():
    raw = "<div>ok<script>alert('x')</script></div>"
    result = clean_text_value(raw, TextCleanConfig(strip_html=True))
    assert "alert" not in result.cleaned
    assert result.cleaned == "ok"


def test_remove_punctuation_and_custom_op():
    cfg = TextCleanConfig(
        remove_punctuation=True,
        custom=(("mask_digits", lambda s: "".join("#" if c.isdigit() else c for c in s)),),
    )
    result = clean_text_value("a.b,c 12", cfg)
    assert result.cleaned == "abc ##"
    assert "remove_punctuation" in result.transforms
    assert "custom_mask_digits" in result.transforms


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"unicode_form": "NFX"}, "unicode_form"), ({"case": "sponge"}, "case")],
)
def test_invalid_config_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TextCleanConfig(**kwargs)


# ---------------------------------------------------------------------------
# field-type awareness: aggressive ops withheld from structural values
# ---------------------------------------------------------------------------

AGGRESSIVE = TextCleanConfig(
    strip_html=True, strip_urls=True, case="lower", remove_punctuation=True,
    max_char_repeat=2,
)


@pytest.mark.parametrize(
    ("field_type", "value"),
    [
        ("currency_amount", "1,200.50"),
        ("rate", "0.05"),
        ("identifier", "AC-0007/2026"),
        ("account_number", "0012-3456-789"),
        ("email", "Jo.Ann+tag@Example.COM"),
        ("url", "https://example.com/A?b=1"),
        ("date_like", "2026-01-15"),
        ("ticker", "BRK.B"),
        ("phone", "+1 (555) 010-9999"),
    ],
)
def test_structural_fields_never_corrupted(field_type, value):
    result = clean_text_value(value, AGGRESSIVE, field_type=field_type)
    assert result.cleaned == value, (
        f"{field_type} value {value!r} was corrupted to {result.cleaned!r}"
    )


def test_entity_fields_keep_punctuation_and_case():
    cfg = config_for_field("company_name", AGGRESSIVE)
    assert cfg.remove_punctuation is False
    assert cfg.case is None  # "lower" is unsafe for names; only "title" survives
    result = clean_text_value("O'Neil & Sons, Ltd.", AGGRESSIVE, field_type="company_name")
    assert result.cleaned == "O'Neil & Sons, Ltd."


def test_entity_fields_allow_title_case():
    cfg = TextCleanConfig(case="title")
    result = clean_text_value("acme corp", cfg, field_type="company_name")
    assert result.cleaned == "Acme Corp"


def test_free_text_gets_full_pipeline():
    result = clean_text_value(
        "<p>GREAT   product!!!</p>", AGGRESSIVE, field_type="free_text")
    assert result.cleaned == "great product"  # html gone, lowered, punctuation removed


# ---------------------------------------------------------------------------
# frame-level cleaning
# ---------------------------------------------------------------------------


def _frame():
    return pd.DataFrame({
        "note": ["  spaced  ", "fine", "a\u200bb"],
        "amount": ["1,200.50", "99.99", "45.00"],
        "n": [1, 2, 3],
    })


def test_clean_text_returns_copy_and_report():
    df = _frame()
    snapshot = df.copy()
    cleaned, report = clean_text(df)
    pd.testing.assert_frame_equal(df, snapshot)  # input untouched
    assert cleaned.loc[0, "note"] == "spaced"
    assert cleaned.loc[2, "note"] == "ab"
    assert report.values_seen == 6  # two object columns, three rows each
    changed = {(c["row"], c["column"]) for c in report.changes}
    assert changed == {(0, "note"), (2, "note")}
    first = report.changes[0]
    assert first["original"] == "  spaced  "
    assert first["cleaned"] == "spaced"
    assert first["transforms"] == ["collapse_whitespace", "strip"]


def test_clean_text_field_types_guard_columns():
    df = pd.DataFrame({"amount": [" 1,200.50 "], "note": [" hi  there "]})
    cleaned, report = clean_text(
        df,
        config=TextCleanConfig(remove_punctuation=True),
        field_types={"amount": "currency_amount"},
    )
    assert cleaned.loc[0, "amount"] == "1,200.50"  # only whitespace trimmed
    assert cleaned.loc[0, "note"] == "hi there"


def test_clean_text_column_selection_and_errors():
    df = _frame()
    cleaned, report = clean_text(df, columns=["amount"])
    assert cleaned.loc[0, "note"] == "  spaced  "  # untouched
    with pytest.raises(KeyError, match="nope"):
        clean_text(df, columns=["nope"])


def test_clean_text_empty_and_numeric_frames():
    empty, report = clean_text(pd.DataFrame())
    assert empty.empty and len(report) == 0
    nums = pd.DataFrame({"n": [1, 2]})
    out, report = clean_text(nums)
    pd.testing.assert_frame_equal(out, nums)
    assert len(report) == 0


def test_report_summary_and_frame():
    _, report = clean_text(_frame())
    frame = report.to_frame()
    assert list(frame.columns) == ["row", "column", "original", "cleaned", "transforms"]
    assert "strip" in report.transform_counts()
    assert "value(s) changed" in report.summary()


# ---------------------------------------------------------------------------
# adversarial input
# ---------------------------------------------------------------------------


def test_homoglyphs_survive_default_cleaning():
    # Cyrillic А (U+0410): default cleaning must not silently latinize it —
    # detection is the validator's job, not the cleaner's.
    raw = "АPPLE"
    result = clean_text_value(raw)
    assert result.cleaned == raw


def test_bidi_override_and_whitespace_only():
    result = clean_text_value("abc\u202edef")
    assert result.cleaned == "abcdef"
    assert "strip_control_chars" in result.transforms
    ws = clean_text_value(" \t \n ")
    assert ws.cleaned == ""


def test_very_long_string_truncation_is_audited():
    raw = "x" * 100_000
    result = clean_text_value(raw, TextCleanConfig(max_length=1000))
    assert len(result.cleaned) == 1000
    assert "truncate" in result.transforms
    assert result.original == raw  # original preserved in full


def test_public_exports():
    assert fd.clean_text is clean_text
    assert fd.clean_text_value is clean_text_value
    assert fd.TextCleanConfig is TextCleanConfig
    assert isinstance(fd.clean_text_value("x"), CleanedText)
