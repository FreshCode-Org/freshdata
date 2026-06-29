"""Tests for mixed-language / encoding linting."""

from __future__ import annotations

import unicodedata

import pandas as pd

import freshdata as fd


def test_detects_mixed_script() -> None:
    # Second value's "A" is Cyrillic U+0410, not Latin.
    df = pd.DataFrame({"name": ["Alice", "Аlice", "Bob"]})
    rep = fd.lint_text_encoding(df, columns=["name"])
    kinds = {i.issue_type for i in rep.issues}
    assert "mixed_script" in kinds


def test_detects_mojibake() -> None:
    df = pd.DataFrame({"note": ["école", "Ã©cole", "fine"]})
    rep = fd.lint_text_encoding(df, columns=["note"])
    assert any(i.issue_type == "mojibake" for i in rep.issues)


def test_detects_nfc_nfd_inconsistency() -> None:
    nfd = unicodedata.normalize("NFD", "café")  # decomposed e + combining acute
    df = pd.DataFrame({"city": [nfd, "Paris"]})
    rep = fd.lint_text_encoding(df, columns=["city"])
    issue = next(i for i in rep.issues if i.issue_type == "nfc_nfd_inconsistency")
    assert issue.auto_repair_safe is True  # NFC normalization is safe


def test_detects_replacement_char_and_marks_unsafe() -> None:
    df = pd.DataFrame({"x": ["good", "bad�byte"]})
    rep = fd.lint_text_encoding(df, columns=["x"])
    issue = next(i for i in rep.issues if i.issue_type == "replacement_char")
    assert issue.severity == "high"
    assert issue.auto_repair_safe is False  # cannot safely auto-fix lost bytes


def test_no_unsafe_auto_modification_of_input() -> None:
    df = pd.DataFrame({"x": ["bad�byte"]})
    before = df.copy()
    fd.lint_text_encoding(df, columns=["x"])
    pd.testing.assert_frame_equal(df, before)  # input untouched


def test_ambiguous_date_flagged_for_review() -> None:
    df = pd.DataFrame({"d": ["01/02/2020", "11/12/2020"]})
    rep = fd.lint_text_encoding(df, columns=["d"], locale_hints=["en_IN"])
    issue = next(i for i in rep.issues if i.issue_type == "ambiguous_date")
    assert issue.human_review is True
    assert rep.locale_hints == ["en_IN"]


def test_report_surfaces() -> None:
    df = pd.DataFrame({"name": ["Аlice", "Bob"]})
    rep = fd.lint_text_encoding(df)
    assert "column" in rep.to_frame().columns
    assert rep.to_dict()["n_issues"] == len(rep.issues)
    assert "<div class=\"fd-report\"" in rep.to_html()
