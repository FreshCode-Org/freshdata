"""Tests for stakeholder-safe business-language summaries."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd


@pytest.fixture
def report():
    df = pd.DataFrame({
        "amount": [1.0, 2.0, None, 4.0, 100000.0, 2.0],
        "email": ["a@x.com", "a@x.com", None, "b@x.com", "c@x.com", "a@x.com"],
        "id": [1, 2, 3, 4, 5, 2],
    })
    _, rep = fd.clean(df, return_report=True)
    return rep


def test_markdown_export(report) -> None:
    summary = fd.stakeholder_summary(report, audience="business", format="markdown")
    md = summary.to_markdown()
    assert md.startswith("# Data quality summary")
    assert "## What changed" in md or "## What was preserved" in md


def test_html_export(report) -> None:
    summary = fd.stakeholder_summary(report, format="html")
    html = summary.to_html()
    assert "<div class=\"fd-report\"" in html
    assert summary.render() == html  # format="html" → render() returns HTML


def test_business_language_no_dtype_jargon(report) -> None:
    summary = fd.stakeholder_summary(report, audience="business")
    text = summary.summary().lower()
    assert "dtype" not in text and "int64" not in text


def test_dict_export_and_metrics(report) -> None:
    summary = fd.stakeholder_summary(report)
    d = summary.to_dict()
    assert "completeness" in d["metrics"]
    assert "what_changed" in d and "needs_review" in d


def test_invalid_arguments(report) -> None:
    with pytest.raises(ValueError, match="audience"):
        fd.stakeholder_summary(report, audience="nope")
    with pytest.raises(ValueError, match="format"):
        fd.stakeholder_summary(report, format="pdf")
