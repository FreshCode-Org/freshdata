"""Layer 6: the Peel notebook HTML renderer and its opt-in gate."""

from __future__ import annotations

import pytest

import freshdata as fd
from freshdata.render.normalize import normalize_clean_report
from freshdata.render.notebook import render_notebook
from freshdata.render.options import reset_display
from freshdata.report import CleanReport


@pytest.fixture(autouse=True)
def _clean_display_state(monkeypatch):
    for var in (
        "FRESHDATA_DISPLAY",
        "FRESHDATA_LEGACY_DISPLAY",
        "NO_COLOR",
        "FRESHDATA_NO_PREVIEWS",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_display()
    yield
    reset_display()


def make_report(**kw) -> CleanReport:
    rep = CleanReport(
        rows_before=1_000, rows_after=998, cols_before=5, cols_after=5,
        missing_before=40, missing_after=0, duration_seconds=0.2,
    )
    for key, value in kw.items():
        setattr(rep, key, value)
    rep.add("missing", "filled 40 value(s) with median", column="age", count=40)
    return rep


def make_view(**kw):
    return normalize_clean_report(make_report(**kw))


class TestPeelHtml:
    def test_contains_status_headline_attention(self):
        html = render_notebook(make_view(warnings=["column 'x' is mostly empty"]))
        assert "freshdata clean" in html
        assert "CHANGED" in html and "REVIEW" in html
        assert "998 of 1,000 rows kept" in html
        assert "Needs attention (1)" in html
        assert "column &#x27;x&#x27; is mostly empty" in html or "column 'x'" in html
        assert "[W1]" in html

    def test_empty_attention_is_explicit(self):
        assert "nothing needs review" in render_notebook(make_view())

    def test_sections_are_collapsed_details(self):
        html = render_notebook(make_view())
        assert "<details>" in html
        assert "Column changes" in html
        assert "All actions" in html
        # collapsed by default: no open attribute
        assert "<details open>" not in html

    def test_banner_present_for_partial(self):
        html = render_notebook(make_view(materialized=False))
        assert "PARTIAL" in html
        assert "kept in the engine" in html

    def test_next_step_rendered_as_code(self):
        html = render_notebook(make_view(warnings=["w"]))
        assert "<code>" in html and "explain_clean" in html

    def test_json_export_control_present(self):
        assert "JSON" in render_notebook(make_view())

    def test_hostile_column_name_is_escaped(self):
        rep = make_report()
        rep.add_warning('<script>alert(1)</script> in column "<img onerror=x>"')
        html = render_notebook(normalize_clean_report(rep))
        assert "<script>" not in html
        assert "<img" not in html
        assert "&lt;script&gt;" in html

    def test_large_sections_are_capped_with_honest_count(self):
        rep = make_report()
        for i in range(120):
            rep.add("missing", f"filled value {i}", column=f"c{i}", count=1)
        html = render_notebook(normalize_clean_report(rep))
        assert "showing 50 of 121" in html


class TestOptInGate:
    def test_default_is_legacy_layout(self):
        html = make_report().to_html()
        assert "Needs attention" not in html  # legacy renderer

    def test_set_display_peel_switches(self):
        fd.set_display("peel")
        html = make_report().to_html()
        assert "Needs attention" in html or "nothing needs review" in html

    def test_env_opt_in(self, monkeypatch):
        monkeypatch.setenv("FRESHDATA_DISPLAY", "peel")
        html = make_report().to_html()
        assert "nothing needs review" in html or "Needs attention" in html

    def test_legacy_escape_hatch_wins(self, monkeypatch):
        fd.set_display("peel")
        monkeypatch.setenv("FRESHDATA_LEGACY_DISPLAY", "1")
        html = make_report().to_html()
        assert "Needs attention" not in html

    def test_unknown_style_rejected(self):
        with pytest.raises(ValueError, match="unknown display style"):
            fd.set_display("shiny")

    def test_peel_style_without_normalizer_falls_back_to_legacy(self):
        fd.set_display("peel")
        profile = fd.profile(__import__("pandas").DataFrame({"a": [1, 2]}))
        html = profile.to_html()
        assert html  # legacy profile renderer still works, no exception

    def test_repr_html_uses_peel_when_opted_in(self):
        fd.set_display("peel")
        html = make_report()._repr_html_()
        assert "nothing needs review" in html or "Needs attention" in html
