"""Layer 3: RenderOptions and the plain-text renderer."""

from __future__ import annotations

import json

import pytest

from freshdata.render.normalize import normalize_clean_report
from freshdata.render.options import (
    MODES,
    RenderOptions,
    get_display,
    reset_display,
    set_display,
)
from freshdata.render.plain import render_plain
from freshdata.report import CleanReport


@pytest.fixture(autouse=True)
def _clean_display_state(monkeypatch):
    for var in ("FRESHDATA_DISPLAY", "NO_COLOR", "FRESHDATA_NO_PREVIEWS"):
        monkeypatch.delenv(var, raising=False)
    reset_display()
    yield
    reset_display()


def make_view(**report_kw):
    rep = CleanReport(
        rows_before=10_000,
        rows_after=9_988,
        cols_before=14,
        cols_after=14,
        missing_before=1_204,
        missing_after=0,
        duration_seconds=0.412,
    )
    for key, value in report_kw.items():
        setattr(rep, key, value)
    rep.add("missing", "filled 214 value(s) with median", column="age", count=214)
    return normalize_clean_report(rep)


class TestOptions:
    def test_defaults(self):
        opts = get_display()
        assert opts.mode == "auto"
        assert opts.color == "auto"
        assert opts.previews is True

    def test_set_display_is_process_wide(self):
        set_display(mode="compact")
        assert get_display().mode == "compact"

    def test_set_display_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown display mode"):
            set_display(mode="fancy")

    def test_env_mode_applies_when_auto(self, monkeypatch):
        monkeypatch.setenv("FRESHDATA_DISPLAY", "verbose")
        assert get_display().mode == "verbose"

    def test_set_display_beats_env(self, monkeypatch):
        monkeypatch.setenv("FRESHDATA_DISPLAY", "verbose")
        set_display(mode="compact")
        assert get_display().mode == "compact"

    def test_overrides_beat_everything(self, monkeypatch):
        monkeypatch.setenv("FRESHDATA_DISPLAY", "verbose")
        set_display(mode="compact")
        assert get_display(mode="debug").mode == "debug"

    def test_no_color_env(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert get_display().color == "never"

    def test_no_previews_env(self, monkeypatch):
        monkeypatch.setenv("FRESHDATA_NO_PREVIEWS", "1")
        assert get_display().previews is False

    def test_plain_mode_implies_ascii_and_no_color(self):
        opts = get_display(mode="plain")
        assert opts.ascii_icons is True
        assert opts.color == "never"

    def test_auto_resolves_by_tty(self):
        opts = RenderOptions()
        assert opts.resolved_mode(isatty=True) == "standard"
        assert opts.resolved_mode(isatty=False) == "compact"

    def test_all_modes_are_known(self):
        assert set(MODES) == {
            "auto", "compact", "standard", "verbose", "debug", "json", "plain", "silent",
        }


class TestPlainRenderer:
    def test_silent_is_empty(self):
        assert render_plain(make_view(), get_display(mode="silent")) == ""

    def test_json_mode_is_report_to_dict(self):
        out = render_plain(make_view(), get_display(mode="json"))
        payload = json.loads(out)
        assert payload["rows_before"] == 10_000
        assert payload["actions"][0]["step"] == "missing"

    def test_compact_is_two_lines_with_pointer(self):
        out = render_plain(
            make_view(warnings=["income is 38% missing"]),
            get_display(mode="compact"),
        )
        lines = out.splitlines()
        assert len(lines) == 2
        assert "freshdata clean" in lines[0]
        assert "CHANGED" in lines[0]
        assert "REVIEW" in lines[0]
        assert "1 attention (W1)" in lines[0]
        assert "report.show()" in lines[1]

    def test_standard_layout(self):
        out = render_plain(
            make_view(warnings=["income is 38% missing"]),
            get_display(mode="standard"),
        )
        assert "freshdata clean" in out
        assert "CHANGED · REVIEW" in out
        assert "9,988 of 10,000 rows kept" in out
        assert "Needs attention (1)" in out
        assert "Warning" in out and "[W1]" in out
        assert "next   fd.explain_clean" in out
        # sections are not rendered in standard mode
        assert "All actions" not in out

    def test_empty_attention_is_explicit(self):
        out = render_plain(make_view(), get_display(mode="standard"))
        assert "nothing needs review" in out

    def test_banner_rendered_prominently(self):
        out = render_plain(make_view(materialized=False), get_display(mode="standard"))
        assert "!! PARTIAL — result kept in the engine" in out
        assert "PARTIAL" in out.splitlines()[0]

    def test_verbose_includes_sections_but_not_audit(self):
        out = render_plain(
            make_view(decisions_hash="abc123"),
            get_display(mode="verbose"),
        )
        assert "Column changes" in out
        assert "All actions" in out
        assert "filled missing values" in out
        assert "abc123" not in out

    def test_debug_includes_audit(self):
        out = render_plain(
            make_view(decisions_hash="abc123", backend="polars"),
            get_display(mode="debug"),
        )
        assert "Audit" in out
        assert "abc123" in out
        assert "polars" in out

    def test_plain_mode_has_no_unicode_frame(self):
        out = render_plain(
            make_view(warnings=["w"]),
            get_display(mode="plain"),
        )
        for char in ("─", "·", "→"):
            assert char not in out
        assert "Needs attention" in out  # content identical to standard

    def test_width_is_respected(self):
        for width in (60, 100):
            out = render_plain(make_view(), get_display(mode="standard", width=width))
            assert len(out.splitlines()[0]) == width

    def test_grepable_status_labels(self):
        out = render_plain(make_view(warnings=["w"]), get_display(mode="standard"))
        assert "REVIEW" in out  # grep REVIEW works with no ANSI in the way
        assert "\x1b[" not in out
