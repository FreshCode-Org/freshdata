"""Layer 5: the styled terminal renderer (rich optional)."""

from __future__ import annotations

import builtins

import pytest

from freshdata.render.normalize import normalize_clean_report
from freshdata.render.options import get_display, reset_display
from freshdata.render.plain import render_plain
from freshdata.render.terminal import render_terminal_text
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
        rows_before=1_000, rows_after=998, cols_before=5, cols_after=5,
        missing_before=40, missing_after=0, duration_seconds=0.2,
    )
    for key, value in report_kw.items():
        setattr(rep, key, value)
    rep.add("missing", "filled 40 value(s) with median", column="age", count=40)
    return normalize_clean_report(rep)


class TestTerminalRenderer:
    def test_standard_contains_all_plain_content(self):
        view = make_view(warnings=["column 'x' is mostly empty"])
        out = render_terminal_text(view, get_display(mode="standard", color="never"))
        assert "freshdata clean" in out
        assert "CHANGED" in out and "REVIEW" in out
        assert "998 of 1,000 rows kept" in out
        assert "column 'x' is mostly empty" in out
        assert "[W1]" in out
        assert "nothing needs review" not in out

    def test_no_color_has_no_ansi(self):
        out = render_terminal_text(
            make_view(warnings=["w"]), get_display(mode="standard", color="never")
        )
        assert "\x1b[" not in out

    def test_color_emits_ansi(self):
        out = render_terminal_text(
            make_view(warnings=["w"]), get_display(mode="standard", color="always")
        )
        assert "\x1b[" in out

    def test_verbose_includes_sections(self):
        out = render_terminal_text(make_view(), get_display(mode="verbose", color="never"))
        assert "Column changes" in out
        assert "All actions" in out

    def test_compact_mode_delegates_to_plain(self):
        view = make_view()
        opts = get_display(mode="compact")
        assert render_terminal_text(view, opts) == render_plain(view, opts)

    def test_missing_rich_falls_back_to_plain(self, monkeypatch):
        real_import = builtins.__import__

        def no_rich(name, *args, **kwargs):
            if name.startswith("rich"):
                raise ImportError("rich unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_rich)
        view = make_view(warnings=["w"])
        opts = get_display(mode="standard", color="never")
        assert render_terminal_text(view, opts) == render_plain(view, opts)

    def test_width_is_respected(self):
        out = render_terminal_text(
            make_view(), get_display(mode="standard", width=60, color="never")
        )
        assert out.splitlines()
        assert all(len(line) <= 60 for line in out.splitlines())

    def test_show_renderer_terminal_prints(self, capsys):
        rep = CleanReport(rows_before=10, rows_after=10, cols_before=2, cols_after=2)
        rep.add_warning("check column 'a'")
        rep.show(mode="standard", renderer="terminal")
        out = capsys.readouterr().out
        assert "freshdata clean" in out
        assert "check column 'a'" in out
