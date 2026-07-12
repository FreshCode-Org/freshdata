"""Layer 4: show(mode=, renderer=) integration and fd.set_display."""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import pytest

import freshdata as fd
from freshdata.render.mixins import HtmlReprMixin
from freshdata.render.options import reset_display
from freshdata.report import CleanReport


@pytest.fixture(autouse=True)
def _clean_display_state(monkeypatch):
    for var in ("FRESHDATA_DISPLAY", "NO_COLOR", "FRESHDATA_NO_PREVIEWS"):
        monkeypatch.delenv(var, raising=False)
    reset_display()
    yield
    reset_display()


def make_report() -> CleanReport:
    rep = CleanReport(
        rows_before=100, rows_after=99, cols_before=3, cols_after=3, duration_seconds=0.01
    )
    rep.add("duplicates", "removed 1 duplicate row", count=1)
    rep.add_warning("column 'x' is mostly empty")
    return rep


class TestShowModes:
    def test_mode_prints_peel_text(self, capsys):
        result = make_report().show(mode="standard")
        out = capsys.readouterr().out
        assert result is None
        assert "freshdata clean" in out
        assert "CHANGED · REVIEW" in out
        assert "[W1]" in out

    def test_renderer_terminal_defaults_to_auto_mode(self, capsys):
        make_report().show(renderer="terminal")
        out = capsys.readouterr().out
        assert "freshdata clean" in out  # compact or standard depending on TTY

    def test_mode_silent_prints_nothing(self, capsys):
        make_report().show(mode="silent")
        assert capsys.readouterr().out == ""

    def test_mode_json_prints_to_dict(self, capsys):
        make_report().show(mode="json")
        payload = json.loads(capsys.readouterr().out)
        assert payload["rows_before"] == 100

    def test_set_display_governs_default_mode(self, capsys):
        fd.set_display(mode="compact")
        make_report().show(renderer="terminal")
        out = capsys.readouterr().out
        assert len(out.rstrip("\n").splitlines()) == 2

    def test_no_args_keeps_legacy_html_file_behavior(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        tempfile.tempdir = None  # re-read TMPDIR
        try:
            path = make_report().show()
            assert path is not None and path.endswith(".html")
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as fh:
                assert "<!doctype html>" in fh.read()
        finally:
            tempfile.tempdir = None


class TestFallbackSafety:
    def test_unregistered_kind_falls_back_to_summary(self, capsys):
        class Odd(HtmlReprMixin):
            _render_kind = "no_such_kind"

            def summary(self):
                return "odd summary text"

        Odd().show(mode="standard")
        assert "odd summary text" in capsys.readouterr().out

    def test_unregistered_kind_without_summary_uses_repr(self, capsys):
        class Odd(HtmlReprMixin):
            _render_kind = "no_such_kind"

        Odd().show(mode="standard")
        assert "Odd" in capsys.readouterr().out

    def test_bad_mode_falls_back_not_raises(self, capsys):
        make_report().show(mode="not-a-mode")
        out = capsys.readouterr().out
        assert "freshdata clean report" in out  # legacy summary() text


class TestTopLevelExports:
    def test_set_display_exported(self):
        assert fd.set_display is not None
        assert "set_display" in fd.__all__
        assert "get_display" in fd.__all__

    def test_display_options_do_not_touch_cleaning(self):
        fd.set_display(mode="silent")
        df = pd.DataFrame({"a": [1, 1, None]})
        cleaned, report = fd.clean(df, return_report=True)
        assert report.rows_before == 3  # cleaning unaffected by display config
