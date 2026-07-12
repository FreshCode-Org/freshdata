"""Layer 10: additive Peel display flags on `freshdata clean`."""

from __future__ import annotations

import json

import pytest

from freshdata.enterprise import cli


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("a,b\n1,x\n1,x\n,y\n", encoding="utf-8")
    return str(path)


def run(argv, capsys):
    code = cli.main(argv)
    return code, capsys.readouterr().out


class TestBackwardCompatibleDefault:
    def test_default_output_is_legacy_summary(self, csv, capsys):
        code, out = run(["clean", csv], capsys)
        assert code == 0
        # legacy enterprise engine summary wording, unchanged
        assert "freshdata enterprise" in out
        assert "trust" in out
        assert "╭─" not in out  # no Peel panel by default

    def test_quiet_still_suppresses(self, csv, capsys):
        _, out = run(["clean", csv, "--quiet"], capsys)
        assert "freshdata enterprise" not in out


class TestDisplayFlags:
    def test_output_format_json_emits_report_dict(self, csv, capsys):
        code, out = run(["clean", csv, "--output-format", "json"], capsys)
        assert code == 0
        payload = json.loads(out)
        assert payload["rows_before"] == 3
        assert payload["rows_after"] == 2

    def test_verbose_renders_peel(self, csv, capsys):
        _, out = run(["clean", csv, "--verbose", "--no-color"], capsys)
        assert "freshdata clean" in out
        assert "CHANGED" in out or "CLEAN" in out
        assert "\x1b[" not in out  # --no-color strips ANSI

    def test_vv_is_debug_mode(self, csv, capsys):
        _, out = run(["clean", csv, "-vv", "--no-color"], capsys)
        # debug mode surfaces the audit section
        assert "Audit" in out

    def test_display_peel_without_verbose(self, csv, capsys):
        _, out = run(["clean", csv, "--display", "peel", "--no-color"], capsys)
        assert "freshdata clean" in out
        assert "freshdata clean report" not in out  # not the legacy text

    def test_report_file_flag_unaffected(self, csv, tmp_path, capsys):
        # --report still writes the existing enterprise wrapper (clean_report nested)
        report_path = tmp_path / "r.json"
        run(["clean", csv, "--report", str(report_path)], capsys)
        payload = json.loads(report_path.read_text())
        assert payload["clean_report"]["rows_before"] == 3

    def test_json_stdout_and_report_file_coexist(self, csv, tmp_path, capsys):
        report_path = tmp_path / "r.json"
        _, out = run(
            ["clean", csv, "--report", str(report_path), "--output-format", "json"], capsys
        )
        # stdout json is the inner CleanReport; the file keeps the wrapper
        assert json.loads(out)["rows_before"] == 3
        assert json.loads(report_path.read_text())["clean_report"]["rows_before"] == 3
