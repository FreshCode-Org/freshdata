"""Malformed CLI input files and non-UTF-8 stdout (#289, #295).

Exit-code convention: ``freshdata clean`` reports a bad input/config file as a
one-line ``freshdata: error: ...`` with exit 1 (the same as a missing input file);
``freshdata validate`` reports rules it cannot load with exit 2 ("usage/load error")
so it is never confused with exit 1 ("validation failed").
"""

from __future__ import annotations

import io
import json
import sys

import pandas as pd
import pytest

from freshdata.enterprise import cli
from freshdata.validation_suite import ValidationSuite


@pytest.fixture
def src(tmp_path):
    path = tmp_path / "in.csv"
    pd.DataFrame({"id": [1, 2], "email": ["a@b.com", "c@d.com"]}).to_csv(path, index=False)
    return path


def _clean_with_config(src, cfg, tmp_path):
    return cli.main(
        ["clean", str(src), "-o", str(tmp_path / "out.csv"), "--config", str(cfg), "--quiet"]
    )


def _assert_one_line_error(capsys, *needles):
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "freshdata: error:" in err
    for needle in needles:
        assert needle in err
    return err


# --------------------------------------------------------------------------- #
# #289: freshdata clean --config                                              #
# --------------------------------------------------------------------------- #
def test_clean_invalid_yaml_config_is_one_line_error(src, tmp_path, capsys):
    pytest.importorskip("yaml")
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("enterprise:\n  masking: [\n")
    assert _clean_with_config(src, cfg, tmp_path) == 1
    err = _assert_one_line_error(capsys, "invalid YAML", "bad.yaml")
    assert len(err.strip().splitlines()) == 1


def test_clean_unknown_masking_rule_key_is_one_line_error(src, tmp_path, capsys):
    cfg = tmp_path / "typo.json"
    cfg.write_text(
        json.dumps(
            {"enterprise": {"masking": [{"name": "m", "columns": ["email"], "colums": 1}]}}
        )
    )
    assert _clean_with_config(src, cfg, tmp_path) == 1
    _assert_one_line_error(capsys, "typo.json", "colums")


@pytest.mark.parametrize("payload", ["[1, 2]", '"text"', "3"])
def test_clean_non_object_json_config_is_one_line_error(src, tmp_path, capsys, payload):
    cfg = tmp_path / "list.json"
    cfg.write_text(payload)
    assert _clean_with_config(src, cfg, tmp_path) == 1
    _assert_one_line_error(capsys, "list.json", "must contain a JSON/YAML object")


def test_clean_non_object_yaml_config_is_one_line_error(src, tmp_path, capsys):
    pytest.importorskip("yaml")
    cfg = tmp_path / "list.yaml"
    cfg.write_text("- 1\n- 2\n")
    assert _clean_with_config(src, cfg, tmp_path) == 1
    _assert_one_line_error(capsys, "list.yaml", "must contain a JSON/YAML object")


@pytest.mark.parametrize("key", ["clean", "enterprise"])
def test_clean_non_object_config_section_is_one_line_error(src, tmp_path, capsys, key):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({key: [1, 2]}))
    assert _clean_with_config(src, cfg, tmp_path) == 1
    _assert_one_line_error(capsys, f"'{key}'", "must be an object")


def test_clean_non_object_masking_entry_is_one_line_error(src, tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"enterprise": {"masking": ["email"]}}))
    assert _clean_with_config(src, cfg, tmp_path) == 1
    _assert_one_line_error(capsys, "'enterprise'", "cfg.json")


def test_clean_unknown_clean_option_is_one_line_error(src, tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"clean": {"stratgy": "conservative"}}))
    assert _clean_with_config(src, cfg, tmp_path) == 1
    _assert_one_line_error(capsys, "cfg.json", "stratgy")


def test_clean_empty_yaml_config_still_works(src, tmp_path):
    pytest.importorskip("yaml")
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")
    assert _clean_with_config(src, cfg, tmp_path) == 0


def test_clean_valid_config_sections_still_work(src, tmp_path):
    cfg = tmp_path / "ok.json"
    cfg.write_text(
        json.dumps(
            {
                "clean": {"strategy": "conservative"},
                "enterprise": {"masking": [{"name": "m", "columns": ["email"]}]},
            }
        )
    )
    assert _clean_with_config(src, cfg, tmp_path) == 0
    assert "a@b.com" not in (tmp_path / "out.csv").read_text()


# --------------------------------------------------------------------------- #
# #289: freshdata validate --suite / --contract exit 2 on unloadable rules    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", ["--suite", "--contract"])
@pytest.mark.parametrize("payload", ["[]", "[1, 2]", '"text"'])
def test_validate_non_object_rules_exit_2(src, tmp_path, capsys, flag, payload):
    rules = tmp_path / "rules.json"
    rules.write_text(payload)
    assert cli.main(["validate", str(src), flag, str(rules)]) == 2
    err = capsys.readouterr().err
    assert "could not load rules" in err
    assert "must be a JSON object" in err
    assert "Traceback" not in err


def test_suite_from_dict_rejects_non_mapping():
    with pytest.raises(ValueError, match="must be a JSON object"):
        ValidationSuite.from_dict([])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# #295: a non-UTF-8 stdout must not turn a passed gate into exit 1            #
# --------------------------------------------------------------------------- #
def _ascii_stdout(monkeypatch):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    return stream, raw


def test_clean_summary_on_ascii_stdout_exits_0(src, tmp_path, monkeypatch):
    stream, raw = _ascii_stdout(monkeypatch)
    out = tmp_path / "out.csv"
    code = cli.main(["clean", str(src), "-o", str(out), "--fail-under-trust", "50"])
    stream.flush()
    assert code == 0
    assert out.exists()
    printed = raw.getvalue().decode("ascii")
    assert "freshdata enterprise" in printed
    assert "?" in printed  # the arrow was replaced, not raised


def test_clean_failed_gate_on_ascii_stdout_still_exits_1(src, tmp_path, monkeypatch):
    stream, _ = _ascii_stdout(monkeypatch)
    code = cli.main(
        ["clean", str(src), "-o", str(tmp_path / "out.csv"), "--fail-under-trust", "101"]
    )
    stream.flush()
    assert code == 1


def test_validate_verdict_on_ascii_stdout_exits_0(src, tmp_path, monkeypatch):
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"name": "s"}))
    stream, raw = _ascii_stdout(monkeypatch)
    code = cli.main(["validate", str(src), "--suite", str(suite)])
    stream.flush()
    assert code == 0
    assert "PASS" in raw.getvalue().decode("ascii")


def test_safe_print_replaces_unencodable_characters(monkeypatch):
    stream, raw = _ascii_stdout(monkeypatch)
    cli._safe_print("trust 1.0 → 2.0 — ok")
    stream.flush()
    assert raw.getvalue() == b"trust 1.0 ? 2.0 ? ok\n"
