"""Malformed CLI input files and non-UTF-8 stdout (#289, #295).

Exit-code convention: ``freshdata clean`` reports a bad input/config file as a
one-line ``freshdata: error: ...`` with exit 1 (the same as a missing input file);
``freshdata validate`` reports rules it cannot load with exit 2 ("usage/load error")
so it is never confused with exit 1 ("validation failed").
"""

from __future__ import annotations

import dataclasses
import io
import json
import sys

import pandas as pd
import pytest

from freshdata.enterprise import cli
from freshdata.enterprise.config import EnterpriseConfig
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
# Unknown / unsupported 'enterprise' keys and top-level sections              #
# --------------------------------------------------------------------------- #
def _write_cfg(tmp_path, payload):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(payload))
    return cfg


def test_clean_unknown_enterprise_keys_error_with_did_you_mean(src, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, {"enterprise": {"enable_maskin": True, "fail_under_trus": 99}})
    assert _clean_with_config(src, cfg, tmp_path) == 1
    err = _assert_one_line_error(
        capsys,
        "cfg.json",
        "'enable_maskin' (did you mean 'enable_masking'?)",
        "'fail_under_trus' (did you mean 'fail_under_trust'?)",
    )
    assert len(err.strip().splitlines()) == 1
    assert not (tmp_path / "out.csv").exists()


def test_clean_unknown_top_level_section_errors_with_did_you_mean(src, tmp_path, capsys):
    cfg = _write_cfg(tmp_path, {"enterprize": {"enable_masking": True}})
    assert _clean_with_config(src, cfg, tmp_path) == 1
    err = _assert_one_line_error(
        capsys, "cfg.json", "'enterprize' (did you mean 'enterprise'?)", "clean, enterprise"
    )
    assert len(err.strip().splitlines()) == 1
    assert not (tmp_path / "out.csv").exists()


@pytest.mark.parametrize(
    "payload, needle",
    [
        ({"privacy": {"min_scor": 0.5}}, "'min_scor' (did you mean 'min_score'?)"),
        ({"clustering": {"colums": ["email"]}}, "'colums' (did you mean 'columns'?)"),
        (
            {"entity_resolution": {"comparisons": [{"colum": "email"}]}},
            "'colum' (did you mean 'column'?)",
        ),
    ],
)
def test_clean_unknown_nested_enterprise_key_is_one_line_error(
    src, tmp_path, capsys, payload, needle
):
    cfg = _write_cfg(tmp_path, {"enterprise": payload})
    assert _clean_with_config(src, cfg, tmp_path) == 1
    _assert_one_line_error(capsys, "cfg.json", needle)


@pytest.mark.parametrize(
    "key, value", [("enable_contracts", True), ("drift", {}), ("anonymization", [])]
)
def test_clean_unsupported_enterprise_field_is_rejected(src, tmp_path, capsys, key, value):
    cfg = _write_cfg(tmp_path, {"enterprise": {key: value}})
    assert _clean_with_config(src, cfg, tmp_path) == 1
    _assert_one_line_error(capsys, "cfg.json", f"'{key}' is not supported in --config")


@pytest.mark.parametrize(
    "payload, needle",
    [
        ({"enable_masking": "false"}, "'enable_masking' must be true or false"),
        ({"fail_under_trust": "80"}, "'fail_under_trust' must be a number or null"),
        ({"actor": 7}, "'actor' must be a string or null"),
        ({"privacy": [1]}, "'privacy' must be an object"),
    ],
)
def test_clean_wrongly_typed_enterprise_value_is_one_line_error(
    src, tmp_path, capsys, payload, needle
):
    cfg = _write_cfg(tmp_path, {"enterprise": payload})
    assert _clean_with_config(src, cfg, tmp_path) == 1
    _assert_one_line_error(capsys, "cfg.json", needle)


def test_every_enterprise_config_field_is_accepted_or_rejected():
    """A new EnterpriseConfig field must be wired into --config or rejected explicitly."""
    fields = {f.name for f in dataclasses.fields(EnterpriseConfig)}
    handled = set(cli._ENTERPRISE_KEYS) | set(cli._ENTERPRISE_UNSUPPORTED_KEYS)
    assert fields == handled
    assert not set(cli._ENTERPRISE_KEYS) & set(cli._ENTERPRISE_UNSUPPORTED_KEYS)


def _clean_loud(src, cfg, tmp_path, *extra):
    return cli.main(
        ["clean", str(src), "-o", str(tmp_path / "out.csv"), "--config", str(cfg), *extra]
    )


def test_clean_config_privacy_detection_takes_effect(src, tmp_path, capsys):
    cfg = _write_cfg(
        tmp_path, {"enterprise": {"enable_privacy_detection": True, "privacy": {}}}
    )
    assert _clean_loud(src, cfg, tmp_path) == 0
    assert "privacy:" in capsys.readouterr().out
    assert "a@b.com" not in (tmp_path / "out.csv").read_text()


def test_clean_config_lineage_takes_effect(src, tmp_path):
    cfg = _write_cfg(tmp_path, {"enterprise": {"lineage": {"job_name": "nightly.customers"}}})
    lineage = tmp_path / "lineage.json"
    assert _clean_loud(src, cfg, tmp_path, "--quiet", "--lineage", str(lineage)) == 0
    assert "nightly.customers" in lineage.read_text()


def test_clean_config_k_anonymity_takes_effect(src, tmp_path, capsys):
    cfg = _write_cfg(
        tmp_path,
        {"enterprise": {"k_anonymity": {"enabled": True, "quasi_identifiers": ["email"], "k": 2}}},
    )
    assert _clean_loud(src, cfg, tmp_path) == 0
    assert "k-anonymity (k=2)" in capsys.readouterr().out


def test_clean_config_entity_resolution_takes_effect(src, tmp_path, capsys):
    cfg = _write_cfg(
        tmp_path,
        {
            "enterprise": {
                "enable_entity_resolution": True,
                "entity_resolution": {
                    "backend": "pandas",
                    "unique_id_column": "id",
                    "blocking_rules": [{"sql": "l.email = r.email"}],
                    "comparisons": [{"column": "email"}],
                },
            }
        },
    )
    assert _clean_loud(src, cfg, tmp_path) == 0
    assert "entity resolution (pandas)" in capsys.readouterr().out


def test_clean_config_empty_clustering_object_still_means_no_clustering():
    ec = cli._build_enterprise({"clustering": {}, "enable_clustering": True})
    assert ec.clustering is None
    assert ec.enable_clustering is True


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
