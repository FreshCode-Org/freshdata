"""CLI tests — driven through ``cli.main(argv)`` so they double as coverage."""

import json

import pandas as pd
import pytest

from freshdata.enterprise import cli


def test_infer_format():
    assert cli._infer_format("a.csv") == "csv"
    assert cli._infer_format("a.parquet") == "parquet"
    assert cli._infer_format("a.PQ") == "parquet"
    assert cli._infer_format("a.json") == "json"
    assert cli._infer_format("a.weird") == "csv"


def test_clean_writes_outputs_and_passes_gate(tmp_path, capsys):
    src = tmp_path / "in.csv"
    pd.DataFrame({"email": ["a@x.com", "b@y.io"], "v": [1, 2]}).to_csv(src, index=False)
    out, report, lineage = tmp_path / "out.csv", tmp_path / "r.json", tmp_path / "l.json"
    code = cli.main([
        "clean", str(src), "-o", str(out), "--mask", "email:hash",
        "--report", str(report), "--lineage", str(lineage), "--fail-under-trust", "10",
        "--actor", "ci",
    ])
    assert code == 0
    assert "trust" in capsys.readouterr().out
    assert json.loads(report.read_text())["percent_clean"] >= 0
    assert json.loads(lineage.read_text())[0]["eventType"] == "START"
    assert pd.read_csv(out)["email"].iloc[0] != "a@x.com"  # hashed


def test_clean_fails_gate_returns_nonzero(tmp_path):
    src = tmp_path / "in.csv"
    # Two columns so the NaN row survives row-pruning and depresses completeness.
    pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]}).to_csv(src, index=False)
    code = cli.main(["clean", str(src), "--strategy", "conservative",
                     "--fail-under-trust", "100", "--quiet"])
    assert code == 1


def test_cluster_flag_merges(tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"vendor": ["Acme Inc", "ACME  inc", "acme inc"]}).to_csv(src, index=False)
    out = tmp_path / "o.csv"
    assert cli.main(["clean", str(src), "-o", str(out), "--cluster", "vendor", "--quiet"]) == 0
    assert pd.read_csv(out)["vendor"].nunique() == 1


def test_profile_command(tmp_path, capsys):
    src = tmp_path / "in.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(src, index=False)
    assert cli.main(["profile", str(src)]) == 0
    assert "profile" in capsys.readouterr().out.lower()
    assert cli.main(["profile", str(src), "--json"]) == 0
    json.loads(capsys.readouterr().out)


def test_trust_command(tmp_path, capsys):
    src = tmp_path / "in.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(src, index=False)
    assert cli.main(["trust", str(src)]) == 0
    assert "Trust Score" in capsys.readouterr().out
    assert cli.main(["trust", str(src), "--json"]) == 0
    json.loads(capsys.readouterr().out)


def test_trust_fail_under(tmp_path):
    src = tmp_path / "dup.csv"
    pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]}).to_csv(src, index=False)
    assert cli.main(["trust", str(src), "--fail-under", "100"]) == 1


def test_clean_with_json_config(tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"email": ["a@x.com"], "vendor": ["Acme Inc"], "country": ["US"]}).to_csv(
        src, index=False
    )
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "clean": {"strategy": "balanced"},
        "enterprise": {
            "enable_clustering": True,
            "clustering": {"columns": ["vendor"]},
            "masking": [{"name": "pii", "columns": ["email"], "strategy": "hash"}],
            "semantic": [{"name": "iso", "kind": "reference", "columns": ["country"],
                          "reference": ["US", "CA"]}],
            "fail_under_trust": 10,
        },
    }))
    out = tmp_path / "o.csv"
    assert cli.main(["clean", str(src), "-o", str(out), "--config", str(cfg), "--quiet"]) == 0
    assert pd.read_csv(out)["email"].iloc[0] != "a@x.com"


def test_clean_with_yaml_config(tmp_path):
    pytest.importorskip("yaml")
    src = tmp_path / "in.csv"
    pd.DataFrame({"email": ["a@x.com"]}).to_csv(src, index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "enterprise:\n"
        "  masking:\n"
        "    - name: p\n"
        "      columns: [email]\n"
        "      strategy: redact\n"
        "      placeholder: X\n"
    )
    out = tmp_path / "o.csv"
    assert cli.main(["clean", str(src), "-o", str(out), "--config", str(cfg), "--quiet"]) == 0
    assert pd.read_csv(out)["email"].iloc[0] == "X"


def test_parquet_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    src = tmp_path / "in.parquet"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_parquet(src, index=False)
    out = tmp_path / "o.parquet"
    assert cli.main(["clean", str(src), "-o", str(out), "--quiet"]) == 0
    assert out.exists() and not pd.read_parquet(out).empty


def test_json_round_trip(tmp_path):
    src = tmp_path / "in.json"
    pd.DataFrame({"a": [1, 2]}).to_json(src, orient="records")
    out = tmp_path / "o.json"
    assert cli.main(["clean", str(src), "-o", str(out), "--quiet"]) == 0
    assert out.exists()


def test_main_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])
