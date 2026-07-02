"""CLI: freshdata clean --context-file and freshdata policy compile."""

import json

import pandas as pd
import pytest

from freshdata.enterprise.cli import main

RULES = """CustomerID is unique.
Never modify revenue values.
"""


@pytest.fixture()
def data_csv(tmp_path):
    path = tmp_path / "in.csv"
    pd.DataFrame(
        {
            "CustomerID": [1, 2, 2],
            "Emails": ["a@x.com", "b@x.com", "c@x.com"],
            "revenue": [10.0, 20.0, 30.0],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture()
def rules_txt(tmp_path):
    path = tmp_path / "rules.txt"
    path.write_text(RULES, encoding="utf-8")
    return path


def test_policy_compile_prints_summary(data_csv, rules_txt, capsys):
    code = main(["policy", "compile", str(rules_txt), "--schema", str(data_csv)])
    out = capsys.readouterr().out
    assert code == 0
    assert "unique" in out and "customer_id" in out
    assert "protected" in out and "revenue" in out


def test_policy_compile_writes_json(data_csv, rules_txt, tmp_path, capsys):
    out_path = tmp_path / "policy.json"
    code = main(
        ["policy", "compile", str(rules_txt), "--schema", str(data_csv),
         "--output", str(out_path)]
    )
    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["policy_version"] == "1"
    assert {c["rule"] for c in payload["constraints"]} == {"unique", "protected"}


def test_policy_compile_schema_free(rules_txt, capsys):
    code = main(["policy", "compile", str(rules_txt)])
    assert code == 0
    assert "CustomerID" in capsys.readouterr().out


def test_policy_compile_strict_fails_on_unparsed(tmp_path, data_csv, capsys):
    bad = tmp_path / "bad.txt"
    bad.write_text("Utter gibberish sentence.\n", encoding="utf-8")
    code = main(["policy", "compile", str(bad), "--schema", str(data_csv), "--strict"])
    assert code == 2
    assert "unparsed_sentence" in capsys.readouterr().out


def test_policy_compile_strict_fails_on_unresolved(tmp_path, data_csv, capsys):
    bad = tmp_path / "bad.txt"
    bad.write_text("heart_rate must be between 0 and 200.\n", encoding="utf-8")
    code = main(["policy", "compile", str(bad), "--schema", str(data_csv), "--strict"])
    assert code == 2
    assert "unresolved" in capsys.readouterr().out


def test_clean_with_context_file(data_csv, rules_txt, tmp_path, capsys):
    out_csv = tmp_path / "out.csv"
    code = main(
        ["clean", str(data_csv), "-o", str(out_csv), "--context-file", str(rules_txt), "--quiet"]
    )
    assert code == 0
    written = pd.read_csv(out_csv)
    assert written["revenue"].tolist() == [10.0, 20.0, 30.0]


def test_clean_strict_context_fails_nonzero(data_csv, tmp_path, capsys):
    bad = tmp_path / "bad.txt"
    bad.write_text("Utter gibberish sentence.\n", encoding="utf-8")
    out_csv = tmp_path / "out.csv"
    code = main(
        ["clean", str(data_csv), "-o", str(out_csv), "--context-file", str(bad),
         "--strict", "--quiet"]
    )
    assert code == 2
    assert not out_csv.exists()


def test_clean_without_context_unchanged(data_csv, tmp_path):
    out_csv = tmp_path / "out.csv"
    assert main(["clean", str(data_csv), "-o", str(out_csv), "--quiet"]) == 0
    assert out_csv.exists()
