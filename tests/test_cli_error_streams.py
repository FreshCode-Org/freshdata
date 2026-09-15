"""Exit-2/3 CLI errors go to stderr, never stdout (FDC-L5-010).

Every ``freshdata`` failure uses the one-line ``freshdata: error: ...`` form on
stderr, as the top-level handler does for exit 1, so a pipeline reading stdout
(for example ``--output-format json | jq``) never gets an error line in its data.
Exit codes are unchanged.
"""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.enterprise import cli


@pytest.fixture
def files(tmp_path):
    data = tmp_path / "in.csv"
    pd.DataFrame({"id": [1, 2], "v": [2, 3]}).to_csv(data, index=False)
    rules = tmp_path / "rules.txt"
    rules.write_text("frobnicate the wibble\n")
    return {
        "in": str(data),
        "rules": str(rules),
        "missing": str(tmp_path / "nope.fdprofile"),
        "out": str(tmp_path / "out.csv"),
    }


_CASES = {
    "clean --profile missing": (
        ["clean", "{in}", "-o", "{out}", "--profile", "{missing}"], 2, "nope.fdprofile"
    ),
    "clean --strict unparsed context": (
        ["clean", "{in}", "-o", "{out}", "--context-file", "{rules}", "--strict"], 2, ""
    ),
    "clean --engine with --context-file": (
        ["clean", "{in}", "--engine", "polars", "--context-file", "{rules}"],
        2,
        "--context-file is only supported on the pandas engine",
    ),
    "clean --engine with --profile": (
        ["clean", "{in}", "--engine", "duckdb", "--profile", "{missing}"],
        2,
        "--profile is only supported on the pandas engine",
    ),
    "profile extra arguments": (
        ["profile", "{in}", "extra"], 2, "unexpected extra arguments"
    ),
    "profile audit usage": (["profile", "audit"], 2, "usage: freshdata profile audit"),
    "profile audit missing": (["profile", "audit", "{missing}"], 2, "nope.fdprofile"),
    "profile diff usage": (["profile", "diff", "{missing}"], 2, "usage: freshdata profile diff"),
    "profile merge usage": (
        ["profile", "merge", "{missing}"], 2, "usage: freshdata profile merge"
    ),
    "profile merge without -o": (
        ["profile", "merge", "{missing}", "{missing}"], 2, "requires -o/--output"
    ),
    "policy compile --strict": (["policy", "compile", "{rules}", "--strict"], 2, ""),
    "models pull unknown id": (["models", "pull", "no-such-model"], 2, "no-such-model"),
}


@pytest.mark.parametrize("label", list(_CASES))
def test_error_goes_to_stderr_with_unchanged_exit_code(files, capsys, label):
    argv, code, needle = _CASES[label]
    assert cli.main([arg.format(**files) for arg in argv]) == code
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("freshdata: error: ")
    assert needle in captured.err
    assert "Traceback" not in captured.err
