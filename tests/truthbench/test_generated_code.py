"""Generated-code sandbox: positive contract and one negative test per
prohibited action."""

from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.truthbench import generated_code as gc
from benchmarks.truthbench.fixtures.base import FixtureBuilder
from benchmarks.truthbench.generated_code import (
    GeneratedCodeResult,
    verify_generated_code,
)
from benchmarks.truthbench.models import Disposition


def _fixture():
    frame = pd.DataFrame(
        {
            "account": ["TB-1", "TB-2"],
            "amount": [1.5, 2.5],
            "memo": ["tb.gen+leak@example.invalid", "fine"],
        },
        index=["r1", "r2"],
    )
    builder = FixtureBuilder("v1", "gen", frame, protected_columns=("account",))
    builder.inject(
        "r1",
        "memo",
        "tb.gen+leak@example.invalid",
        Disposition.FLAG,
        family="pii-memo",
        sensitive=True,
    )
    return builder.build()


GOOD = """
import pandas as pd

df = pd.read_csv("your_data.csv")
print("rows:", len(df))
"""


def test_wellformed_code_passes_all_stages():
    result = verify_generated_code(GOOD, _fixture())
    assert isinstance(result, GeneratedCodeResult)
    assert result.stages == ("parse", "allowlist", "compile", "execute")
    assert result.passed, result.failures


def test_syntax_error_fails_at_parse():
    result = verify_generated_code("def broken(:\n  pass", _fixture())
    assert not result.passed
    assert any("does not parse" in f for f in result.failures)


@pytest.mark.parametrize(
    ("snippet", "needle"),
    [
        ("import socket\n", "forbidden import: socket"),
        ("import subprocess\n", "forbidden import: subprocess"),
        ("from os import system\n", "forbidden import"),
        ("import urllib.request\n", "forbidden import"),
        ("eval('1+1')\n", "forbidden call: eval"),
        ("exec('x = 1')\n", "forbidden call: exec"),
        ("open('/etc/passwd')\n", "forbidden call: open"),
        ("__import__('socket')\n", "forbidden call: __import__"),
        ("getattr(object, 'x', None)\n", "forbidden call: getattr"),
        ("x = ().__class__\n", "forbidden dunder attribute"),
        ("b = __builtins__\n", "forbidden name: __builtins__"),
    ],
)
def test_prohibited_actions_are_rejected_statically(snippet, needle):
    result = verify_generated_code(snippet, _fixture())
    assert not result.passed
    assert any(needle in failure for failure in result.failures), result.failures


def test_runtime_poison_blocks_banned_module_even_if_statically_allowed(monkeypatch):
    # Defence in depth: even if the static allowlist were relaxed (or missed a
    # path), importing a poisoned module still fails inside the sandbox.
    monkeypatch.setattr(
        gc, "ALLOWED_IMPORTS", frozenset({*gc.ALLOWED_IMPORTS, "socket"})
    )
    result = verify_generated_code("import socket\n", _fixture())
    assert not result.passed
    assert any("exited" in f for f in result.failures)


def test_timeout_is_enforced():
    result = verify_generated_code(
        "while True:\n    pass\n", _fixture(), timeout=3.0
    )
    assert not result.passed
    assert any("timeout" in f for f in result.failures)


def test_pii_canary_in_source_is_reported():
    leaking = 'print("contact: tb.gen+leak@example.invalid")\n'
    result = verify_generated_code(leaking, _fixture())
    assert not result.passed
    assert any("generated source leaked canary" in f for f in result.failures)


def test_pii_canary_in_stdout_is_reported():
    code = (
        "import pandas as pd\n"
        'df = pd.read_csv("your_data.csv")\n'
        'print(df["memo"].tolist())\n'
    )
    result = verify_generated_code(code, _fixture())
    assert not result.passed
    assert any("stdout leaked canary" in f for f in result.failures)
    assert "example.invalid" not in result.stdout  # evidence itself is redacted


def test_input_file_overwrite_is_reported():
    code = (
        "import pandas as pd\n"
        'df = pd.read_csv("your_data.csv")\n'
        'df.head(1).to_csv("your_data.csv", index=False)\n'
    )
    result = verify_generated_code(code, _fixture())
    assert not result.passed
    assert any("modified its input file" in f for f in result.failures)
