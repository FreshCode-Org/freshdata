"""Report stand-ins for sensitive values must not be reversible by guessing."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import threading

import pandas as pd

import freshdata as fd
from freshdata import _util
from freshdata._util import mask_sensitive_value

TOKEN_RE = re.compile(r"\[SENSITIVE:[0-9a-f]{8}\]")
SECRET_SSN = "078-05-1120"
COMMON = {"return_report": True, "verbose": False}


def _unkeyed_token(value: object) -> str:
    """The token scheme used before the fix: ``sha256(repr(v))[:8]``."""
    return f"[SENSITIVE:{hashlib.sha256(repr(value).encode('utf-8')).hexdigest()[:8]}]"


def _report_tokens(report) -> set[str]:
    blob = repr(report.to_dict()) + repr(report.warnings) + repr(report.coerced_cells)
    return set(TOKEN_RE.findall(blob))


def test_dictionary_attack_finds_no_sensitive_value_in_the_report():
    df = pd.DataFrame({"ssn": [str(i) for i in range(19)] + [SECRET_SSN], "id": range(20)})
    _, report = fd.clean(df, sensitive_columns=("ssn",), **COMMON)

    tokens = _report_tokens(report)
    assert tokens  # the column really was masked; the attack has targets
    assert SECRET_SSN not in repr(report.to_dict())

    # An attacker who knows the format hashes a guess list that includes the
    # real value, plus the str() forms callers might pass.
    guesses = [f"078-05-{n:04d}" for n in range(1000, 1200)] + [SECRET_SSN]
    guesses += [str(i) for i in range(1000)]
    dictionary = {_unkeyed_token(g) for g in guesses}
    assert _unkeyed_token(SECRET_SSN) in dictionary
    assert tokens.isdisjoint(dictionary)


def test_same_value_gives_same_token_within_a_process():
    assert mask_sensitive_value(SECRET_SSN) == mask_sensitive_value(SECRET_SSN)
    assert mask_sensitive_value(SECRET_SSN) != mask_sensitive_value("078-05-1121")
    # repr() is still what gets hashed, so 1 and "1" stay distinct.
    assert mask_sensitive_value(1) != mask_sensitive_value("1")


def test_same_value_correlates_across_one_report():
    df = pd.DataFrame(
        {"ssn": [str(i) for i in range(38)] + [SECRET_SSN, SECRET_SSN], "id": range(40)}
    )
    _, report = fd.clean(df, sensitive_columns=("ssn",), **COMMON)
    # Both unparseable rows are named in one warning; the matching tokens show
    # they are the same value without revealing it.
    (warning,) = [w for w in report.warnings if "'ssn'" in w]
    assert SECRET_SSN not in warning
    assert TOKEN_RE.findall(warning) == [mask_sensitive_value(SECRET_SSN)] * 2


def test_child_process_gives_different_tokens():
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(fd.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src_root, env.get("PYTHONPATH")) if p)
    code = (
        "from freshdata._util import mask_sensitive_value as m; "
        f"print(m({SECRET_SSN!r})); print(m({SECRET_SSN!r}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    ).stdout.split()
    assert len(out) == 2 and out[0] == out[1]
    assert TOKEN_RE.fullmatch(out[0])
    assert out[0] != mask_sensitive_value(SECRET_SSN)
    assert out[0] != _unkeyed_token(SECRET_SSN)


def test_token_format_and_length_are_unchanged():
    for value in (SECRET_SSN, 42, None, 3.5, ("a", 1), "", "ü"):
        token = mask_sensitive_value(value)
        assert TOKEN_RE.fullmatch(token)
        assert len(token) == len("[SENSITIVE:]") + 8 == len(_unkeyed_token(value))


def test_concurrent_first_use_shares_one_key(monkeypatch):
    monkeypatch.setattr(_util, "_SENSITIVE_TOKEN_KEY", None)
    barrier = threading.Barrier(8)
    seen: list[str] = []

    def worker() -> None:
        barrier.wait()
        seen.append(mask_sensitive_value(SECRET_SSN))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 8 and len(set(seen)) == 1
    key = _util._SENSITIVE_TOKEN_KEY
    assert isinstance(key, bytes) and len(key) == 32
