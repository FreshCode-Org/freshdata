"""Regression tests: inclusive ``mostly`` boundaries and a deterministic plan hash."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise.contracts import (
    ColumnContract,
    DataContract,
    _mostly_tolerated,
    enforce_contract,
)
from freshdata.repairplan import _json_safe
from freshdata.validation_suite import (
    ColumnRule,
    CrossColumnRule,
    ValidationSuite,
    run_suite,
)

# (mostly, n_bad, n_total, tolerated)
BOUNDARY_CASES = [
    (0.9, 1, 10, True),
    (0.8, 2, 10, True),
    (0.95, 1, 20, True),
    (0.7, 3, 10, True),
    (0.9, 2, 10, False),
    (0.95, 2, 20, False),
]


def _col_frame(n_bad: int, n_total: int) -> pd.DataFrame:
    return pd.DataFrame({"v": [1] * (n_total - n_bad) + [-1] * n_bad})


def _cross_frame(n_bad: int, n_total: int) -> pd.DataFrame:
    lo = list(range(n_total))
    hi = [x + 1 for x in lo[: n_total - n_bad]] + [-1] * n_bad
    return pd.DataFrame({"lo": lo, "hi": hi})


# --------------------------------------------------------------------------- #
# mostly boundaries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("mostly", "n_bad", "n_total", "tolerated"), BOUNDARY_CASES)
def test_mostly_tolerated_helper(mostly, n_bad, n_total, tolerated):
    assert _mostly_tolerated(n_bad, n_total, mostly) is tolerated


def test_mostly_one_never_tolerates_and_large_frames_stay_exact():
    assert not _mostly_tolerated(1, 10, 1.0)
    assert not _mostly_tolerated(1, 10**9, 1.0)
    # One row past the boundary still fails on a billion-row frame.
    assert _mostly_tolerated(10**8, 10**9, 0.9)
    assert not _mostly_tolerated(10**8 + 1, 10**9, 0.9)


@pytest.mark.parametrize(("mostly", "n_bad", "n_total", "tolerated"), BOUNDARY_CASES)
def test_contract_value_check_mostly_boundary(mostly, n_bad, n_total, tolerated):
    df = _col_frame(n_bad, n_total)
    contract = DataContract(name="c", columns=[ColumnContract("v", min_value=0, mostly=mostly)])
    report = enforce_contract(df, contract)
    assert report.passed is tolerated
    statuses = {f.status for f in report.findings if f.check_id == "contract.min_value"}
    assert statuses == {"warned" if tolerated else "failed"}


@pytest.mark.parametrize(("mostly", "n_bad", "n_total", "tolerated"), BOUNDARY_CASES)
def test_suite_column_rule_mostly_boundary(mostly, n_bad, n_total, tolerated):
    df = _col_frame(n_bad, n_total)
    suite = ValidationSuite(name="s", rules=[ColumnRule("v", min_value=0, mostly=mostly)])
    assert run_suite(df, suite).passed is tolerated


@pytest.mark.parametrize(("mostly", "n_bad", "n_total", "tolerated"), BOUNDARY_CASES)
def test_suite_cross_column_mostly_boundary(mostly, n_bad, n_total, tolerated):
    df = _cross_frame(n_bad, n_total)
    rule = CrossColumnRule("lo", "<=", "hi", mostly=mostly)
    result = run_suite(df, ValidationSuite(name="s", cross_column=[rule]))
    assert result.passed is tolerated
    statuses = {f.status for f in result.report.findings if f.check_id == "suite.cross_column"}
    assert statuses == {"warned" if tolerated else "failed"}


def test_issue_reproduction_both_checks_warn():
    df = pd.DataFrame(
        {
            "start": range(10),
            "end": [i + 1 for i in range(9)] + [-1],
            "x": [1] * 9 + [500],
        }
    )
    cross = fd.validate(
        df,
        suite=fd.ValidationSuite(
            name="s", cross_column=[fd.CrossColumnRule("start", "<=", "end", mostly=0.9)]
        ),
    )
    rule = fd.ColumnRule("x", max_value=100, mostly=0.9)
    col = fd.validate(df, suite=fd.ValidationSuite(name="s", rules=[rule]))
    assert cross.passed
    assert col.passed


# --------------------------------------------------------------------------- #
# deterministic decisions_hash
# --------------------------------------------------------------------------- #

_HASH_SCRIPT = """
import pandas as pd, freshdata as fd
df = pd.DataFrame({"status": [" Active ", "INACTIVE", "pend-ing"]})
rp = fd.suggest_plan(df, context="Allowed status values are active, inactive, pending.",
                     semantic_mode="auto", verbose=False).repair_plan
rp.override(rp.actions[0].id, {"reviewers": {"alice", "bob", "carol", "dave", "erin", "frank"},
                               "tags": frozenset({"x", "y", "z", "w"})})
print(rp.decisions_hash())
print(rp.to_json())
"""


def _run_hash(seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    src_root = str(Path(fd.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src_root, env.get("PYTHONPATH", "")) if p)
    out = subprocess.run(
        [sys.executable, "-c", _HASH_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=300,
    )
    return out.stdout


def test_decisions_hash_independent_of_hash_seed():
    outputs = {_run_hash(seed) for seed in ("1", "2", "3", "4")}
    assert len(outputs) == 1
    text = outputs.pop()
    digest, _, payload = text.partition("\n")
    assert len(digest) == 64
    params = json.loads(payload)["actions"][0]["params"]
    assert params["reviewers"] == ["alice", "bob", "carol", "dave", "erin", "frank"]
    assert params["tags"] == ["w", "x", "y", "z"]


def _override_plan(value: object):
    df = pd.DataFrame({"status": [" Active ", "INACTIVE", "pend-ing"]})
    rp = fd.suggest_plan(
        df,
        context="Allowed status values are active, inactive, pending.",
        semantic_mode="auto",
        verbose=False,
    ).repair_plan
    rp.override(rp.actions[0].id, {"proposed_value": value})
    return rp


@pytest.mark.parametrize(
    ("np_value", "py_value"),
    [
        (np.int64(42), 42),
        (np.int32(-7), -7),
        (np.float32(0.5), 0.5),
        (np.float64(1.25), 1.25),
        (np.bool_(True), True),
        (np.str_("abc"), "abc"),
    ],
)
def test_numpy_scalars_hash_like_python_scalars(np_value, py_value):
    np_plan = _override_plan(np_value)
    py_plan = _override_plan(py_value)
    assert np_plan.decisions_hash() == py_plan.decisions_hash()
    assert np_plan.to_json() == py_plan.to_json()
    loaded = json.loads(np_plan.to_json())["actions"][0]["params"]["proposed_value"]
    assert loaded == py_value
    assert type(loaded) is type(py_value)


def test_json_safe_canonicalizes_unordered_and_numpy_inputs():
    assert _json_safe({3, 1, 2}) == [1, 2, 3]
    assert _json_safe(frozenset({"b", "a"})) == ["a", "b"]
    assert _json_safe({("b", 1), ("a", 2)}) == [["a", 2], ["b", 1]]
    assert _json_safe(np.array([1, 2])) == [1, 2]
    assert _json_safe(np.datetime64("2020-01-02")) == "2020-01-02T00:00:00"
    assert _json_safe(np.datetime64("NaT", "ns")) is None
    assert _json_safe(np.timedelta64(1, "D")) == "Timedelta('1 days 00:00:00')"
    # Unchanged behaviour for plain values.
    assert _json_safe((1, "a", None)) == [1, "a", None]
    assert _json_safe(pd.Timestamp("2020-01-02")) == "2020-01-02T00:00:00"
    assert _json_safe(float("nan")) != _json_safe(float("nan"))  # NaN passes through
    assert _json_safe(pd.NaT) is None
