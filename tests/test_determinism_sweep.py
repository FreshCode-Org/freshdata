"""The same semantic input must give the same semantic output.

Determinism was thin at the public-API level: ``tests/test_properties.py:11``
is two calls on one fixture, and ``PYTHONHASHSEED`` was set at exactly one site
(``test_plan_hash_and_mostly.py:134``, for repair-plan hashing only). Nothing
swept the hash seed across ``fd.clean`` itself, even though set and dict
iteration order feeds category normalisation, dominant-variant selection and
action ordering.

What is compared: cell values, dtypes, and every action's step, column, count,
risk and status, plus ``report.decisions_hash``. Timing and memory fields are
excluded deliberately -- they are documented as volatile, and
``report.peak_memory`` is process-lifetime RSS, which cannot be reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

import freshdata as fd

#: Documented volatile report fields; see CleanReport and the TruthBench
#: APPROVED_TELEMETRY list. Excluding them is not a weakening: including them
#: would assert that a wall clock is reproducible.
VOLATILE = {
    "duration_seconds",
    "peak_memory",
    "rows_per_second",
    "created_at",
    "memory_before",
    "memory_after",
    "memory_bytes",
    "profiled_at",
    "stage_timings",
}


def _frame() -> pd.DataFrame:
    """A frame that reaches dtype repair, sentinels, currency and semantics."""
    return pd.DataFrame(
        {
            "row_key": [f"r{i}" for i in range(12)],
            "customer_id": [f"{i:03d}" for i in range(1, 13)],
            "amount": [
                10.5, 20.0, "  30.5 ", 40.0, 50.0, "N/A",
                70.0, 80.0, 90.0, 100.0, "$1,200.50", 120.0,
            ],
            "country": ["US", "GB", "FR", "DE", "JP", "CA", "AU", "BR", "IN", "US", "GB", "FR"],
            "note": list("abcdefghijkl"),
        }
    )


def _digest(out: pd.DataFrame, report) -> str:
    payload = {
        "values": [[str(v) for v in row] for row in out.itertuples(index=False)],
        "dtypes": [str(t) for t in out.dtypes],
        "actions": sorted(
            (a.step, str(a.column), a.count, a.risk, a.status) for a in report.actions
        ),
        "decisions_hash": getattr(report, "decisions_hash", None),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def test_one_hundred_runs_agree():
    """Repeated cleaning of equal input must not drift."""
    digests = set()
    for _ in range(100):
        out, report = fd.clean(_frame(), verbose=False, semantic_mode="auto", return_report=True)
        digests.add(_digest(out, report))
    assert len(digests) == 1, f"{len(digests)} distinct results across 100 runs"


def test_the_report_payload_is_stable_apart_from_documented_volatile_fields():
    def stable(report):
        return {k: v for k, v in report.to_dict().items() if k not in VOLATILE}

    _, first = fd.clean(_frame(), verbose=False, semantic_mode="auto", return_report=True)
    _, second = fd.clean(_frame(), verbose=False, semantic_mode="auto", return_report=True)
    assert stable(first) == stable(second)


_CHILD = textwrap.dedent(
    """
    import hashlib, json, warnings
    warnings.filterwarnings("ignore")
    import pandas as pd, freshdata as fd
    df = pd.DataFrame({
        "row_key": [f"r{i}" for i in range(12)],
        "customer_id": [f"{i:03d}" for i in range(1, 13)],
        "amount": [10.5, 20.0, "  30.5 ", 40.0, 50.0, "N/A",
                   70.0, 80.0, 90.0, 100.0, "$1,200.50", 120.0],
        "country": ["US","GB","FR","DE","JP","CA","AU","BR","IN","US","GB","FR"],
        "note": list("abcdefghijkl"),
    })
    out, rep = fd.clean(df, verbose=False, semantic_mode="auto", return_report=True)
    payload = {
        "values": [[str(v) for v in row] for row in out.itertuples(index=False)],
        "dtypes": [str(t) for t in out.dtypes],
        "actions": sorted((a.step, str(a.column), a.count, a.risk, a.status)
                          for a in rep.actions),
        "decisions_hash": getattr(rep, "decisions_hash", None),
    }
    print(hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest())
    """
)


@pytest.mark.parametrize("seed", ["0", "1", "42", "31337", "65535"])
def test_the_result_does_not_depend_on_pythonhashseed(seed, tmp_path):
    """Set/dict iteration order must not reach a cleaning decision.

    Run in a subprocess because PYTHONHASHSEED is fixed at interpreter start.
    """
    script = tmp_path / "child.py"
    script.write_text(_CHILD)

    def run(hash_seed: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, env=env, timeout=300, check=True,
        )
        return proc.stdout.strip()

    assert run(seed) == run("0"), f"PYTHONHASHSEED={seed} changed the result"
