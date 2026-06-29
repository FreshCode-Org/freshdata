"""Lightweight import-cost regression guards.

These keep ``import freshdata`` cheap and ensure the base import never pulls in
heavy optional dependencies (visualization, ML, backends). They are deliberately
generous so they don't flake on slow/constrained CI, and the timing one can be
skipped via ``FRESHDATA_SKIP_PERF=1``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_base_import_does_not_pull_heavy_deps() -> None:
    """A bare ``import freshdata`` must not import viz/ML/backend libraries."""
    code = (
        "import sys, freshdata\n"
        "heavy = [m for m in "
        "('plotly','itables','great_tables','anywidget','sklearn',"
        "'polars','duckdb','pyspark','presidio_analyzer') "
        "if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    leaked = [m for m in out.stdout.strip().split(",") if m]
    assert not leaked, f"base import unexpectedly imported: {leaked}"


@pytest.mark.skipif(
    os.environ.get("FRESHDATA_SKIP_PERF") == "1",
    reason="perf guard disabled via FRESHDATA_SKIP_PERF=1",
)
def test_import_time_budget() -> None:
    """Cold import stays under a generous budget (median of 3 subprocesses)."""
    import time

    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", "import freshdata"],
            check=True, capture_output=True,
        )
        samples.append(time.perf_counter() - t0)
    samples.sort()
    median = samples[len(samples) // 2]
    # 3s is well above a healthy ~0.5-0.8s cold import but won't flake on CI.
    assert median < 3.0, f"import freshdata too slow: {median:.2f}s (median)"
