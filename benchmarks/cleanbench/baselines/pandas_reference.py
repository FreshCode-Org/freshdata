"""CleanBench's public name for the pandas baseline (see ``run``/``AUTHORED_LINES``).

Thin re-export: the actual hand-written pandas cleaning script lives at
``benchmarks/baselines/pandas_baseline.py`` (shared with the legacy harness
in ``benchmarks/bench.py``) so there is exactly one authored-lines count to
keep honest.
"""

from __future__ import annotations

from benchmarks.baselines.pandas_baseline import AUTHORED_LINES, run

__all__ = ["AUTHORED_LINES", "run"]
