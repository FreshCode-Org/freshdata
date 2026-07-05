"""CleanBench's public name for the pyjanitor baseline.

Thin re-export of ``benchmarks/baselines/pyjanitor_baseline.py``. ``run``
raises ``ImportError`` when pyjanitor is not installed (or is installed but
incompatible with the active pandas version); the harness in
:mod:`benchmarks.cleanbench.baselines` catches that and records an honest
skip reason rather than failing the whole run.
"""

from __future__ import annotations

from benchmarks.baselines.pyjanitor_baseline import AUTHORED_LINES, run

__all__ = ["AUTHORED_LINES", "run"]
