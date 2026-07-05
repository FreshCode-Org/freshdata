"""CleanBench's public name for the Great Expectations baseline.

Thin re-export of ``benchmarks/baselines/great_expectations_baseline.py``: GE
validates but does not repair, so its output reports validation coverage and
manual-fix cost — never a fabricated repair score (see ``run``'s docstring).
"""

from __future__ import annotations

from benchmarks.baselines.great_expectations_baseline import run

__all__ = ["run"]
