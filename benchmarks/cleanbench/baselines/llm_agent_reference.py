"""CleanBench's public name for the disclosed LLM-agent baseline.

Thin re-export of ``benchmarks/baselines/llm_agent_baseline.py``. Skipped by
default everywhere (including CI): set ``FRESHDATA_LLM_BASELINE=1`` plus
provider environment variables to run it. See that module's docstring for the
full disclosure rules — this is a benchmark-only comparison and is never part
of the FreshData runtime.
"""

from __future__ import annotations

from benchmarks.baselines.llm_agent_baseline import run

__all__ = ["run"]
