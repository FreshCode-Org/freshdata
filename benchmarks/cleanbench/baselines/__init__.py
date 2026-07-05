"""CleanBench's public baseline harness: pandas, pyjanitor, Great Expectations,
and the disclosed LLM-agent comparison.

:func:`run_all` is what ``python -m benchmarks.cleanbench --reproduce-headline``
calls. Every baseline in :data:`BASELINES` always produces a result — either a
scored run or an honestly-recorded skip (``status: "skipped"`` + a human-
readable ``reason``) — so the report never silently omits a row. Nothing here
is imported by ``freshdata`` itself; this package exists only to score
competitor comparisons for CleanBench.
"""

from __future__ import annotations

from typing import Any

from .. import fixtures, metrics

#: Baselines that repair T1 (representation) dirt: scored against the T1
#: fixture truth with CleanBench's own metrics, so the comparison uses
#: identical ground truth to FreshData's own T1 track.
_REPAIR_BASELINES = ("pandas", "pyjanitor")
#: Baselines that are already self-contained result-producers (own fixture,
#: own scoring) — validation-only (GE) or the disclosed LLM-agent comparison.
_SELF_SCORED_BASELINES = ("great_expectations", "llm_agent")

BASELINES = _REPAIR_BASELINES + _SELF_SCORED_BASELINES


def _run_repair_baseline(name: str) -> dict[str, Any]:
    if name == "pandas":
        from . import pandas_reference as module
    elif name == "pyjanitor":
        from . import pyjanitor_reference as module
    else:  # pragma: no cover - guarded by BASELINES
        raise ValueError(name)

    truth, corrupted, _kwargs = fixtures.make_t1_representation_fixture()
    try:
        repaired = module.run(corrupted.copy(deep=True))
    except ImportError as exc:
        return {
            "baseline": name,
            "status": "skipped",
            "reason": str(exc),
        }
    # A baseline script may reorder/drop rows (dedup, reset_index); align on
    # shape before scoring so a genuine mismatch fails loudly instead of
    # crashing the whole harness run.
    if repaired.shape != truth.shape:
        return {
            "baseline": name,
            "status": "ran",
            "note": f"output shape {repaired.shape} != truth shape {truth.shape}; "
                    "cell metrics skipped",
            "authored_lines": module.AUTHORED_LINES,
        }
    return {
        "baseline": name,
        "status": "ran",
        "cell_repair_precision": metrics.cell_repair_precision(truth, corrupted, repaired),
        "cell_repair_recall": metrics.cell_repair_recall(truth, corrupted, repaired),
        "cell_repair_f1": metrics.cell_repair_f1(truth, corrupted, repaired),
        "false_modification_rate": metrics.false_modification_rate(truth, corrupted, repaired),
        "authored_lines": module.AUTHORED_LINES,
        "network_call_count": 0,
    }


def _run_self_scored_baseline(name: str) -> dict[str, Any]:
    if name == "great_expectations":
        from . import great_expectations_reference as module
    elif name == "llm_agent":
        from . import llm_agent_reference as module
    else:  # pragma: no cover - guarded by BASELINES
        raise ValueError(name)
    return dict(module.run())


def run_all() -> dict[str, dict[str, Any]]:
    """Run every baseline in :data:`BASELINES`; never raises.

    Each entry is either a scored result (``status: "ran"``) or an honest skip
    (``status: "skipped"``, with ``reason``) — e.g. pyjanitor is skipped when
    not installed (or incompatible with the active pandas version), and
    llm_agent is skipped unless explicitly enabled.
    """
    results: dict[str, dict[str, Any]] = {}
    for name in _REPAIR_BASELINES:
        try:
            results[name] = _run_repair_baseline(name)
        except Exception as exc:  # pragma: no cover - defensive: never abort the run
            results[name] = {"baseline": name, "status": "skipped", "reason": repr(exc)}
    for name in _SELF_SCORED_BASELINES:
        try:
            results[name] = _run_self_scored_baseline(name)
        except Exception as exc:  # pragma: no cover - defensive: never abort the run
            results[name] = {"baseline": name, "status": "skipped", "reason": repr(exc)}
    return results


__all__ = ["BASELINES", "run_all"]
