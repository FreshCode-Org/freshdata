"""Disclosed LLM-agent baseline — benchmark only, never part of the runtime.

Rules (enforced):

- runs only when ``FRESHDATA_LLM_BASELINE=1`` **and** provider settings are
  present in the environment; CI never sets these, so it is skipped by
  default everywhere;
- runs the benchmark **three times** and reports per-run cell accuracy so
  determinism (or lack of it) is measured, not assumed;
- full disclosure in the output: model, provider, date, and token cost;
- only synthetic fixture data is ever sent; API keys come from the
  environment and are never written anywhere;
- every call is counted (``network_call_count``) and the protected-column /
  false-modification rates are measured against the same truth used for the
  FreshData T2 track, so the comparison is apples-to-apples.

Run: ``FRESHDATA_LLM_BASELINE=1 FRESHDATA_TEACHER_URL=... \\
      FRESHDATA_TEACHER_PROVIDER=... FRESHDATA_TEACHER_MODEL=... \\
      FRESHDATA_TEACHER_API_KEY=... [FRESHDATA_TEACHER_COST_USD=...] \\
      python benchmarks/baselines/llm_agent_baseline.py``
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanbench import make_t2_semantic_fixture  # noqa: E402
from cleanbench.metrics import (  # noqa: E402
    cell_repair_f1,
    determinism_score,
    false_modification_rate,
    protected_column_violation_rate,
)

REPEATS = 3
#: T2's untouched column: a real protected-column check for the comparison.
PROTECTED_COLUMNS = ["monthly_revenue"]


def _enabled() -> bool:
    return os.environ.get("FRESHDATA_LLM_BASELINE", "") == "1"


def _call_llm(prompt: str) -> str:
    url = os.environ["FRESHDATA_TEACHER_URL"]
    body = json.dumps({
        "model": os.environ["FRESHDATA_TEACHER_MODEL"],
        "prompt": prompt,
        "schema": "csv",
    }).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['FRESHDATA_TEACHER_API_KEY']}",
    }, method="POST")
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read().decode("utf-8")


def run() -> dict[str, object]:
    if not _enabled():
        return {
            "baseline": "llm_agent",
            "status": "skipped",
            "reason": "set FRESHDATA_LLM_BASELINE=1 plus provider env vars to run "
                      "(never enabled in CI; benchmark-only, isolated from runtime)",
            "provider": None,
            "model": None,
            "date": None,
            "cost_usd": None,
            "runs": 0,
            "determinism_score": None,
            "protected_column_violation_rate": None,
            "false_modification_rate": None,
            "network_call_count": 0,
            "notes": "not run; the FreshData runtime never calls an LLM regardless",
        }
    import io  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    truth, corrupted, _ = make_t2_semantic_fixture()
    prompt = (
        "Clean this CSV: fix email formatting, normalize Indian phone numbers to "
        "+91XXXXXXXXXX, and normalize status to one of active/inactive/pending. "
        "Return ONLY the corrected CSV with the same columns and row order.\n\n"
        + corrupted.to_csv(index=False)
    )
    scores = []
    protected_violations = []
    false_mods = []
    network_calls = 0
    for _ in range(REPEATS):
        response = _call_llm(prompt)
        network_calls += 1
        repaired = pd.read_csv(io.StringIO(response), dtype=str)
        if repaired.shape != corrupted.shape:
            scores.append(0.0)
            continue
        scores.append(cell_repair_f1(truth, corrupted, repaired))
        protected_violations.append(
            protected_column_violation_rate(corrupted, repaired, PROTECTED_COLUMNS)
        )
        false_mods.append(false_modification_rate(truth, corrupted, repaired))
    cost_usd = os.environ.get("FRESHDATA_TEACHER_COST_USD")
    return {
        "baseline": "llm_agent",
        "status": "ran",
        "provider": os.environ.get("FRESHDATA_TEACHER_PROVIDER"),
        "model": os.environ.get("FRESHDATA_TEACHER_MODEL"),
        "date": datetime.date.today().isoformat(),
        "cost_usd": float(cost_usd) if cost_usd else None,
        "runs": REPEATS,
        "cell_repair_f1_per_run": [round(s, 4) for s in scores],
        "determinism_score": determinism_score(scores),
        "protected_column_violation_rate": (
            max(protected_violations) if protected_violations else None
        ),
        "false_modification_rate": max(false_mods) if false_mods else None,
        "network_call_count": network_calls,
        "notes": "benchmark-only; the FreshData runtime never calls an LLM. "
                 "cost_usd is None unless FRESHDATA_TEACHER_COST_USD is set.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
