"""Golden report snapshot helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import freshdata as fd
from expectations import FIXTURES_DIR, ONLINE_DIR

GOLDEN_DIR = FIXTURES_DIR / "golden"
ONLINE_GOLDEN_DIR = ONLINE_DIR / "golden"
GOLDEN_DIFF_SUMMARY_PATH = FIXTURES_DIR / "golden_diff_summary.jsonl"


def normalize_report(report: fd.CleanReport) -> dict[str, Any]:
    """Stable dict for snapshot comparison (strip timing/memory noise)."""
    payload = report.to_dict()
    for key in ("duration_seconds", "memory_before", "memory_after"):
        payload.pop(key, None)
    for action in payload.get("actions", []):
        action["confidence"] = round(action["confidence"], 4)
        if "description" in action and isinstance(action["description"], str):
            desc = action["description"]
            desc = re.sub(
                r"datetime64\[(us|s|ms|ns|M|D|h|m),\s*UTC\]",
                "datetime64[ns, UTC]",
                desc,
            )
            desc = re.sub(r"datetime64\[(us|s|ms|ns|M|D|h|m)\]", "datetime64[ns]", desc)
            action["description"] = desc
    return payload



def golden_path(fixture_name: str, strategy: str = "balanced", *, online: bool = False) -> Path:
    base = ONLINE_GOLDEN_DIR if online else GOLDEN_DIR
    return base / f"{fixture_name}.{strategy}.report.json"


def load_golden(
    fixture_name: str, strategy: str = "balanced", *, online: bool = False
) -> dict[str, Any]:
    path = golden_path(fixture_name, strategy, online=online)
    return json.loads(path.read_text())


def write_golden(
    fixture_name: str,
    report: fd.CleanReport,
    strategy: str = "balanced",
    *,
    online: bool = False,
) -> Path:
    path = golden_path(fixture_name, strategy, online=online)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_report(report)
    previous = json.loads(path.read_text()) if path.exists() else None
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    diff_summary = {
        "fixture": fixture_name,
        "strategy": strategy,
        "online": online,
        "created": previous is None,
        "changed": previous != normalized,
        "previous_action_count": len(previous.get("actions", [])) if isinstance(previous, dict) else 0,
        "new_action_count": len(normalized.get("actions", [])),
    }
    GOLDEN_DIFF_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_DIFF_SUMMARY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(diff_summary, sort_keys=True) + "\n")
    return path
