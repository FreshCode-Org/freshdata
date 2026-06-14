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
    path.write_text(json.dumps(normalize_report(report), indent=2, sort_keys=True) + "\n")
    return path
