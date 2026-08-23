"""Run freshdata core API against 50+ fixture datasets; print JSON summary."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import freshdata as fd  # noqa: E402
from expectations import (  # noqa: E402
    ALL_FIXTURES,
    ALL_ONLINE_FIXTURES,
    load_fixture,
    load_online_fixture,
)

EXTERNAL_DATASETS = {
    "spotify_114k": Path("/Users/kevincostner/Desktop/data science/dataset.csv"),
    "retail_dirty": Path(
        "/Users/kevincostner/Desktop/data science/synthetic_eda_dataset/retail_customers_dirty.csv"
    ),
    "retail_clean": Path(
        "/Users/kevincostner/Desktop/data science/synthetic_eda_dataset/retail_customers_clean.csv"
    ),
}

STEPS = ("profile", "clean", "suggest_plan", "detect_outliers", "remove_outliers", "fill_missing")


def run_step(df: pd.DataFrame, step: str) -> dict:
    t0 = time.perf_counter()
    try:
        out = getattr(fd, step)(df)
        info: dict = {"ok": True, "sec": round(time.perf_counter() - t0, 3)}
        if step == "clean":
            info["rows_out"] = len(out)
        elif step == "detect_outliers":
            info["flagged"] = int(out.sum())
        elif step == "remove_outliers":
            info["rows_out"] = len(out)
        return info
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "sec": round(time.perf_counter() - t0, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=3),
        }


def main() -> None:
    datasets: list[tuple[str, str, pd.DataFrame | None]] = []

    for name in ALL_FIXTURES:
        datasets.append((name, "local", None))
    for name in ALL_ONLINE_FIXTURES:
        datasets.append((name, "online", None))
    for name, path in EXTERNAL_DATASETS.items():
        if path.exists():
            datasets.append((f"external:{name}", "external", pd.read_csv(path)))

    results: list[dict] = []
    totals = {"ok": 0, "fail": 0}

    for name, kind, preloaded in datasets:
        if preloaded is not None:
            df = preloaded
        elif kind == "local":
            df = load_fixture(name)
        else:
            df = load_online_fixture(name)

        bool_cols = [c for c in df.columns if str(df[c].dtype) == "bool"]
        row: dict = {
            "dataset": name,
            "kind": kind,
            "shape": list(df.shape),
            "bool_cols": bool_cols,
            "steps": {},
        }

        for step in STEPS:
            outcome = run_step(df, step)
            row["steps"][step] = outcome
            totals["ok" if outcome["ok"] else "fail"] += 1

        results.append(row)

    summary = {
        "datasets": len(results),
        "step_ok": totals["ok"],
        "step_fail": totals["fail"],
        "failed": [
            {"dataset": r["dataset"], "step": s, "error": r["steps"][s].get("error")}
            for r in results
            for s, o in r["steps"].items()
            if not o["ok"]
        ],
    }
    payload = {"summary": summary, "results": results}
    out_path = ROOT / "scripts" / ".dataset_matrix_report.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
