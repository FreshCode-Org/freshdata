#!/usr/bin/env python3
"""Batch profile + clean runner for local fixtures and CSV inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

import freshdata as fd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from expectations import ALL_FIXTURES, load_fixture  # noqa: E402


def parse_steps(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def run_steps(df: pd.DataFrame, steps: list[str]) -> tuple[pd.DataFrame, dict]:
    """Run freshdata steps; only mutating steps update the working frame."""
    out = df.copy()
    meta: dict = {}
    for name in steps:
        fn = getattr(fd, name, None)
        if not callable(fn):
            raise ValueError(f"unknown step: {name}")
        t0 = time.perf_counter()
        if name == "profile":
            prof = fn(out)
            meta["profile"] = {
                "columns": len(getattr(prof, "columns", prof)),
                "rows": len(out),
            }
        elif name == "clean":
            result = fn(out, return_report=True)
            out = result[0] if isinstance(result, tuple) else result
            if isinstance(result, tuple) and len(result) > 1:
                report = result[1]
                meta["clean"] = {
                    "actions": len(getattr(report, "actions", report)),
                }
        else:
            result = fn(out)
            out = result[0] if isinstance(result, tuple) else result
        meta.setdefault("timing", {})[name] = round(time.perf_counter() - t0, 4)
    return out, meta


def collect_csvs(inp: str | None) -> list[tuple[str, pd.DataFrame]]:
    if inp:
        path = Path(inp)
        if path.is_file() and path.suffix.lower() == ".csv":
            return [(path.stem, pd.read_csv(path))]
        if path.is_dir():
            return [(item.stem, pd.read_csv(item)) for item in sorted(path.glob("*.csv"))]
        return []
    return [(str(name), load_fixture(name)) for name in ALL_FIXTURES]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reusable FreshData automation runner")
    parser.add_argument("--input", help="CSV file path or directory of CSV files")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "scripts" / ".automation_out"),
    )
    parser.add_argument("--steps", default="profile,clean", help="Comma-separated steps")
    parser.add_argument("--report", help="Optional summary report path")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = parse_steps(args.steps)
    datasets = collect_csvs(args.input)
    reports: list[dict] = []
    ok = fail = 0

    for name, df in datasets:
        t0 = time.perf_counter()
        rec = {
            "dataset": name,
            "steps": steps,
            "before_shape": list(df.shape),
            "after_shape": None,
            "seconds": None,
            "error": None,
        }
        try:
            cleaned, step_meta = run_steps(df, steps)
            rec["after_shape"] = list(cleaned.shape)
            rec["step_meta"] = step_meta
            cleaned.to_csv(out_dir / f"{Path(name).stem}_cleaned.csv", index=False)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            fail += 1
        rec["seconds"] = round(time.perf_counter() - t0, 6)
        report_path = out_dir / f"{Path(name).stem}_report.json"
        report_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        reports.append(rec)

    report_names = [f"{Path(row['dataset']).stem}_report.json" for row in reports]
    summary = {
        "total": len(reports),
        "ok": ok,
        "fail": fail,
        "steps": steps,
        "output_dir": str(out_dir),
        "reports": report_names,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.report:
        report_out = Path(args.report)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
