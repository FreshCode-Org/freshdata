#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import freshdata as fd  # noqa: E402

def parse_steps(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

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
        p = Path(inp)
        if p.is_file() and p.suffix.lower() == ".csv":
            return [(p.stem, pd.read_csv(p))]
        if p.is_dir():
            return [(f.stem, pd.read_csv(f)) for f in sorted(p.glob("*.csv"))]
        return []
    sys.path.insert(0, str(ROOT))
    from tests.expectations import ALL_FIXTURES, load_fixture  # noqa: E402
    return [(str(name), load_fixture(name)) for name in ALL_FIXTURES]

def main() -> int:
    ap = argparse.ArgumentParser(description="Reusable FreshData automation runner")
    ap.add_argument("--input", help="CSV file path or directory of CSV files")
    ap.add_argument("--output-dir", default=str(ROOT / "scripts" / ".automation_out"))
    ap.add_argument("--steps", default="profile,clean", help="Comma-separated steps")
    ap.add_argument("--report", help="Optional summary report path")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = parse_steps(args.steps)
    datasets = collect_csvs(args.input)
    reports, ok, fail = [], 0, 0

    for name, df in datasets:
        t0 = time.perf_counter()
        rec = {"dataset": name, "steps": steps, "before_shape": list(df.shape), "after_shape": None, "seconds": None, "error": None}
        try:
            cleaned, step_meta = run_steps(df, steps)
            rec["after_shape"] = list(cleaned.shape)
            rec["step_meta"] = step_meta
            cleaned.to_csv(out_dir / f"{Path(name).stem}_cleaned.csv", index=False)
            ok += 1
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            fail += 1
        rec["seconds"] = round(time.perf_counter() - t0, 6)
        (out_dir / f"{Path(name).stem}_report.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        reports.append(rec)

    summary = {"total": len(reports), "ok": ok, "fail": fail, "steps": steps, "output_dir": str(out_dir), "reports": [f"{Path(r['dataset']).stem}_report.json" for r in reports]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
