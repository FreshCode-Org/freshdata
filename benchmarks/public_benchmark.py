#!/usr/bin/env python
"""Public benchmark refresh: one reproducible report for sharing numbers.

Runs the two existing harnesses in subprocesses and folds their output into a
single markdown + JSON report stamped with the freshdata version and the
environment it ran on (no numbers are ever committed; see docs/benchmarks.md):

* ``benchmarks/bench_report.py`` — the scaling cases behind the
  "Strategic-report scaling benchmarks" table (CSV ingest, profile, null-fill,
  import time, peak memory), balanced vs aggressive;
* ``benchmarks/bench.py run`` + ``report`` — the per-fixture quality/speed table.

Usage::

    python benchmarks/public_benchmark.py                   # CI-safe sizes
    python benchmarks/public_benchmark.py --scale full      # harness defaults (10M rows)
    python benchmarks/public_benchmark.py --skip-scaling --repeat 1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS_DIR = HERE / "results"
SCALING_JSON = RESULTS_DIR / "report_bench.json"

#: CLI presets. ``ci`` fits a 2-core / 7 GB GitHub runner; ``full`` uses the
#: harness defaults (100 MB CSV, 10M-row null-fill) for a workstation.
SCALES: dict[str, dict[str, int | None]] = {
    "ci": {"csv_mb": 50, "rows": 1_000_000},
    "full": {"csv_mb": 100, "rows": None},
}

_SCALING_LABELS = {
    "csv_ingest": "CSV ingest + clean",
    "profile": "Mixed-schema profile",
    "nullfill": "Null-fill / flag",
    "import_time": "Import time (`import freshdata`)",
    "memory": "Peak memory of a full clean",
}

Runner = Callable[[list[str]], None]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)  # noqa: S603 - fixed argv


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


def collect_environment() -> dict[str, Any]:
    import freshdata  # noqa: PLC0415 - report the version actually benchmarked

    ram = None
    try:
        import psutil  # noqa: PLC0415

        ram = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:  # pragma: no cover - psutil is a core dependency
        pass
    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            sha = "unknown"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "freshdata_version": freshdata.__version__,
        "git_sha": sha,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": ram,
        "packages": {
            name: _version(name) for name in ("pandas", "numpy", "pyarrow", "polars", "duckdb")
        },
    }


# --------------------------------------------------------------------------- #
# harnesses
# --------------------------------------------------------------------------- #

def run_scaling(*, csv_mb: int, rows: int | None, runner: Runner = _run) -> list[dict[str, Any]]:
    cmd = [sys.executable, "benchmarks/bench_report.py", "all", "--mb", str(csv_mb)]
    if rows is not None:
        cmd += ["--rows", str(rows)]
    runner(cmd)
    return json.loads(SCALING_JSON.read_text(encoding="utf-8"))["results"]


def run_fixtures(*, repeat: int, size: int | None, runner: Runner = _run) -> str:
    before = {p.name for p in RESULTS_DIR.iterdir()} if RESULTS_DIR.exists() else set()
    cmd = [sys.executable, "benchmarks/bench.py", "run", "--repeat", str(repeat)]
    if size is not None:
        cmd += ["--size", str(size)]
    runner(cmd)
    new_runs = sorted(
        p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name not in before
    )
    if not new_runs:
        raise RuntimeError("bench.py run did not create a results directory")
    run_dir = new_runs[-1]
    runner([sys.executable, "benchmarks/bench.py", "report", "--run-dir", str(run_dir)])
    return (run_dir / "report.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _cell(result: dict[str, Any] | None) -> str:
    if result is None:
        return "n/a"
    if "import_sec_median" in result:
        median, fastest = result["import_sec_median"], result.get("import_sec_min", 0)
        return f"{median:.3f} s median ({fastest:.3f} s min)"
    parts = []
    if "wall_sec" in result:
        parts.append(f"{result['wall_sec']:.2f} s")
    if result.get("peak_rss_mb") is not None:
        parts.append(f"{result['peak_rss_mb']:.0f} MB peak RSS")
    return ", ".join(parts) or "measured"


def _size(results: list[dict[str, Any]]) -> str:
    first = results[0]
    if first.get("csv_mb"):
        return f"{first['csv_mb']} MB"
    if first.get("rows"):
        return f"{first['rows']:,} rows"
    return ""


def scaling_table(results: list[dict[str, Any]]) -> list[str]:
    lines = ["| Benchmark | Size | Balanced | Aggressive |", "|---|---|---|---|"]
    for key, label in _SCALING_LABELS.items():
        rows = [r for r in results if r.get("benchmark") == key]
        if not rows:
            continue
        by_strategy = {r.get("strategy"): r for r in rows}
        balanced = by_strategy.get("balanced") or by_strategy.get("n/a")
        aggressive = by_strategy.get("aggressive")
        lines.append(f"| {label} | {_size(rows)} | {_cell(balanced)} | {_cell(aggressive)} |")
    return lines


def render_markdown(
    environment: dict[str, Any],
    scaling: list[dict[str, Any]] | None,
    fixture_report: str | None,
) -> str:
    packages = ", ".join(f"{k} {v}" for k, v in environment["packages"].items())
    lines = [
        f"# FreshData public benchmark — freshdata {environment['freshdata_version']}",
        "",
        f"- **freshdata:** `{environment['freshdata_version']}` "
        f"(commit `{environment['git_sha'][:12]}`)",
        f"- **Generated:** {environment['generated_at']}",
        f"- **Hardware:** {environment['cpu_count']} CPUs, {environment['ram_gb']} GB RAM",
        f"- **Software:** Python {environment['python']} on {environment['platform']}; {packages}",
        "",
        "Numbers are environment-specific. Re-run `python benchmarks/public_benchmark.py` "
        "on your own hardware before quoting them.",
        "",
    ]
    if scaling:
        lines += ["## Strategic-report scaling benchmarks", "", *scaling_table(scaling), ""]
    if fixture_report:
        body = fixture_report
        if body.startswith("# "):
            body = body.split("\n", 1)[1]  # drop the harness report's own title
        lines += ["## Fixture benchmarks (`benchmarks/bench.py`)", "", body.strip(), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, runner: Runner = _run) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scale", choices=sorted(SCALES), default="ci",
                        help="size preset for the scaling cases (default: ci)")
    parser.add_argument("--csv-mb", type=int, default=None, help="override the CSV ingest size")
    parser.add_argument("--rows", type=int, default=None,
                        help="override profile/null-fill/memory row counts")
    parser.add_argument("--repeat", type=int, default=5, help="bench.py timing repeats")
    parser.add_argument("--size", type=int, default=None, help="bench.py row-count override")
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--skip-fixtures", action="store_true")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "public"),
                        help="where public-benchmark.md/json are written")
    args = parser.parse_args(argv)
    if args.skip_scaling and args.skip_fixtures:
        parser.error("nothing to run: both --skip-scaling and --skip-fixtures given")

    preset = SCALES[args.scale]
    csv_mb = args.csv_mb if args.csv_mb is not None else preset["csv_mb"]
    rows = args.rows if args.rows is not None else preset["rows"]

    environment = collect_environment()
    scaling = None
    if not args.skip_scaling:
        scaling = run_scaling(csv_mb=int(csv_mb), rows=rows, runner=runner)
    fixture_report = None if args.skip_fixtures else run_fixtures(
        repeat=args.repeat, size=args.size, runner=runner)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(environment, scaling, fixture_report)
    (out_dir / "public-benchmark.md").write_text(markdown, encoding="utf-8")
    (out_dir / "public-benchmark.json").write_text(json.dumps({
        "environment": environment,
        "options": {"scale": args.scale, "csv_mb": csv_mb, "rows": rows,
                    "repeat": args.repeat, "size": args.size},
        "scaling": scaling,
        "fixture_report_markdown": fixture_report,
    }, indent=2) + "\n", encoding="utf-8")
    print(markdown)
    print(f"written: {out_dir / 'public-benchmark.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
