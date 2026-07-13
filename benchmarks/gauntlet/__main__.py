"""Validation Gauntlet CLI.

Run from the repo root::

    python -m benchmarks.gauntlet run                 # run + write results/
    python -m benchmarks.gauntlet run --check         # also gate (CI mode)
    python -m benchmarks.gauntlet run --update-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .fixtures import DEFAULT_ROWS, DEFAULT_SEED, FIXTURES
from .metrics import compute_metrics
from .report import check_gates, render_markdown, results_payload, write_json
from .runner import run_gauntlet

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_PATH = Path(__file__).parent / "baseline.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.gauntlet")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run the gauntlet and write JSON + Markdown")
    run_p.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    run_p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run_p.add_argument("--fixtures", nargs="*", choices=sorted(FIXTURES))
    run_p.add_argument("--check", action="store_true",
                       help="exit 1 when a gate fails or the baseline regresses")
    run_p.add_argument("--update-baseline", action="store_true",
                       help="write this run as the stored baseline")
    args = parser.parse_args(argv)

    runs = run_gauntlet(n_rows=args.rows, seed=args.seed, fixtures=args.fixtures)
    metrics = {name: compute_metrics(r) for name, r in runs.items()}
    payload = results_payload(metrics, n_rows=args.rows, seed=args.seed)

    write_json(payload, RESULTS_DIR / "gauntlet.json")
    markdown = render_markdown(payload)
    (RESULTS_DIR / "gauntlet.md").write_text(markdown)
    print(markdown)
    print(f"results: {RESULTS_DIR / 'gauntlet.json'}")

    if args.update_baseline:
        write_json(payload, BASELINE_PATH)
        print(f"baseline updated: {BASELINE_PATH}")

    if args.check:
        baseline = (json.loads(BASELINE_PATH.read_text())
                    if BASELINE_PATH.exists() else None)
        problems = check_gates(payload, baseline)
        if problems:
            print("\nGATE FAILURES:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print("all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
