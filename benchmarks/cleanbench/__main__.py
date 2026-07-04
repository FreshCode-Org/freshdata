"""CLI: ``python -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site``."""

from __future__ import annotations

import argparse
import sys

from .report import update_docs_site, write_results
from .runner import ALL_TRACKS, run_full
from .tasks import build_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.cleanbench")
    parser.add_argument("--tracks", default=",".join(ALL_TRACKS),
                        help="comma-separated subset of T1,T2,T3,T4,T5")
    parser.add_argument("--report", choices=("json", "md", "site"), default="md",
                        help="'site' additionally refreshes docs/benchmarks.md")
    parser.add_argument("--target-rows", type=int, default=50_000, help="T5 frame size")
    parser.add_argument("--update-baseline", action="store_true",
                        help="re-pin the T5 perf baseline to this run")
    parser.add_argument("--build-tasks", action="store_true",
                        help="regenerate benchmarks/cleanbench/tasks/ and exit")
    parser.add_argument("--check-gates", action="store_true",
                        help="exit non-zero when any release gate fails")
    args = parser.parse_args(argv)

    if args.build_tasks:
        for path in build_all():
            print(f"task written: {path}")
        return 0

    tracks = tuple(t.strip().upper() for t in args.tracks.split(",") if t.strip())
    unknown = [t for t in tracks if t not in ALL_TRACKS]
    if unknown:
        parser.error(f"unknown tracks: {unknown}; valid: {ALL_TRACKS}")

    result = run_full(tracks, target_rows=args.target_rows,
                      update_baseline=args.update_baseline)
    json_path, md_path = write_results(result)
    print(f"results: {json_path}")
    print(f"report:  {md_path}")
    if args.report == "site":
        docs = update_docs_site(result)
        print(f"docs:    {docs}")

    gates = result["release_gates"]
    if gates["passed"]:
        print("ALL RELEASE GATES PASSED")
    else:
        for failure in gates["failures"]:
            print(f"GATE FAIL: {failure}", file=sys.stderr)
        if args.check_gates:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
