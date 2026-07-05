"""CLI: ``python -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site``.

Reproduce and verify the committed headline numbers:

    python -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site --reproduce-headline
    python -m benchmarks.cleanbench --verify-results benchmarks/cleanbench/results/latest.json
"""

from __future__ import annotations

import argparse
import json
import sys

from .report import update_docs_site, write_results
from .reproducibility import environment_info, verify_results
from .runner import ALL_TRACKS, run_full
from .tasks import build_all


def _reproduce_command(tracks: tuple[str, ...], report_mode: str) -> str:
    return (
        f"python -m benchmarks.cleanbench --tracks {','.join(tracks)} "
        f"--report {report_mode} --reproduce-headline"
    )


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
    parser.add_argument("--reproduce-headline", action="store_true",
                        help="print environment + version/commit, run the baseline "
                             "harness alongside the tracks, and exit non-zero on any "
                             "hard release-gate failure (implies --check-gates)")
    parser.add_argument("--verify-results", metavar="PATH", default=None,
                        help="verify a committed result JSON (schema, dataset hashes, "
                             "environment metadata, release gates, README claim links) "
                             "and exit; ignores every other flag")
    args = parser.parse_args(argv)

    if args.verify_results is not None:
        failures = verify_results(args.verify_results)
        if failures:
            for f in failures:
                print(f"FAIL: {f}", file=sys.stderr)
            return 1
        print(f"VERIFIED: {args.verify_results}")
        return 0

    if args.build_tasks:
        for path in build_all():
            print(f"task written: {path}")
        return 0

    tracks = tuple(t.strip().upper() for t in args.tracks.split(",") if t.strip())
    unknown = [t for t in tracks if t not in ALL_TRACKS]
    if unknown:
        parser.error(f"unknown tracks: {unknown}; valid: {ALL_TRACKS}")

    if args.reproduce_headline:
        env = environment_info()
        print("== CleanBench reproduce-headline ==")
        for key, value in env.items():
            print(f"  {key}: {value}")
        print(f"  tracks: {', '.join(tracks)}")

    result = run_full(
        tracks,
        target_rows=args.target_rows,
        update_baseline=args.update_baseline,
        include_baselines=args.reproduce_headline,
        command=_reproduce_command(tracks, args.report),
    )
    json_path, md_path = write_results(result)
    print(f"results: {json_path}")
    print(f"report:  {md_path}")
    if args.report == "site":
        docs = update_docs_site(result)
        print(f"docs:    {docs}")

    if args.reproduce_headline and "baselines" in result:
        print("== baselines ==")
        print(json.dumps(result["baselines"], indent=2, default=str))

    gates = result["release_gates"]
    check_gates = args.check_gates or args.reproduce_headline
    if gates["passed"]:
        print("ALL RELEASE GATES PASSED")
    else:
        for failure in gates["failures"]:
            print(f"GATE FAIL: {failure}", file=sys.stderr)
        if check_gates:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
