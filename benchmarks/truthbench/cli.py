"""Command-line interface for TruthBench.

``python -m benchmarks.truthbench run --profile release --backends
pandas,polars,duckdb --require-backends --repeats 2 --check`` is the official
release command.  A partial run, missing adapter, missing backend, empty
record set, or infrastructure error exits nonzero.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .report import RESULTS_DIR
from .surfaces.backends import BackendUnavailableError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.truthbench",
        description="TruthBench: absolute, fail-closed release verification.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute a verification run")
    run.add_argument("--profile", choices=("release", "extended"), default="release")
    run.add_argument(
        "--backends",
        default="pandas,polars,duckdb",
        help="comma-separated required backends",
    )
    run.add_argument(
        "--require-backends",
        action="store_true",
        help="fail (never skip) when a required backend is unavailable",
    )
    run.add_argument("--repeats", type=int, default=2)
    run.add_argument(
        "--domains",
        default=None,
        help="comma-separated fixture domains (default: all)",
    )
    run.add_argument("--results-dir", default=str(RESULTS_DIR))
    run.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero unless every gate passes (the release gate)",
    )
    run.add_argument(
        "--check-regressions",
        action="store_true",
        help=(
            "exit nonzero when a gate that passes in the committed "
            "baseline.json fails now, or on any infrastructure error — the "
            "PR ratchet. Known-red gates are reported loudly but do not "
            "fail the run; the release workflow still enforces --check."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":  # pragma: no cover - argparse enforces choices
        return 2
    from .fixtures import DOMAINS
    from .runner import TruthBenchRunError, run_release

    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    domains = (
        tuple(d.strip() for d in args.domains.split(",") if d.strip())
        if args.domains
        else DOMAINS
    )
    try:
        outcome = run_release(
            profile=args.profile,
            backends=backends,
            repeats=args.repeats,
            require_backends=args.require_backends,
            domains=domains,
            results_dir=args.results_dir,
        )
    except (TruthBenchRunError, BackendUnavailableError) as exc:
        print(f"TRUTHBENCH INFRASTRUCTURE FAILURE: {exc}", file=sys.stderr)
        return 2

    payload = outcome.run.to_dict()
    print(f"run id:  {outcome.run.run_id}")
    print(f"records: {len(outcome.run.records)}")
    for gate in payload["gates"]:
        status = "pass" if gate["passed"] else "FAIL"
        print(f"gate {gate['name']}: {status} ({gate['failure_count']} failures)")
    for note in outcome.baseline_notes:
        print(f"baseline: {note}")
    for name, path in outcome.artifacts.items():
        print(f"artifact: {name} -> {path}")
    print(f"overall: {'PASS' if outcome.passed else 'FAIL'}")
    if args.check and not outcome.passed:
        return 1
    if args.check_regressions:
        regressions = [
            note for note in outcome.baseline_notes if note.startswith("regression:")
        ]
        known_red = [gate for gate in payload["gates"] if not gate["passed"]]
        for gate in known_red:
            print(
                f"KNOWN-RED (release-blocking, not a PR regression): "
                f"{gate['name']} ({gate['failure_count']} failures)"
            )
        if regressions:
            for note in regressions:
                print(f"REGRESSION: {note}", file=sys.stderr)
            return 1
    return 0


__all__ = ["build_parser", "main"]
