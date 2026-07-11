from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .runner import expand_cases, run_matrix


def _comma_separated(value: str) -> list[str]:
    return value.split(",")


def _report_modes(value: str) -> list[bool]:
    modes = _comma_separated(value)
    if any(mode not in {"false", "true"} for mode in modes):
        raise argparse.ArgumentTypeError("report modes must be 'false' or 'true'")
    return [mode == "true" for mode in modes]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.performance")
    subcommands = parser.add_subparsers(dest="subcommand", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--rows", default="10000,100000,500000,1000000")
    run.add_argument("--widths", default="narrow,medium,wide")
    run.add_argument("--dataset-types", default="mixed")
    run.add_argument(
        "--configs",
        default="default,conservative,representation_off,statistical_off,explicit",
    )
    run.add_argument("--report-modes", default="false,true")
    run.add_argument("--backends", default="pandas")
    run.add_argument("--output-formats", default="pandas")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--repetitions", type=int, default=5)
    run.add_argument("--timeout", type=int, default=1800)
    run.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    cases = expand_cases(
        rows=[int(value) for value in _comma_separated(arguments.rows)],
        widths=_comma_separated(arguments.widths),
        dataset_types=_comma_separated(arguments.dataset_types),
        configs=_comma_separated(arguments.configs),
        report_modes=_report_modes(arguments.report_modes),
        backends=_comma_separated(arguments.backends),
        output_formats=_comma_separated(arguments.output_formats),
        seed=arguments.seed,
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    results = run_matrix(cases, Path(arguments.output), arguments.timeout)
    for result in results:
        print(f"{result.case.case_id} {result.status}")
    counts = Counter(result.status for result in results)
    print(" ".join(f"{status}={counts[status]}" for status in sorted(counts)))
    return 0 if all(result.status in {"completed", "skipped"} for result in results) else 1
