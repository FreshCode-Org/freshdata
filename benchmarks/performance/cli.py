from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter
from pathlib import Path

from .analysis import analyze_results, load_results
from .baselines import BASELINES, expand_baseline_cases, run_baseline_matrix
from .render import render_report
from .runner import expand_cases, run_matrix
from .schema import validate_finite_numbers, validate_result
from .worker import execute_profile_case


def _comma_separated(value: str) -> list[str]:
    return value.split(",")


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(f"summary JSON contains non-standard constant: {value}")


def _report_modes(value: str) -> list[bool]:
    modes = _comma_separated(value)
    if any(mode not in {"false", "true"} for mode in modes):
        raise argparse.ArgumentTypeError("report modes must be 'false' or 'true'")
    return [mode == "true" for mode in modes]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.performance")
    subcommands = parser.add_subparsers(dest="subcommand", required=True)
    run = subcommands.add_parser("run")
    _add_case_arguments(
        run,
        rows="10000,100000,500000,1000000",
        widths="narrow,medium,wide",
        configs="default,conservative,representation_off,statistical_off,explicit",
        report_modes="false,true",
    )
    run.add_argument("--timeout", type=int, default=1800)
    run.add_argument("--output", required=True)

    profile = subcommands.add_parser("profile")
    _add_case_arguments(
        profile,
        rows="10000",
        widths="narrow",
        configs="default",
        report_modes="false",
    )
    profile.add_argument("--output", required=True)

    baseline = subcommands.add_parser("baseline")
    baseline.add_argument("--rows", default="10000,100000")
    baseline.add_argument("--widths", default="narrow,medium,wide")
    baseline.add_argument("--dataset-types", default="mixed")
    baseline.add_argument("--baselines", default=",".join(BASELINES))
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--warmups", type=int, default=1)
    baseline.add_argument("--repetitions", type=int, default=5)
    baseline.add_argument("--timeout", type=int, default=1800)
    baseline.add_argument("--output", required=True)

    analyze = subcommands.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--output", required=True)

    render = subcommands.add_parser("render")
    render.add_argument("--input", required=True)
    render.add_argument("--output", required=True)
    return parser


def _add_case_arguments(
    parser: argparse.ArgumentParser,
    *,
    rows: str,
    widths: str,
    configs: str,
    report_modes: str,
) -> None:
    parser.add_argument("--rows", default=rows)
    parser.add_argument("--widths", default=widths)
    parser.add_argument("--dataset-types", default="mixed")
    parser.add_argument(
        "--configs",
        default=configs,
    )
    parser.add_argument("--report-modes", default=report_modes)
    parser.add_argument("--backends", default="pandas")
    parser.add_argument("--output-formats", default="pandas")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.subcommand == "analyze":
        summary = analyze_results(load_results(Path(arguments.input)))
        Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.output).write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return 0
    if arguments.subcommand == "render":
        payload = json.loads(
            Path(arguments.input).read_text(encoding="utf-8"),
            parse_constant=_reject_non_standard_json_constant,
        )
        if not isinstance(payload, dict):
            parser.error("render input must be a JSON object")
        validate_finite_numbers(payload, "summary JSON")
        Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.output).write_text(render_report(payload), encoding="utf-8")
        return 0
    if arguments.subcommand == "baseline":
        baseline_names = _comma_separated(arguments.baselines)
        unknown = sorted(set(baseline_names) - set(BASELINES))
        if unknown:
            parser.error(f"unknown pandas baseline(s): {', '.join(unknown)}")
        cases = expand_baseline_cases(
            rows=[int(value) for value in _comma_separated(arguments.rows)],
            widths=_comma_separated(arguments.widths),
            dataset_types=_comma_separated(arguments.dataset_types),
            baseline_names=baseline_names,
            seed=arguments.seed,
            warmups=arguments.warmups,
            repetitions=arguments.repetitions,
        )
        results = run_baseline_matrix(cases, Path(arguments.output), arguments.timeout)
        for result in results:
            print(f"{result.case.case_id} {result.baseline_name} {result.status}")
        counts = Counter(result.status for result in results)
        print(" ".join(f"{status}={counts[status]}" for status in sorted(counts)))
        return 0 if all(result.status == "completed" for result in results) else 1

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
    if arguments.subcommand == "profile":
        if len(cases) != 1:
            parser.error("profile requires exactly one benchmark case")
        output_dir = Path(arguments.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_arguments = argv if argv is not None else sys.argv[1:]
        command = shlex.join([sys.executable, "-m", "benchmarks.performance", *raw_arguments])
        result = execute_profile_case(cases[0], command=command)
        payload = result.to_dict()
        validate_result(payload)
        output_path = output_dir / f"{cases[0].case_id}.profile.json"
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        print(f"{result.case.case_id} {result.status}")
        return 0

    results = run_matrix(cases, Path(arguments.output), arguments.timeout)
    for result in results:
        print(f"{result.case.case_id} {result.status}")
    counts = Counter(result.status for result in results)
    print(" ".join(f"{status}={counts[status]}" for status in sorted(counts)))
    return 0 if all(result.status in {"completed", "skipped"} for result in results) else 1
