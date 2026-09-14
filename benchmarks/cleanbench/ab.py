"""Same-run A/B performance gate on the CleanBench T5 workload.

The old T5 gate compared this run's wall-clock against a number recorded once
on a different machine, so it passed or failed with runner noise. This module
measures a *base* and a *head* checkout of freshdata in the same job, on the
same runner, and gates on their ratio instead:

* every measurement runs in a fresh worker subprocess whose ``PYTHONPATH`` puts
  that side's ``src`` first; the harness and the fixture always come from HEAD,
  and the worker refuses to run if freshdata was imported from anywhere else;
* sides alternate (base/head, then head/base) across pairs, so slow drift on
  the runner hits both equally;
* runtime compares the fastest run of each side (the least noisy estimator of
  the true cost) and memory the median peak-RSS delta per worker;
* a breach only fails the gate when a full confirmation re-run breaches too.

Usage::

    python -m benchmarks.cleanbench.ab --base-src ../base/src --head-src src --check-gates
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .metrics import FULL_GATE_MEMORY_OVERHEAD, FULL_GATE_RUNTIME_SLOWDOWN

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "cleanbench" / "results"

#: Below this base peak-RSS delta the memory ratio is dominated by allocator
#: noise, so the memory gate is reported as not applicable.
MIN_RSS_DELTA_BYTES = 4 * 1024 * 1024

Worker = Callable[..., dict[str, Any]]


@dataclass
class SideSamples:
    """Everything measured for one side (base or head) across all workers."""

    seconds: list[float] = field(default_factory=list)
    rss_deltas: list[int] = field(default_factory=list)
    freshdata_path: str = ""


# --------------------------------------------------------------------------- #
# worker (runs inside a subprocess with one side's src first on sys.path)
# --------------------------------------------------------------------------- #

def _worker(src: str, *, target_rows: int, warmups: int, runs: int) -> dict[str, Any]:
    import freshdata as fd  # noqa: PLC0415 - must resolve from this side's src

    loaded = Path(fd.__file__).resolve()
    expected = Path(src).resolve()
    if expected not in loaded.parents:
        raise SystemExit(f"freshdata was imported from {loaded}, expected under {expected}")

    from .fixtures import make_t5_scale_fixture  # noqa: PLC0415
    from .runner import _PeakRss  # noqa: PLC0415

    _truth, corrupted, kwargs = make_t5_scale_fixture(target_rows=target_rows)

    def clean_once() -> None:
        fd.clean(corrupted, return_report=True, **kwargs)

    # Memory is measured on the first (cold) clean: once a clean has run, the
    # allocator keeps its pages and later peak-RSS deltas read as ~0. That run
    # also serves as the first warmup for the timings.
    gc.collect()
    with _PeakRss() as rss:
        clean_once()
    for _ in range(max(0, warmups - 1)):
        clean_once()
    seconds: list[float] = []
    for _ in range(runs):
        gc.collect()
        start = time.perf_counter()
        clean_once()
        seconds.append(time.perf_counter() - start)
    return {
        "freshdata_path": str(loaded),
        "seconds": seconds,
        "peak_rss_delta_bytes": rss.delta,
    }


def _run_worker(
    src: str | Path,
    *,
    target_rows: int,
    warmups: int,
    runs: int,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Run one measurement in a fresh interpreter importing freshdata from *src*."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(src).resolve()), str(REPO_ROOT)])
    cmd = [
        sys.executable, "-m", "benchmarks.cleanbench.ab", "--worker",
        "--src", str(src),
        "--target-rows", str(target_rows),
        "--warmups", str(warmups),
        "--runs", str(runs),
    ]
    proc = subprocess.run(  # noqa: S603 - fixed interpreter and arguments
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"A/B worker for {src} exited {proc.returncode}:\n{proc.stderr[-2000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def measure(
    base_src: str | Path,
    head_src: str | Path,
    *,
    pairs: int,
    target_rows: int,
    warmups: int,
    runs: int,
    worker: Worker = _run_worker,
) -> dict[str, SideSamples]:
    """Interleave base/head workers, alternating which side goes first."""
    sources = {"base": base_src, "head": head_src}
    sides = {"base": SideSamples(), "head": SideSamples()}
    for pair in range(pairs):
        order = ("base", "head") if pair % 2 == 0 else ("head", "base")
        for label in order:
            payload = worker(
                sources[label], target_rows=target_rows, warmups=warmups, runs=runs
            )
            side = sides[label]
            side.seconds.extend(float(s) for s in payload["seconds"])
            side.rss_deltas.append(int(payload["peak_rss_delta_bytes"]))
            side.freshdata_path = str(payload.get("freshdata_path", ""))
    return sides


def summarize(sides: dict[str, SideSamples]) -> dict[str, Any]:
    base, head = sides["base"], sides["head"]
    base_fastest, head_fastest = min(base.seconds), min(head.seconds)
    base_rss = statistics.median(base.rss_deltas)
    head_rss = statistics.median(head.rss_deltas)
    memory_overhead = (
        round(head_rss / base_rss - 1.0, 4) if base_rss >= MIN_RSS_DELTA_BYTES else None
    )
    return {
        "samples_per_side": len(base.seconds),
        "base_fastest_seconds": round(base_fastest, 4),
        "head_fastest_seconds": round(head_fastest, 4),
        "base_median_seconds": round(statistics.median(base.seconds), 4),
        "head_median_seconds": round(statistics.median(head.seconds), 4),
        "runtime_slowdown": round(head_fastest / base_fastest - 1.0, 4),
        "base_median_rss_delta_bytes": int(base_rss),
        "head_median_rss_delta_bytes": int(head_rss),
        "memory_overhead": memory_overhead,
        "base_freshdata_path": base.freshdata_path,
        "head_freshdata_path": head.freshdata_path,
    }


def gate_failures(
    summary: dict[str, Any],
    *,
    runtime_threshold: float = FULL_GATE_RUNTIME_SLOWDOWN,
    memory_threshold: float = FULL_GATE_MEMORY_OVERHEAD,
) -> dict[str, str]:
    """Breached gates for one measurement, keyed by metric name."""
    failures: dict[str, str] = {}
    slowdown = summary["runtime_slowdown"]
    if slowdown > runtime_threshold:
        failures["runtime"] = (
            f"runtime slowdown {slowdown:+.1%} > {runtime_threshold:.0%} "
            f"(fastest head {summary['head_fastest_seconds']}s vs "
            f"base {summary['base_fastest_seconds']}s)"
        )
    overhead = summary["memory_overhead"]
    if overhead is not None and overhead > memory_threshold:
        failures["memory"] = (
            f"memory overhead {overhead:+.1%} > {memory_threshold:.0%} "
            f"(median peak RSS delta head {summary['head_median_rss_delta_bytes']} "
            f"vs base {summary['base_median_rss_delta_bytes']} bytes)"
        )
    return failures


def run_ab(
    base_src: str | Path,
    head_src: str | Path,
    *,
    pairs: int = 5,
    target_rows: int = 200_000,
    warmups: int = 2,
    runs: int = 2,
    confirm: bool = True,
    worker: Worker = _run_worker,
    runtime_threshold: float = FULL_GATE_RUNTIME_SLOWDOWN,
    memory_threshold: float = FULL_GATE_MEMORY_OVERHEAD,
) -> dict[str, Any]:
    """Measure, gate, and (on a breach) confirm with a full second measurement."""
    options = {"pairs": pairs, "target_rows": target_rows, "warmups": warmups, "runs": runs}
    thresholds = {"runtime_threshold": runtime_threshold, "memory_threshold": memory_threshold}
    first = summarize(measure(base_src, head_src, worker=worker, **options))
    failures = gate_failures(first, **thresholds)
    confirmation = None
    if failures and confirm:
        confirmation = summarize(measure(base_src, head_src, worker=worker, **options))
        repeated = gate_failures(confirmation, **thresholds)
        # A metric fails only when it breached on both measurements.
        failures = {metric: repeated[metric] for metric in failures if metric in repeated}
    return {
        "base_src": str(base_src),
        "head_src": str(head_src),
        "options": options,
        "thresholds": thresholds,
        "measurement": first,
        "confirmation": confirmation,
        "failures": list(failures.values()),
        "passed": not failures,
    }


def render_markdown(result: dict[str, Any]) -> str:
    m = result["measurement"]
    base_label = result.get("base_label") or result["base_src"]
    head_label = result.get("head_label") or result["head_src"]
    mib = 1024 * 1024
    memory_change = (
        f"{m['memory_overhead']:+.1%}" if m["memory_overhead"] is not None else "n/a (tiny)"
    )
    thresholds = result["thresholds"]
    lines = [
        "## Performance A/B gate (CleanBench T5 workload)",
        "",
        f"Base **{base_label}** vs head **{head_label}**, "
        f"{result['options']['target_rows']:,} rows, {result['options']['pairs']} interleaved "
        f"worker pairs, {m['samples_per_side']} timed runs per side.",
        "",
        "| metric | base | head | change | limit |",
        "|---|---|---|---|---|",
        f"| runtime (fastest run) | {m['base_fastest_seconds']:.3f}s | "
        f"{m['head_fastest_seconds']:.3f}s | {m['runtime_slowdown']:+.1%} | "
        f"+{thresholds['runtime_threshold']:.0%} |",
        f"| runtime (median run) | {m['base_median_seconds']:.3f}s | "
        f"{m['head_median_seconds']:.3f}s | | |",
        f"| peak RSS delta (median) | {m['base_median_rss_delta_bytes'] / mib:.1f} MiB | "
        f"{m['head_median_rss_delta_bytes'] / mib:.1f} MiB | {memory_change} | "
        f"+{thresholds['memory_threshold']:.0%} |",
        "",
    ]
    if result["confirmation"] is not None:
        c = result["confirmation"]
        confirm_memory = (
            f"{c['memory_overhead']:+.1%}" if c["memory_overhead"] is not None else "n/a"
        )
        lines.append(
            f"A breach triggered a confirmation run: runtime {c['runtime_slowdown']:+.1%}, "
            f"memory {confirm_memory}."
        )
        lines.append("")
    if result["passed"]:
        lines.append("**PASS**")
    else:
        lines.append("**FAIL** (reproduced on the confirmation run):")
        lines.extend(f"- {failure}" for failure in result["failures"])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.cleanbench.ab")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--src", help=argparse.SUPPRESS)
    parser.add_argument("--base-src", help="src/ directory of the base checkout")
    parser.add_argument("--head-src", default=str(REPO_ROOT / "src"),
                        help="src/ directory of the head checkout (default: this repo)")
    parser.add_argument("--base-label", default=None, help="label for the report")
    parser.add_argument("--head-label", default=None, help="label for the report")
    parser.add_argument("--pairs", type=int, default=5, help="interleaved worker pairs")
    parser.add_argument("--runs", type=int, default=2, help="timed runs per worker")
    parser.add_argument("--warmups", type=int, default=2, help="untimed runs per worker")
    parser.add_argument("--target-rows", type=int, default=200_000, help="T5 frame size")
    parser.add_argument("--no-confirm", action="store_true",
                        help="fail on the first breach without a confirmation run")
    parser.add_argument("--check-gates", action="store_true",
                        help="exit non-zero when a regression is confirmed")
    parser.add_argument("--output", default=str(RESULTS_DIR / "latest.ab.json"),
                        help="where to write the JSON result")
    parser.add_argument("--summary", default=None,
                        help="append the markdown summary to this file "
                             "(e.g. $GITHUB_STEP_SUMMARY)")
    args = parser.parse_args(argv)

    if args.worker:
        payload = _worker(
            args.src, target_rows=args.target_rows, warmups=args.warmups, runs=args.runs
        )
        print(json.dumps(payload))
        return 0
    if not args.base_src:
        parser.error("--base-src is required")

    result = run_ab(
        args.base_src,
        args.head_src,
        pairs=args.pairs,
        target_rows=args.target_rows,
        warmups=args.warmups,
        runs=args.runs,
        confirm=not args.no_confirm,
    )
    result["base_label"] = args.base_label
    result["head_label"] = args.head_label

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    print(markdown)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    for failure in result["failures"]:
        print(f"GATE FAIL: {failure}", file=sys.stderr)
    return 1 if args.check_gates and result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
