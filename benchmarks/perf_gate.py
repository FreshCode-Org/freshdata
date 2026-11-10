from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from benchmarks.bench_v3_lazy_vs_pandas import generate_messy_parquet


@dataclass(frozen=True)
class GateThresholds:
    max_compile_ms: float = 10.0
    max_total_regression_pct: float = 15.0
    max_rss_regression_pct: float = 20.0


def run_median_benchmark(
    rows: int, iterations: int, worker: str = "freshdata_v3_lazy"
) -> dict[str, float]:
    bench_script = Path(__file__).with_name("bench_v3_lazy_vs_pandas.py")
    samples: list[dict[str, float]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        src_path = tmp / "input.parquet"
        generate_messy_parquet(src_path, rows)

        for i in range(iterations):
            dst_path = tmp / f"out_{i}.parquet"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(bench_script),
                    "--worker",
                    worker,
                    "--src",
                    str(src_path),
                    "--dst",
                    str(dst_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            samples.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    keys = ("plan_ms", "compile_ms", "execute_ms", "total_ms", "peak_rss_mb")
    return {
        k: round(statistics.median(s[k] for s in samples), 3)
        for k in keys
    }


def evaluate_gate(
    baseline: dict[str, float],
    candidate: dict[str, float],
    thresholds: GateThresholds,
) -> list[str]:
    failures: list[str] = []

    if candidate["compile_ms"] > thresholds.max_compile_ms:
        failures.append(
            f"compile_ms ({candidate['compile_ms']:.2f} ms) exceeded absolute O(1) ceiling "
            f"({thresholds.max_compile_ms:.2f} ms). Check apply_plan() for eager .collect() calls."
        )

    base_total = max(baseline["total_ms"], 1e-6)
    total_delta_pct = ((candidate["total_ms"] - base_total) / base_total) * 100.0
    if total_delta_pct > thresholds.max_total_regression_pct:
        failures.append(
            f"total_ms regressed by +{total_delta_pct:.1f}% "
            f"(baseline={baseline['total_ms']:.1f} ms, candidate={candidate['total_ms']:.1f} ms, "
            f"limit=+{thresholds.max_total_regression_pct:.1f}%)."
        )

    base_rss = max(baseline["peak_rss_mb"], 1e-6)
    rss_delta_pct = ((candidate["peak_rss_mb"] - base_rss) / base_rss) * 100.0
    if rss_delta_pct > thresholds.max_rss_regression_pct:
        failures.append(
            f"peak_rss_mb regressed by +{rss_delta_pct:.1f}% "
            f"(baseline={baseline['peak_rss_mb']:.1f} MB, candidate={candidate['peak_rss_mb']:.1f} MB, "
            f"limit=+{thresholds.max_rss_regression_pct:.1f}%)."
        )

    return failures


def format_markdown_summary(
    baseline: dict[str, float],
    candidate: dict[str, float],
    thresholds: GateThresholds,
    failures: list[str],
) -> str:
    status = "FAILED" if failures else "PASSED"
    lines = [
        f"### FreshData v3.0 Performance Gate: {status}",
        "",
        "| Metric | Base (Median) | PR Candidate (Median) | Delta | Gate Threshold |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]

    metrics_meta = [
        ("compile_ms", "ms", f"Max {thresholds.max_compile_ms:.1f} ms (O(1) AST)"),
        ("plan_ms", "ms", "Informational"),
        ("total_ms", "ms", f"Max +{thresholds.max_total_regression_pct:.1f}% vs Base"),
        ("peak_rss_mb", "MB", f"Max +{thresholds.max_rss_regression_pct:.1f}% vs Base"),
    ]

    for key, unit, limit_str in metrics_meta:
        b_val = baseline[key]
        c_val = candidate[key]
        pct = ((c_val - max(b_val, 1e-6)) / max(b_val, 1e-6)) * 100.0
        lines.append(
            f"| `{key}` | {b_val:.2f} {unit} | {c_val:.2f} {unit} | {pct:+.1f}% | {limit_str} |"
        )

    if failures:
        lines.append("\n**Regression Failures:**")
        for f in failures:
            lines.append(f"- {f}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="CI Performance Gate for freshdata v3.0")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Run median benchmark and save JSON metrics.")
    rec.add_argument("--rows", type=int, default=2_000_000)
    rec.add_argument("--iterations", type=int, default=3)
    rec.add_argument("--out", type=Path, required=True)

    cmp_p = sub.add_parser("compare", help="Compare candidate JSON against baseline JSON.")
    cmp_p.add_argument("--baseline", type=Path, required=True)
    cmp_p.add_argument("--candidate", type=Path, required=True)
    cmp_p.add_argument("--max-compile-ms", type=float, default=10.0)
    cmp_p.add_argument("--max-total-regression-pct", type=float, default=15.0)
    cmp_p.add_argument("--max-rss-regression-pct", type=float, default=20.0)
    cmp_p.add_argument("--summary-out", type=Path, default=None)

    args = parser.parse_args()

    if args.cmd == "record":
        metrics = run_median_benchmark(rows=args.rows, iterations=args.iterations)
        args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved median metrics to {args.out}: {metrics}")
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    thresholds = GateThresholds(
        max_compile_ms=args.max_compile_ms,
        max_total_regression_pct=args.max_total_regression_pct,
        max_rss_regression_pct=args.max_rss_regression_pct,
    )

    failures = evaluate_gate(baseline, candidate, thresholds)
    summary_md = format_markdown_summary(baseline, candidate, thresholds, failures)
    sys.stdout.write(summary_md)

    if args.summary_out:
        with args.summary_out.open("a", encoding="utf-8") as f:
            f.write(summary_md)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
