#!/usr/bin/env python
"""FreshData benchmark harness.

Reproducible, schema-stable benchmarks for FreshData against an enterprise
fixture library and a set of competitor baselines. See ``benchmarks/README.md``
and ``docs/benchmarks.md`` for the full story.

Subcommands::

    python benchmarks/bench.py run        # all fixtures (small sizes), write results/
    python benchmarks/bench.py compare    # FreshData vs baselines on one fixture/size
    python benchmarks/bench.py report      # markdown + JSON summary of a results dir
    python benchmarks/bench.py fixtures    # generate fixture files to disk
    python benchmarks/bench.py single --fixture crm --size 100000 --metric time

All nine metrics are written to ``results/<run_id>/<fixture>/<size>.json`` in
the schema defined in ``results_schema.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the sibling modules importable whether run as a script or a module.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import freshdata as fd  # noqa: E402

import harness_metrics as hm  # noqa: E402
from fixtures import REGISTRY  # noqa: E402
from results_schema import RESULTS_SCHEMA, SCHEMA_VERSION  # noqa: E402

RESULTS_DIR = HERE / "results"

# Default in-CI sizes (small). Large variants are local-only; see each fixture's
# SCALE_VARIANTS for the full ladder.
DEFAULT_SIZES = {
    "crm": 10_000,
    "finance": 10_000,
    "event_log": 10_000,
    "wide_schema": 10_000,
    "provenance": 10_000,
    "gold": 10_000,
}
TRUST_SWEEP_SIZE = 5_000  # bound the monotonicity sweep cost


# -- core: run one (fixture, size) ----------------------------------------
def run_single(name: str, size: int, *, seed: int = 42, aggressive: bool = False,
               repeat: int = 5, sweep_size: int | None = None) -> dict:
    """Compute all nine metrics for one (fixture, size) and return a result dict."""
    if name not in REGISTRY:
        raise SystemExit(f"unknown fixture {name!r}; choose from {sorted(REGISTRY)}")

    df = hm.make_frame(name, size, seed)
    config = hm.config_for(name, df, aggressive=aggressive)
    cleaned, report = fd.clean(df, config=config, return_report=True)
    cleaned = cleaned.reset_index(drop=True)

    wall = hm.metric_wall_clock(df, config, repeat=repeat)
    mem = hm.metric_peak_memory(df, config)
    diag = hm.metric_diagnosis_speed(report)
    lines = hm.metric_authored_lines()
    trust = hm.metric_trust(name, sweep_size or min(size, TRUST_SWEEP_SIZE), seed)
    exports = hm.metric_export_completeness(name, df, config, report)

    details: dict = {"trust_sweep": trust["sweep"], "export_missing": exports["fields_missing"]}

    if name == "gold":
        gr = hm.gold_repair_report(size, seed)
        repair_fidelity = gr["repair_fidelity_pct"]
        false_repair = gr["false_repair_rate_pct"]
        preservation = gr["preservation_rate_pct"]
        details["gold"] = {
            "per_family": gr["per_family"],
            "per_column": gr["per_column"],
            "per_trap": gr["per_trap"],
        }
    else:
        pres = hm.preservation_report(name, df, cleaned)
        fidelity = hm.manifest_repair_fidelity(name, df, cleaned, report)
        repair_fidelity = fidelity["repair_fidelity_pct"]
        false_repair = pres["false_repair_rate_pct"]
        preservation = pres["preservation_rate_pct"]
        details["preservation"] = pres
        details["repair_families"] = fidelity["per_family"]

    metrics = {
        "wall_clock_p50_sec": round(wall["p50"], 6),
        "wall_clock_p95_sec": round(wall["p95"], 6),
        "peak_memory_mb": mem["peak_mb"],
        "repair_fidelity_pct": repair_fidelity,
        "false_repair_rate_pct": false_repair,
        "preservation_rate_pct": preservation,
        "authored_lines_fd": lines["fd_lines"],
        "authored_lines_pandas": lines["pandas_lines"],
        "reduction_vs_pandas_pct": lines["reduction_vs_pandas_pct"],
        "diagnosis_summary_sec": diag["summary_sec"],
        "diagnosis_to_frame_sec": diag["to_frame_sec"],
        "diagnosis_to_dict_sec": diag["to_dict_sec"],
        "trust_score": trust["trust_score"],
        "trust_monotonic_valid": trust["trust_monotonic_valid"],
        "export_completeness_pct": exports["export_completeness_pct"],
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": _now(),
        "freshdata_version": fd.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "fixture": name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "mode": "aggressive" if aggressive else "balanced",
        "seed": seed,
        "metrics": metrics,
        "details": details,
        "reduction_vs_pyjanitor_pct": lines["reduction_vs_pyjanitor_pct"],
    }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_result(run_id: str, result: dict) -> Path:
    out = RESULTS_DIR / run_id / result["fixture"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{result['n_rows']}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    return path


# -- subcommand: run -------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    run_id = _now()
    fixtures = args.fixtures or list(DEFAULT_SIZES)
    print(f"benchmark run {run_id}  freshdata={fd.__version__}  mode={'aggressive' if args.aggressive else 'balanced'}")
    summary_rows = []
    for name in fixtures:
        size = args.size or DEFAULT_SIZES[name]
        result = run_single(name, size, seed=args.seed, aggressive=args.aggressive, repeat=args.repeat)
        result["run_id"] = run_id
        path = _write_result(run_id, result)
        m = result["metrics"]
        print(f"  {name:12s} n={result['n_rows']:>8,} cols={result['n_cols']:>4}  "
              f"p50={m['wall_clock_p50_sec']:.3f}s  peak={m['peak_memory_mb']:.1f}MB  "
              f"fidelity={m['repair_fidelity_pct']}%  false_repair={m['false_repair_rate_pct']}%  "
              f"preserve={m['preservation_rate_pct']}%  trust={m['trust_score']}")
        summary_rows.append(result)
    _write_summary(run_id, summary_rows)
    print(f"\nresults written to {RESULTS_DIR / run_id}")
    return 0


def _write_summary(run_id: str, rows: list[dict]) -> None:
    (RESULTS_DIR / run_id).mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "freshdata_version": fd.__version__,
        "fixtures": {r["fixture"]: r["metrics"] for r in rows},
    }
    (RESULTS_DIR / run_id / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


# -- subcommand: compare ---------------------------------------------------
def cmd_compare(args: argparse.Namespace) -> int:
    import importlib
    import time

    name = args.fixture or "crm"
    size = args.size or DEFAULT_SIZES.get(name, 10_000)
    df = hm.make_frame(name, size, args.seed)
    config = hm.config_for(name, df)

    print(f"compare on {name} n={len(df):,} cols={df.shape[1]}\n")
    header = f"{'tool':24s} {'n_rows':>8} {'n_cols':>6} {'p50_sec':>8} {'p95_sec':>8} {'peak_mb':>8}  mode"
    print(header)
    print("-" * len(header))

    def _bench(label, fn, mode):
        import gc
        import tracemalloc
        times = []
        for _ in range(max(1, args.repeat)):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        gc.collect()
        tracemalloc.start()
        fn()
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        print(f"{label:24s} {len(df):>8,} {df.shape[1]:>6} {hm._pctl(times,50):>8.3f} "
              f"{hm._pctl(times,95):>8.3f} {peak:>8.1f}  {mode}")

    _bench("freshdata", lambda: fd.clean(df, config=config, return_report=True), "balanced")

    from baselines import REGISTRY as BREG
    for bname in BREG:
        mod = importlib.import_module(f"baselines.{bname}")
        try:
            mod.run(df.copy())
        except Exception as exc:  # ImportError (missing lib) or a baseline limitation
            reason = (exc.args[0].split(";")[0] if exc.args else type(exc).__name__)[:40]
            print(f"{bname:24s} {'—':>8} {'—':>6} {'—':>8} {'—':>8} {'—':>8}  skip ({reason})")
            continue
        _bench(bname, lambda m=mod: m.run(df.copy()), "n/a")

    # authored-code reduction (separate from timing, per Metric 6)
    lines = hm.metric_authored_lines()
    print(f"\nauthored lines — freshdata={lines['fd_lines']} pandas={lines['pandas_lines']} "
          f"pyjanitor={lines['pyjanitor_lines']}  "
          f"reduction_vs_pandas={lines['reduction_vs_pandas_pct']}% "
          f"reduction_vs_pyjanitor={lines['reduction_vs_pyjanitor_pct']}%")
    return 0


# -- subcommand: report ----------------------------------------------------
def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run()
    if run_dir is None or not run_dir.exists():
        raise SystemExit("no results found; run `bench.py run` first")
    results = []
    for path in sorted(run_dir.glob("*/*.json")):
        if path.name == "summary.json":
            continue
        results.append(json.loads(path.read_text()))
    md = _render_markdown(run_dir.name, results)
    md_path = run_dir / "report.md"
    md_path.write_text(md)
    (run_dir / "report.json").write_text(json.dumps(results, indent=2, default=str))
    print(md)
    print(f"\nwritten: {md_path}")
    return 0


def _latest_run() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    runs = sorted([p for p in RESULTS_DIR.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def _render_markdown(run_id: str, results: list[dict]) -> str:
    lines = [
        f"# FreshData benchmark report — `{run_id}`",
        "",
        f"- freshdata: `{results[0]['freshdata_version'] if results else '?'}`",
        f"- python: `{results[0]['python_version'] if results else '?'}`",
        f"- platform: `{results[0]['platform'] if results else '?'}`",
        "",
        "| fixture | n_rows | n_cols | p50 s | p95 s | peak MB | repair % | false-repair % | preserve % | trust | monotonic | export % |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|--:|",
    ]
    for r in sorted(results, key=lambda x: x["fixture"]):
        m = r["metrics"]
        lines.append(
            f"| {r['fixture']} | {r['n_rows']:,} | {r['n_cols']} | "
            f"{m['wall_clock_p50_sec']:.3f} | {m['wall_clock_p95_sec']:.3f} | {m['peak_memory_mb']:.1f} | "
            f"{m['repair_fidelity_pct']} | {m['false_repair_rate_pct']} | {m['preservation_rate_pct']} | "
            f"{m['trust_score']} | {'✅' if m['trust_monotonic_valid'] else '❌'} | {m['export_completeness_pct']} |"
        )
    lines += ["", "## Authored-code reduction (Metric 6)", ""]
    if results:
        m = results[0]["metrics"]
        lines += [
            f"- FreshData: **{m['authored_lines_fd']}** lines",
            f"- pandas baseline: {m['authored_lines_pandas']} lines "
            f"(**{m['reduction_vs_pandas_pct']}%** reduction)",
        ]
    return "\n".join(lines)


# -- subcommand: fixtures --------------------------------------------------
def cmd_fixtures(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else (HERE / "generated_fixtures")
    out.mkdir(parents=True, exist_ok=True)
    size = args.size or 10_000
    for name in (args.fixtures or list(REGISTRY)):
        mod = REGISTRY[name]
        if name == "gold":
            bundle = mod.generate(size, seed=args.seed)
            bundle.dirty_df.to_csv(out / "gold_dirty.csv", index=False)
            bundle.clean_df.to_csv(out / "gold_clean.csv", index=False)
            print(f"  gold -> gold_dirty.csv ({bundle.dirty_df.shape}), gold_clean.csv ({bundle.clean_df.shape})")
        else:
            df = mod.generate(size, seed=args.seed)
            path = out / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"  {name} -> {path.name} {df.shape}")
    print(f"\nfixtures written to {out}")
    return 0


# -- subcommand: single ----------------------------------------------------
def cmd_single(args: argparse.Namespace) -> int:
    name = args.fixture
    size = args.size or DEFAULT_SIZES.get(name, 10_000)
    result = run_single(name, size, seed=args.seed, aggressive=args.aggressive, repeat=args.repeat)
    if args.metric and args.metric != "all":
        key = _METRIC_ALIASES.get(args.metric, args.metric)
        subset = {k: v for k, v in result["metrics"].items() if k.startswith(key)}
        print(json.dumps(subset or {key: result["metrics"].get(key)}, indent=2, default=str))
    else:
        print(json.dumps(result, indent=2, default=str))
    if args.write:
        path = _write_result(result["run_id"], result)
        print(f"\nwritten: {path}", file=sys.stderr)
    return 0


_METRIC_ALIASES = {
    "time": "wall_clock",
    "memory": "peak_memory",
    "fidelity": "repair_fidelity",
    "false_repair": "false_repair_rate",
    "preservation": "preservation_rate",
    "lines": "authored_lines",
    "diagnosis": "diagnosis",
    "trust": "trust",
    "export": "export_completeness",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--size", type=int, default=None, help="row count override")
    common.add_argument("--repeat", type=int, default=5, help="timing repeats")
    common.add_argument("--aggressive", action="store_true", help="use aggressive mode variant")

    pr = sub.add_parser("run", parents=[common], help="run all benchmarks")
    pr.add_argument("--fixtures", nargs="*", choices=list(REGISTRY), default=None)
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("compare", parents=[common], help="FreshData vs baselines")
    pc.add_argument("--fixture", choices=list(REGISTRY), default="crm")
    pc.set_defaults(func=cmd_compare)

    prep = sub.add_parser("report", help="render markdown + JSON summary")
    prep.add_argument("--run-dir", default=None, help="results/<run_id> dir (default: latest)")
    prep.set_defaults(func=cmd_report)

    pf = sub.add_parser("fixtures", parents=[common], help="write fixtures to disk")
    pf.add_argument("--fixtures", nargs="*", choices=list(REGISTRY), default=None)
    pf.add_argument("--out", default=None)
    pf.set_defaults(func=cmd_fixtures)

    ps = sub.add_parser("single", parents=[common], help="one fixture/size/metric")
    ps.add_argument("--fixture", choices=list(REGISTRY), required=True)
    ps.add_argument("--metric", default="all", help="time|memory|fidelity|trust|... or all")
    ps.add_argument("--write", action="store_true", help="also write the result JSON")
    ps.set_defaults(func=cmd_single)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
