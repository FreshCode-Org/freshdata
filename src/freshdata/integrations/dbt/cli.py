"""``dbt-gate`` console script — trust-gate every model in a dbt manifest.

Run it after ``dbt run`` (typically as a CI step) to score each materialized model
and fail the build on low data quality:

    dbt-gate --manifest target/manifest.json --threshold 80 --fail

The warehouse connection comes from ``--conn`` or ``$FRESHDATA_WAREHOUSE_CONN``.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import gate_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dbt-gate",
        description="Trust-gate every model in a dbt manifest with freshdata.",
    )
    parser.add_argument(
        "--manifest",
        default="target/manifest.json",
        help="Path to dbt's manifest.json (default: target/manifest.json).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Minimum acceptable 0-100 trust score (default: 80).",
    )
    parser.add_argument(
        "--conn",
        default=None,
        help="SQLAlchemy connection string (default: $FRESHDATA_WAREHOUSE_CONN).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-model <model>_audit.json files (default: none).",
    )
    parser.add_argument(
        "--on-low-score",
        choices=["warn", "fail", "skip"],
        default="warn",
        help="Recorded reaction for a failing model (default: warn).",
    )
    parser.add_argument(
        "--fail",
        action="store_true",
        help="Exit non-zero when any model is below the threshold.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``dbt-gate`` script. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    summary = gate_manifest(
        args.manifest,
        conn_str=args.conn,
        trust_score_threshold=args.threshold,
        on_low_score=args.on_low_score,
        output_dir=args.output_dir,
    )
    json.dump(summary, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    if args.fail and not summary["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
