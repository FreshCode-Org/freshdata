"""``freshdata`` command-line interface for batch / orchestration use.

Designed to drop into Airflow, Prefect, cron, or a Makefile: every command is a pure
function of its arguments, writes machine-readable JSON, and returns a process exit code
(non-zero when a trust-score quality gate fails). No required dependency beyond the core;
YAML config files need ``pyyaml`` (``pip install 'freshdata-cleaner[cli]'``).

    freshdata clean in.csv -o out.parquet --mask email:hash --cluster vendor \\
        --report quality.json --lineage lineage.json --fail-under-trust 80
    freshdata profile in.csv --json
    freshdata trust in.csv --fail-under 90
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from ..config import CleanConfig, merge_options
from ..profile import build_profile
from .config import ClusterConfig, EnterpriseConfig, MaskingRule, SemanticValidatorConfig
from .interface import clean_enterprise
from .metrics import compute_trust_score


def _infer_format(path: str) -> str:
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        return "parquet"
    if low.endswith(".json"):
        return "json"
    return "csv"


def _read_frame(path: str, fmt: str | None) -> pd.DataFrame:
    fmt = fmt or _infer_format(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _write_frame(df: pd.DataFrame, path: str, fmt: str | None) -> None:
    fmt = fmt or _infer_format(path)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records")
    else:
        df.to_csv(path, index=False)


def _load_config_file(path: str) -> dict[str, Any]:
    if path.lower().endswith((".yaml", ".yml")):
        import yaml

        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _build_enterprise(spec: dict[str, Any]) -> EnterpriseConfig:
    masking = tuple(MaskingRule(**rule) for rule in spec.get("masking", []))
    semantic = tuple(SemanticValidatorConfig(**val) for val in spec.get("semantic", []))
    clustering = ClusterConfig(**spec["clustering"]) if spec.get("clustering") else None
    scalar_keys = (
        "actor",
        "enable_masking",
        "enable_clustering",
        "enable_validation",
        "enable_lineage",
        "fail_under_trust",
    )
    kwargs = {key: spec[key] for key in scalar_keys if key in spec}
    return EnterpriseConfig(masking=masking, semantic=semantic, clustering=clustering, **kwargs)


def cmd_clean(args: argparse.Namespace) -> int:
    file_clean: dict[str, Any] = {}
    ec = EnterpriseConfig()
    if args.config:
        data = _load_config_file(args.config)
        file_clean = data.get("clean", {})
        ec = _build_enterprise(data.get("enterprise", {}))

    overrides = {"strategy": args.strategy} if args.strategy else {}
    merged_clean = {**file_clean, **overrides}
    clean_config = merge_options(None, **merged_clean) if merged_clean else None

    extra_masks = []
    for spec in args.mask or []:
        column, _, strategy = spec.partition(":")
        extra_masks.append(
            MaskingRule(name=f"cli_{column}", columns=(column,), strategy=strategy or "hash")
        )
    masking = tuple(ec.masking) + tuple(extra_masks)

    clustering = ec.clustering
    enable_clustering = ec.enable_clustering
    if args.cluster:
        clustering = ClusterConfig(columns=tuple(args.cluster))
        enable_clustering = True

    fail_under = (
        args.fail_under_trust if args.fail_under_trust is not None else ec.fail_under_trust
    )
    ec = ec.with_overrides(
        masking=masking,
        clustering=clustering,
        enable_clustering=enable_clustering,
        fail_under_trust=fail_under,
    )

    df = _read_frame(args.input, args.in_format)
    result = clean_enterprise(df, clean_config=clean_config, enterprise=ec, actor=args.actor)

    if args.output:
        _write_frame(result.data, args.output, args.out_format)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(result.quality.to_json())
    if args.lineage:
        result.lineage.emit(args.lineage)
    if not args.quiet:
        print(result.summary())
    return 0 if result.passed_gate else 1


def cmd_profile(args: argparse.Namespace) -> int:
    df = _read_frame(args.input, args.in_format)
    profile = build_profile(df, CleanConfig())
    if args.json:
        print(json.dumps(profile.to_dict(), default=str, indent=2))
    else:
        print(profile)
    return 0


def cmd_trust(args: argparse.Namespace) -> int:
    df = _read_frame(args.input, args.in_format)
    score = compute_trust_score(df)
    if args.json:
        print(json.dumps(score.to_dict(), indent=2))
    else:
        print(score)
    if args.fail_under is not None and score.overall < args.fail_under:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="freshdata", description="freshdata enterprise CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("clean", help="clean a file and emit quality/lineage reports")
    clean.add_argument("input")
    clean.add_argument("-o", "--output")
    clean.add_argument("--in-format", choices=("csv", "parquet", "json"))
    clean.add_argument("--out-format", choices=("csv", "parquet", "json"))
    clean.add_argument("--config", help="JSON/YAML config with 'clean' and 'enterprise' keys")
    clean.add_argument("--strategy", choices=("conservative", "balanced", "aggressive"))
    clean.add_argument("--mask", action="append", metavar="COL:STRATEGY",
                       help="mask a column, e.g. email:hash or ssn:regex_scrub (repeatable)")
    clean.add_argument("--cluster", action="append", metavar="COL",
                       help="fuzzy-cluster a text column (repeatable)")
    clean.add_argument("--report", help="write the JSON quality report here")
    clean.add_argument("--lineage", help="write OpenLineage JSON here")
    clean.add_argument("--fail-under-trust", type=float, metavar="SCORE",
                       help="exit non-zero if the post-clean trust score is below this")
    clean.add_argument("--actor", help="who ran this (recorded in lineage)")
    clean.add_argument("--quiet", action="store_true")
    clean.set_defaults(func=cmd_clean)

    profile = subparsers.add_parser("profile", help="print a read-only profile of a file")
    profile.add_argument("input")
    profile.add_argument("--in-format", choices=("csv", "parquet", "json"))
    profile.add_argument("--json", action="store_true")
    profile.set_defaults(func=cmd_profile)

    trust = subparsers.add_parser("trust", help="print the Data Trust Score of a file")
    trust.add_argument("input")
    trust.add_argument("--in-format", choices=("csv", "parquet", "json"))
    trust.add_argument("--json", action="store_true")
    trust.add_argument("--fail-under", type=float, metavar="SCORE",
                       help="exit non-zero if the trust score is below this")
    trust.set_defaults(func=cmd_trust)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse *argv* (or ``sys.argv``) and dispatch. Returns an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
