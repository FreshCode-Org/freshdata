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
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import CleanConfig, merge_options
from ..context import PolicyError
from ..insight import insight_report, trust_gate_report
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


def _load_profile_arg(path: str, *, quiet: bool = False) -> tuple[Any, int]:
    """Load a .fdprofile for the CLI: (profile, 0) or (None, exit_code)."""
    from ..learning import load_profile  # noqa: PLC0415 - lazy import
    from ..learning.types import ProfileError  # noqa: PLC0415

    try:
        profile = load_profile(path)
    except ProfileError as exc:
        print(f"error: cannot load profile {path}: {exc}")
        return None, 2
    except (OSError, ValueError) as exc:
        print(f"error: cannot read profile {path}: {exc}")
        return None, 2
    if getattr(profile.manifest, "contains_raw_values", False) and not quiet:
        print(
            "warning: profile contains raw sensitive values "
            "(learned with privacy='none', include_sensitive=True)"
        )
    return profile, 0


def cmd_clean(args: argparse.Namespace) -> int:
    if getattr(args, "engine", None) and args.engine != "pandas":
        if getattr(args, "context_file", None):
            print("error: --context-file is only supported on the pandas engine")
            return 2
        if getattr(args, "profile", None):
            print("error: --profile is only supported on the pandas engine")
            return 2
        return _cmd_clean_engine(args)
    learned_profile = None
    if getattr(args, "profile", None):
        learned_profile, code = _load_profile_arg(args.profile, quiet=args.quiet)
        if learned_profile is None:
            return code
    file_clean: dict[str, Any] = {}
    ec = EnterpriseConfig()
    if args.config:
        data = _load_config_file(args.config)
        file_clean = data.get("clean", {})
        ec = _build_enterprise(data.get("enterprise", {}))

    overrides: dict[str, Any] = {"strategy": args.strategy} if args.strategy else {}
    if getattr(args, "semantic_mode", None):
        overrides["semantic_mode"] = args.semantic_mode
    if getattr(args, "semantic_backends", None):
        overrides["semantic_backends"] = _parse_backends(args.semantic_backends)
    if getattr(args, "context_file", None):
        overrides["context"] = Path(args.context_file).read_text(encoding="utf-8")
        if getattr(args, "strict", False):
            overrides["strict"] = True
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
    try:
        result = clean_enterprise(
            df,
            clean_config=clean_config,
            enterprise=ec,
            actor=args.actor,
            profile=learned_profile,
        )
    except PolicyError as exc:
        print(f"error: {exc}")
        return 2

    if args.output:
        _write_frame(result.data, args.output, args.out_format)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(result.quality.to_json())
    if args.lineage:
        result.lineage.emit(args.lineage)
    if not args.quiet:
        print(result.summary())
        for event in result.clean_report.fallback_events:
            if event.get("fallback_step") == "semantic":
                print(
                    f"note: semantic backend '{event.get('backend')}' skipped: "
                    f"{event.get('fallback_reason')}"
                )
        replay = getattr(result.clean_report, "profile_replay", None)
        if replay is not None and not replay.get("ok"):
            reasons = replay.get("reasons") or ["severe schema drift"]
            print(f"note: learned profile not replayed: {reasons[0]}")
        elif replay is not None and replay.get("severity") == "mild":
            print("note: learned profile partially replayed (mild schema drift)")
    return 0 if result.passed_gate else 1


def _cmd_clean_engine(args: argparse.Namespace) -> int:
    """Clean via a scalable execution backend (polars / duckdb / spark / auto).

    The input path is handed straight to the backend so it can read it natively
    (DuckDB/Polars scan files in place; Spark reads via its own readers). The
    cleaned result is converted to pandas for writing and a CleanReport summary.
    """
    import freshdata as fd

    from ..execution import EngineConfig

    overrides = {"strategy": args.strategy} if args.strategy else {}
    clean_config = merge_options(None, **overrides) if overrides else None

    engine_config = EngineConfig(engine=args.engine, output_format="pandas")
    if getattr(args, "memory_limit_gb", None) is not None:
        engine_config.memory_limit_gb = args.memory_limit_gb
    if getattr(args, "fallback_policy", None):
        engine_config.fallback_policy = args.fallback_policy

    try:
        cleaned, report = fd.clean(
            args.input,
            config=clean_config,
            engine=args.engine,
            engine_config=engine_config,
            return_report=True,
        )
    except fd.FallbackError as exc:
        print(f"freshdata: fallback refused: {exc}", file=sys.stderr)
        return 1

    if args.output:
        _write_frame(cleaned, args.output, args.out_format)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
    if not args.quiet:
        print(report.summary())
        if report.backend_differences:
            print(
                f"\n{len(report.backend_differences)} backend difference(s) recorded "
                f"(see report.backend_differences)."
            )
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    # Back-compat dispatch: `freshdata profile data.csv` keeps its original
    # data-profiling meaning; `freshdata profile audit|diff|merge …` routes
    # to the learned-profile tools.
    if args.input in ("audit", "diff", "merge"):
        return _cmd_profile_tools(args)
    if getattr(args, "paths", None):
        print(
            f"error: unexpected extra arguments {args.paths}; "
            "did you mean 'freshdata profile audit|diff|merge'?"
        )
        return 2
    df = _read_frame(args.input, args.in_format)
    profile = build_profile(df, CleanConfig())
    if args.json:
        report = insight_report(df, profile=profile, dataset_name=args.input)
        print(json.dumps(report.to_dict(), default=str, indent=2))
    else:
        print(profile)
    return 0


def _cmd_profile_tools(args: argparse.Namespace) -> int:
    """`freshdata profile audit|diff|merge …` — learned .fdprofile tools."""
    tool = args.input
    paths = list(getattr(args, "paths", []) or [])
    if tool == "audit":
        if len(paths) != 1:
            print("usage: freshdata profile audit PROFILE.fdprofile [--json]")
            return 2
        profile, code = _load_profile_arg(paths[0])
        if profile is None:
            return code
        audit = profile.audit()
        if args.json:
            print(json.dumps(audit.to_dict(), default=str, indent=2))
        else:
            print(audit.render())
        return 0
    if tool == "diff":
        if len(paths) != 2:
            print("usage: freshdata profile diff A.fdprofile B.fdprofile")
            return 2
        left, code = _load_profile_arg(paths[0])
        if left is None:
            return code
        right, code = _load_profile_arg(paths[1])
        if right is None:
            return code
        diff = left.diff(right)
        print(str(diff))
        return 0 if diff.is_empty else 1
    # merge
    if len(paths) != 2:
        print(
            "usage: freshdata profile merge A.fdprofile B.fdprofile "
            "-o MERGED.fdprofile [--strategy STRATEGY]"
        )
        return 2
    if not getattr(args, "output", None):
        print("error: profile merge requires -o/--output for the merged profile")
        return 2
    left, code = _load_profile_arg(paths[0])
    if left is None:
        return code
    right, code = _load_profile_arg(paths[1])
    if right is None:
        return code
    from ..learning.merge import ProfileMergeError  # noqa: PLC0415 - lazy import

    try:
        merged = left.merge(right, strategy=args.strategy)
    except ProfileMergeError as exc:
        print(f"error: {exc}")
        return 2
    from ..learning import save_profile  # noqa: PLC0415 - lazy import

    save_profile(merged, args.output)
    print(f"merged profile written to {args.output} ({merged.profile_id})")
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """`freshdata learn RAW CLEAN -o OUT.fdprofile` — learn a profile."""
    from ..learning import learn as learn_profile  # noqa: PLC0415 - lazy import
    from ..learning import save_profile  # noqa: PLC0415
    from ..learning.types import ProfileError  # noqa: PLC0415

    messy = _read_frame(args.raw, args.in_format)
    clean_df = _read_frame(args.clean, args.in_format)
    context = args.context
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")
    key = args.key.split(",") if args.key and "," in args.key else args.key
    try:
        profile = learn_profile(
            messy,
            clean_df,
            context=context,
            key=key,
            dataset_id=args.dataset_id,
            privacy=args.privacy,
            include_sensitive=args.include_sensitive,
            min_support=args.min_support,
            min_precision=args.min_precision,
        )
    except (ProfileError, ValueError, TypeError) as exc:
        print(f"error: {exc}")
        return 2
    save_profile(profile, args.output)
    if not args.quiet:
        print(f"profile written to {args.output} ({profile.profile_id})")
        print(profile.summary())
        if profile.manifest.contains_raw_values:
            print(
                "warning: profile contains raw sensitive values "
                "(privacy='none', include_sensitive=True)"
            )
    return 0


def cmd_trust(args: argparse.Namespace) -> int:
    df = _read_frame(args.input, args.in_format)
    score = compute_trust_score(df)
    if args.json:
        report = trust_gate_report(df, score, fail_under=args.fail_under, dataset_name=args.input)
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(score)
    if args.fail_under is not None and score.overall < args.fail_under:
        return 1
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a file against a suite or contract. Exit 0 pass, 1 fail, 2 usage."""
    from ..validation_suite import ValidationSuite, run_suite

    if bool(args.suite) == bool(args.contract):
        print(
            "freshdata validate: exactly one of --suite or --contract is required",
            file=sys.stderr,
        )
        return 2
    try:
        if args.suite:
            suite = ValidationSuite.load(args.suite)
        else:
            from .contracts import DataContract

            with open(args.contract, encoding="utf-8") as fh:
                suite = ValidationSuite.from_contract(DataContract.from_dict(json.load(fh)))
    except FileNotFoundError:
        raise
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"freshdata validate: could not load rules: {exc}", file=sys.stderr)
        return 2

    df = _read_frame(args.input, args.in_format)
    result = run_suite(df, suite)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(result.to_json())
    if not args.quiet:
        verdict = "PASS" if result.passed else "FAIL"
        print(
            f"freshdata validate: {verdict} — {result.n_errors} error(s), "
            f"{result.n_warnings} warning(s) against suite {suite.name!r}"
        )
        for f in result.report.findings:
            if f.status != "passed":
                print(f"  [{f.status}] {f.check_id}: {f.message}")
    return 0 if result.passed else 1


def cmd_quality_ops(args: argparse.Namespace) -> int:
    from ..findings import findings_from_dict
    from ..integrations.quality_ops import export_quality_ops

    with open(args.input, encoding="utf-8") as fh:
        report_dict = json.load(fh)
    findings = findings_from_dict(report_dict)

    df = _read_frame(args.data, None) if args.data else None
    model_name = args.model_name or Path(args.input).stem
    suite_name = args.suite_name or f"{model_name}_suite"

    result = export_quality_ops(
        findings,
        model_name=model_name,
        suite_name=suite_name,
        dbt_path=args.dbt,
        gx_path=args.gx,
        exception_table_path=args.exceptions,
        df=df,
        include_pii=args.include_pii,
        exceptions_format=args.exceptions_format,
    )
    if args.lineage:
        with open(args.lineage, "w", encoding="utf-8") as fh:
            json.dump(result.lineage_event, fh, indent=2, default=str)
    if not args.quiet:
        print(f"freshdata quality-ops: {len(result.findings)} finding(s)")
        for label, dest in (
            ("dbt", result.dbt_path),
            ("gx", result.gx_path),
            ("exceptions", result.exception_table_path),
            ("lineage", args.lineage),
        ):
            if dest:
                print(f"  {label}: {dest}")
    return 0


def cmd_policy_compile(args: argparse.Namespace) -> int:
    """Compile a rules text file into a context policy and print its summary."""
    from ..context import compile_context

    text = Path(args.rules).read_text(encoding="utf-8")
    columns = None
    if args.schema:
        frame = _read_frame(args.schema, args.in_format)
        columns = [str(c) for c in frame.columns]
    try:
        policy = compile_context(text, columns=columns, strict=args.strict)
    except PolicyError as exc:
        print(f"error: {exc}")
        return 2
    print(policy.summary())
    if args.output:
        policy.to_json(args.output)
        print(f"policy JSON written to {args.output}")
    return 0


def _parse_backends(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def cmd_models_status(args: argparse.Namespace) -> int:
    """Show install/verification status for every registered model (offline)."""
    del args
    from .. import models

    print(f"model directory: {models.model_dir()}")
    for model_id, info in models.status().items():
        installed = "installed" if info["installed"] else "missing"
        verified = {True: "verified", False: "unverified", None: "-"}[info["verified"]]
        note = f"  ({info['note']})" if info["note"] else ""
        print(f"{model_id:<20} {installed:<10} {verified:<11} {info['quantization']}{note}")
    return 0


def cmd_models_pull(args: argparse.Namespace) -> int:
    """Explicitly download one model; verifies the pinned checksum."""
    from .. import models

    try:
        path = models.pull(args.model_id, force=args.force)
    except models.ModelError as exc:
        print(f"error: {exc}")
        return 2
    print(f"pulled {args.model_id} -> {path}")
    return 0


def _plan_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {"verbose": False}
    if getattr(args, "semantic_mode", None):
        overrides["semantic_mode"] = args.semantic_mode
    if getattr(args, "semantic_backends", None):
        overrides["semantic_backends"] = _parse_backends(args.semantic_backends)
    if getattr(args, "context_file", None):
        overrides["context"] = Path(args.context_file).read_text(encoding="utf-8")
    if getattr(args, "strict", False):
        overrides["strict"] = True
    return overrides


def cmd_plan(args: argparse.Namespace) -> int:
    """Suggest an executable repair plan for a file and write it as JSON."""
    import freshdata as fd  # noqa: PLC0415 — full API only when this command runs

    df = _read_frame(args.input, args.in_format)
    try:
        plan = fd.suggest_plan(df, **_plan_overrides(args))
    except PolicyError as exc:
        print(f"error: {exc}")
        return 2
    repair_plan = plan.repair_plan
    if repair_plan is None:
        print(
            "no repair plan: pass --context-file and/or --semantic-mode so planned actions exist"
        )
        return 2
    if getattr(args, "approve_all", None):
        repair_plan.approve_all(max_risk=args.approve_all)
    if not args.quiet:
        print(repair_plan.summary())
    if args.out:
        repair_plan.to_json(args.out)
        print(f"plan JSON written to {args.out}")
    return 0


def cmd_apply_plan(args: argparse.Namespace) -> int:
    """Apply a reviewed plan JSON to a file: approved actions only."""
    import freshdata as fd  # noqa: PLC0415 — full API only when this command runs

    df = _read_frame(args.input, args.in_format)
    plan = fd.RepairPlan.from_json(Path(args.plan))
    try:
        cleaned, report = fd.apply_plan(df, plan, allow_drift=args.allow_drift)
    except fd.PlanDriftError as exc:
        print(f"error: {exc}")
        return 2
    except fd.ProtectedColumnError as exc:
        print(f"error: {exc}")
        return 3
    if args.output:
        _write_frame(cleaned, args.output, args.out_format)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
    if not args.quiet:
        print(report.summary())
        print(f"decisions_hash: {report.decisions_hash}")
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
    clean.add_argument(
        "--engine",
        choices=("pandas", "polars", "duckdb", "spark", "freshcore", "auto"),
        default="pandas",
        help="execution backend; non-pandas engines run the scalable/out-of-core path",
    )
    clean.add_argument(
        "--fallback-policy",
        choices=("allow", "warn", "error"),
        help="what to do when a native engine must delegate to pandas: "
        "allow (record), warn, or error (refuse before materializing)",
    )
    clean.add_argument(
        "--memory-limit-gb",
        type=float,
        metavar="GB",
        help="memory budget for the DuckDB engine before it spills to disk",
    )
    clean.add_argument(
        "--mask",
        action="append",
        metavar="COL:STRATEGY",
        help="mask a column, e.g. email:hash or ssn:regex_scrub (repeatable)",
    )
    clean.add_argument(
        "--cluster",
        action="append",
        metavar="COL",
        help="fuzzy-cluster a text column (repeatable)",
    )
    clean.add_argument("--report", help="write the JSON quality report here")
    clean.add_argument("--lineage", help="write OpenLineage JSON here")
    clean.add_argument(
        "--fail-under-trust",
        type=float,
        metavar="SCORE",
        help="exit non-zero if the post-clean trust score is below this",
    )
    clean.add_argument("--actor", help="who ran this (recorded in lineage)")
    clean.add_argument("--quiet", action="store_true")
    clean.add_argument(
        "--context-file",
        metavar="PATH",
        help="text file with natural-language cleaning rules, compiled "
        "deterministically into a context policy for this run",
    )
    clean.add_argument(
        "--strict",
        action="store_true",
        help="fail (exit 2) on unresolved or unparsed context lines",
    )
    clean.add_argument(
        "--semantic-mode",
        choices=("off", "assist", "review", "auto"),
        help="semantic cleaning mode for this run (default: off)",
    )
    clean.add_argument(
        "--profile",
        metavar="X.fdprofile",
        help="replay a learned cleaning profile (see 'freshdata learn')",
    )
    clean.add_argument(
        "--semantic-backends",
        metavar="LIST",
        help="comma-separated proposal backends in trust order, e.g. "
        "deterministic,memory,embedding — embedding needs the "
        "[semantic] extra and a pulled model and is skipped (with a "
        "report note) when either is missing",
    )
    clean.set_defaults(func=cmd_clean)

    plan_p = subparsers.add_parser(
        "plan", help="suggest an executable repair plan (review it, then apply-plan)"
    )
    plan_p.add_argument("input")
    plan_p.add_argument("--in-format", choices=("csv", "parquet", "json"))
    plan_p.add_argument(
        "--context-file", metavar="PATH", help="text file with natural-language cleaning rules"
    )
    plan_p.add_argument(
        "--semantic-mode",
        choices=("off", "assist", "review", "auto"),
        default="review",
        help="planning posture (default: review)",
    )
    plan_p.add_argument(
        "--semantic-backends",
        metavar="LIST",
        help="comma-separated proposal backends in trust order, e.g. "
        "deterministic,memory,embedding",
    )
    plan_p.add_argument(
        "--strict",
        action="store_true",
        help="fail (exit 2) on unresolved or unparsed context lines",
    )
    plan_p.add_argument(
        "--approve-all",
        choices=("low", "medium", "high"),
        metavar="MAX_RISK",
        help="pre-approve all actions at or below this risk before writing",
    )
    plan_p.add_argument("--out", metavar="plan.json", help="write the plan JSON here")
    plan_p.add_argument("--quiet", action="store_true")
    plan_p.set_defaults(func=cmd_plan)

    apply_p = subparsers.add_parser(
        "apply-plan", help="execute exactly the approved actions of a plan JSON"
    )
    apply_p.add_argument("input")
    apply_p.add_argument("--plan", required=True, metavar="plan.json")
    apply_p.add_argument("-o", "--output")
    apply_p.add_argument("--in-format", choices=("csv", "parquet", "json"))
    apply_p.add_argument("--out-format", choices=("csv", "parquet", "json"))
    apply_p.add_argument(
        "--report",
        metavar="audit.json",
        help="write the JSON audit report (includes decisions_hash) here",
    )
    apply_p.add_argument(
        "--allow-drift",
        action="store_true",
        help="apply even if the file changed since the plan was suggested",
    )
    apply_p.add_argument("--quiet", action="store_true")
    apply_p.set_defaults(func=cmd_apply_plan)

    profile = subparsers.add_parser(
        "profile",
        help="print a read-only profile of a file, or manage learned "
        ".fdprofile files (profile audit|diff|merge)",
    )
    profile.add_argument("input", help="a data file, or one of: audit, diff, merge")
    profile.add_argument("paths", nargs="*", help=".fdprofile path(s) for audit/diff/merge")
    profile.add_argument("--in-format", choices=("csv", "parquet", "json"))
    profile.add_argument("--json", action="store_true")
    profile.add_argument(
        "-o", "--output", metavar="MERGED.fdprofile", help="output path for 'profile merge'"
    )
    profile.add_argument(
        "--strategy",
        choices=("union_min_precision", "prefer_self", "prefer_other", "error_on_conflict"),
        default="union_min_precision",
        help="merge strategy for 'profile merge'",
    )
    profile.set_defaults(func=cmd_profile)

    learn_p = subparsers.add_parser(
        "learn",
        help="learn a reusable cleaning profile from a (messy, clean) file pair",
    )
    learn_p.add_argument("raw", help="the messy input file")
    learn_p.add_argument("clean", help="the corresponding cleaned file")
    learn_p.add_argument(
        "-o", "--output", required=True, metavar="OUT.fdprofile", help="where to write the profile"
    )
    learn_p.add_argument(
        "--key", metavar="COL[,COL…]", help="key column(s) used to align the two files"
    )
    learn_p.add_argument(
        "--context",
        metavar="TEXT",
        help="natural-language cleaning rules (protected columns etc.)",
    )
    learn_p.add_argument("--context-file", metavar="PATH", help="read --context text from a file")
    learn_p.add_argument(
        "--dataset-id", metavar="ID", help="stable dataset identifier recorded in the profile"
    )
    learn_p.add_argument(
        "--privacy",
        choices=("mask", "none"),
        default="mask",
        help="mask sensitive literals (default) or store raw",
    )
    learn_p.add_argument(
        "--include-sensitive",
        action="store_true",
        help="with --privacy none, store raw sensitive literals",
    )
    learn_p.add_argument(
        "--min-support",
        type=int,
        default=5,
        metavar="N",
        help="minimum occurrences before a pattern is learned",
    )
    learn_p.add_argument(
        "--min-precision",
        type=float,
        default=0.98,
        metavar="X",
        help="minimum holdout precision before a rule replays",
    )
    learn_p.add_argument("--in-format", choices=("csv", "parquet", "json"))
    learn_p.add_argument("--quiet", action="store_true")
    learn_p.set_defaults(func=cmd_learn)

    trust = subparsers.add_parser("trust", help="print the Data Trust Score of a file")
    trust.add_argument("input")
    trust.add_argument("--in-format", choices=("csv", "parquet", "json"))
    trust.add_argument("--json", action="store_true")
    trust.add_argument(
        "--fail-under",
        type=float,
        metavar="SCORE",
        help="exit non-zero if the trust score is below this",
    )
    trust.set_defaults(func=cmd_trust)

    validate_p = subparsers.add_parser(
        "validate",
        help="validate a file against a suite/contract; exit 0 pass, 1 fail, 2 usage",
    )
    validate_p.add_argument("input")
    validate_p.add_argument("--in-format", choices=("csv", "parquet", "json"))
    validate_p.add_argument("--suite", metavar="suite.json", help="a saved ValidationSuite")
    validate_p.add_argument(
        "--contract", metavar="contract.json", help="a saved DataContract (to_dict JSON)"
    )
    validate_p.add_argument("--json", metavar="OUT", help="also write the full result JSON here")
    validate_p.add_argument("--quiet", action="store_true")
    validate_p.set_defaults(func=cmd_validate)

    qops = subparsers.add_parser(
        "quality-ops",
        help="export findings from a report.json to dbt/GX/exception/lineage artifacts",
    )
    qops.add_argument("input", help="path to a freshdata report JSON (CleanReport.to_dict)")
    qops.add_argument("--dbt", metavar="schema.yml", help="write dbt generic tests YAML here")
    qops.add_argument("--gx", metavar="suite.json", help="write a Great Expectations suite here")
    qops.add_argument(
        "--exceptions",
        metavar="PATH",
        help="write an exception table here (.csv/.parquet/.duckdb)",
    )
    qops.add_argument(
        "--exceptions-format",
        choices=("csv", "parquet", "duckdb"),
        help="exception-table format (else inferred from the extension)",
    )
    qops.add_argument(
        "--lineage",
        metavar="lineage.json",
        help="write the OpenLineage event (with artifact facets) here",
    )
    qops.add_argument(
        "--data", metavar="PATH", help="optional source file to enrich exception observed_values"
    )
    qops.add_argument("--model-name", help="dbt model name (default: report filename stem)")
    qops.add_argument("--suite-name", help="GX suite name (default: <model>_suite)")
    qops.add_argument(
        "--include-pii",
        action="store_true",
        help="reveal observed values in the exception table (default: redacted)",
    )
    qops.add_argument("--quiet", action="store_true")
    qops.set_defaults(func=cmd_quality_ops)

    policy = subparsers.add_parser(
        "policy", help="compile and inspect natural-language context policies"
    )
    policy_sub = policy.add_subparsers(dest="policy_command", required=True)
    compile_p = policy_sub.add_parser(
        "compile", help="compile a rules text file into a reviewable policy"
    )
    compile_p.add_argument("rules", help="path to the rules text file")
    compile_p.add_argument(
        "--schema", metavar="PATH", help="data file whose columns the policy resolves against"
    )
    compile_p.add_argument(
        "--in-format",
        choices=("csv", "parquet", "json"),
        help="format of the --schema file (default: by extension)",
    )
    compile_p.add_argument(
        "--output", metavar="policy.json", help="write the compiled policy JSON here"
    )
    compile_p.add_argument(
        "--strict", action="store_true", help="fail (exit 2) on unresolved or unparsed lines"
    )
    compile_p.set_defaults(func=cmd_policy_compile)

    models_p = subparsers.add_parser(
        "models", help="manage optional local semantic models (never auto-downloaded)"
    )
    models_sub = models_p.add_subparsers(dest="models_command", required=True)
    status_p = models_sub.add_parser(
        "status", help="show install/verification status; offline, works without [semantic]"
    )
    status_p.set_defaults(func=cmd_models_status)
    pull_p = models_sub.add_parser(
        "pull", help="explicitly download a model and verify its checksum"
    )
    pull_p.add_argument("model_id", help="e.g. fd-col-encoder-v1")
    pull_p.add_argument("--force", action="store_true", help="re-download even if present")
    pull_p.set_defaults(func=cmd_models_pull)

    from ..streaming._cli import add_stream_subparsers

    add_stream_subparsers(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse *argv* (or ``sys.argv``) and dispatch. Returns an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        # A wrong input path is routine CLI misuse, not a crash: report it in
        # one line instead of a traceback. Everything else propagates intact.
        print(f"freshdata: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
