"""Action-oriented presentation report for FreshData output surfaces.

``FreshDataInsightReport`` is a serializable view model over objects FreshData
already computes: profile findings, inferred roles, ``CleanReport`` actions,
trust-score dictionaries, and lineage metadata. It does not make cleaning
decisions; it packages those decisions for notebooks, IDEs, CLIs, and CI.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ._util import memory_bytes
from .config import CleanConfig, merge_options
from .engine.context import build_contexts
from .profile import Profile, build_profile
from .render.mixins import HtmlReprMixin
from .report import CleanReport

SCHEMA_VERSION = "freshdata.insight.v1"


@dataclass
class FreshDataInsightReport(HtmlReprMixin):
    """JSON-first report for FreshData's decision/action output layer."""

    _render_kind = "insight_report"

    schema_version: str
    report_type: str
    dataset: dict[str, Any]
    run: dict[str, Any]
    summary: dict[str, Any]
    issues: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    trust: dict[str, Any] | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    surfaces: dict[str, Any] = field(default_factory=dict)
    extensions_required: list[dict[str, str]] = field(default_factory=list)
    strategy_comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "dataset": dict(self.dataset),
            "run": dict(self.run),
            "summary": dict(self.summary),
            "issues": [dict(issue) for issue in self.issues],
            "actions": [dict(action) for action in self.actions],
            "trust": dict(self.trust) if self.trust is not None else None,
            "lineage": dict(self.lineage),
            "surfaces": dict(self.surfaces),
            "extensions_required": [dict(item) for item in self.extensions_required],
        }
        if self.strategy_comparison is not None:
            payload["strategy_comparison"] = dict(self.strategy_comparison)
        return payload

    def summary_text(self) -> str:
        highest = self.summary.get("highest_severity", "none")
        return (
            f"freshdata insight report - {self.dataset.get('rows', 0):,} rows x "
            f"{self.dataset.get('columns', 0):,} columns\n"
            f"  issues: {self.summary.get('issue_count', 0)}   "
            f"actions: {self.summary.get('action_count', 0)}   "
            f"highest severity: {highest}\n"
            f"  next: {self.summary.get('recommended_next_step', '')}"
        ).rstrip()

    def __str__(self) -> str:
        return self.summary_text()

    def __repr__(self) -> str:
        return (
            f"<FreshDataInsightReport: {self.summary.get('issue_count', 0)} issue(s), "
            f"{self.summary.get('action_count', 0)} action(s)>"
        )


def insight_report(
    df: pd.DataFrame,
    config: CleanConfig | None = None,
    *,
    clean_report: CleanReport | None = None,
    cleaned_df: pd.DataFrame | None = None,
    enterprise_result: Any | None = None,
    profile: Profile | None = None,
    dataset_name: str | None = None,
    trust: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
    compare_strategies: tuple[str, ...] | None = None,
    **options: object,
) -> FreshDataInsightReport:
    """Build a FreshData insight report for profile and clean-result surfaces.

    The function profiles *df* read-only, maps findings to actionable snippets,
    and optionally folds in a real ``CleanReport`` plus trust/lineage payloads.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
    if enterprise_result is not None:
        clean_report = getattr(enterprise_result, "clean_report", clean_report)
        cleaned_df = to_pandas_safe(getattr(enterprise_result, "data", cleaned_df))
        trust = _trust_from_enterprise(enterprise_result)
        lineage = _lineage_from_tracker(getattr(enterprise_result, "lineage", None))

    cfg = merge_options(config, **options)
    prof = profile or build_profile(df, cfg)
    contexts = build_contexts(df, cfg)
    name = dataset_name or "dataframe"
    actions = _actions_from_clean_report(clean_report, before=df, after=cleaned_df)
    action_lookup = _action_lookup(actions)
    issues = _issues_from_profile(prof, contexts, cfg, action_lookup=action_lookup)
    issues.extend(_validation_issues(enterprise_result))
    highest = _highest_severity(issues)
    report_type = "anomaly_insight"
    api = "fd.profile"
    if clean_report is not None or cleaned_df is not None:
        report_type = "clean_impact"
        api = "fd.clean"
    if enterprise_result is not None:
        report_type = "trust_gate"
        api = "enterprise.validation"
    summary = {
        "issue_count": len(issues),
        "action_count": len(actions),
        "highest_severity": highest,
        "recommended_next_step": _recommended_next_step(name, cfg),
    }
    if clean_report is not None:
        summary.update(_clean_summary(clean_report))
    strategy_comparison = _strategy_comparison(df, cfg, compare_strategies)
    surfaces = {
        "jupyter": {"primary_view": "issue_cards_with_action_context"},
        "vscode": {"primary_view": "notebook_table_with_details_panel"},
        "cli": {
            "json_flag": f"freshdata profile {name} --json",
            "ci_summary": _ci_summary(report_type, trust, issues),
        },
        "web": {"entrypoint": "report.show()"},
    }
    extensions = [
        {
            "name": "distribution_sketches",
            "reason": "Before/after distribution panels need compact per-column summaries.",
        },
        {
            "name": "lineage_consumer_impact",
            "reason": (
                "OpenLineage emission exists; downstream consumer traversal needs graph "
                "integration."
            ),
        },
    ]
    return FreshDataInsightReport(
        schema_version=SCHEMA_VERSION,
        report_type=report_type,
        dataset={
            "name": name,
            "rows": prof.n_rows,
            "columns": prof.n_cols,
            "memory_bytes": prof.memory or memory_bytes(df),
        },
        run={
            "freshdata_version": _freshdata_version(),
            "api": api,
            "strategy": cfg.strategy,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": getattr(clean_report, "duration_seconds", 0.0) or 0.0,
        },
        summary=summary,
        issues=issues,
        actions=actions,
        trust=trust,
        lineage=lineage or {
            "openlineage_run_id": None,
            "input_datasets": [],
            "output_datasets": [],
            "impacted_assets": [],
        },
        surfaces=surfaces,
        extensions_required=extensions,
        strategy_comparison=strategy_comparison,
    )


def trust_gate_report(
    df: pd.DataFrame,
    trust_score: Any,
    *,
    fail_under: float | None = None,
    dataset_name: str | None = None,
) -> FreshDataInsightReport:
    """Build a CI-oriented insight payload for ``freshdata trust``."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
    name = dataset_name or "dataframe"
    score = _trust_score_dict(trust_score)
    gate = _gate_dict(score.get("overall"), fail_under)
    trust = {"before": None, "after": score, "delta": None, "gate": gate}
    summary = {
        "issue_count": 0,
        "action_count": 0,
        "highest_severity": "high" if gate["passed"] is False else "none",
        "recommended_next_step": f"Run freshdata profile {name} --json for issue details.",
    }
    return FreshDataInsightReport(
        schema_version=SCHEMA_VERSION,
        report_type="trust_gate",
        dataset={
            "name": name,
            "rows": len(df),
            "columns": df.shape[1],
            "memory_bytes": memory_bytes(df),
        },
        run={
            "freshdata_version": _freshdata_version(),
            "api": "freshdata trust",
            "strategy": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": 0.0,
        },
        summary=summary,
        issues=[],
        actions=[],
        trust=trust,
        lineage={
            "openlineage_run_id": None,
            "input_datasets": [name],
            "output_datasets": [],
            "impacted_assets": [],
        },
        surfaces={
            "jupyter": {"primary_view": "trust_gate_summary"},
            "vscode": {"primary_view": "trust_gate_summary"},
            "cli": {
                "json_flag": f"freshdata trust {name} --json",
                "ci_summary": _ci_summary("trust_gate", trust, []),
            },
            "web": {"entrypoint": "report.show()"},
        },
        extensions_required=[],
    )


def _freshdata_version() -> str:
    module = sys.modules.get("freshdata")
    return str(getattr(module, "__version__", "unknown"))


def _issues_from_profile(
    profile: Profile,
    contexts: dict[Any, Any],
    config: CleanConfig,
    *,
    action_lookup: dict[tuple[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for column in profile.columns:
        if not column.issues:
            continue
        ctx = contexts.get(column.name)
        role = getattr(ctx, "role", "unknown")
        severity = _severity(column.issues, column.missing_pct)
        hint = _action_hint(column.issues, role)
        action_id = (action_lookup or {}).get(
            (column.name, hint),
            f"action.{_slug(column.name)}.{hint}",
        )
        issues.append(
            {
                "id": f"issue.{_slug(column.name)}.{hint}",
                "column": column.name,
                "severity": severity,
                "finding": "; ".join(column.issues),
                "evidence": {
                    "missing_cells": column.missing,
                    "missing_pct": round(column.missing_pct, 2),
                    "dtype": column.dtype,
                    "suggested_dtype": column.suggested_dtype,
                    "unique": column.unique,
                    "sample_values": list(column.sample_values),
                },
                "inferred_role": role,
                "recommended_action_id": action_id,
                "fix_code": _fix_code(
                    column.name,
                    column.issues,
                    role,
                    config,
                    missing_pct=column.missing_pct,
                    suggested_dtype=column.suggested_dtype,
                ),
                "backend_requirement": "presentation_only",
            }
        )
    return issues


def _actions_from_clean_report(
    report: CleanReport | None,
    *,
    before: pd.DataFrame | None = None,
    after: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if report is None:
        return []
    actions: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for action in report.actions:
        entry = CleanReport._action_dict(action)
        base_id = f"action.{_slug(action.column or 'table')}.{_slug(action.step)}"
        seen[base_id] = seen.get(base_id, 0) + 1
        entry["id"] = base_id if seen[base_id] == 1 else f"{base_id}.{seen[base_id]}"
        entry["impact"] = _action_impact(action.column, before=before, after=after)
        actions.append(entry)
    return actions


def _action_lookup(actions: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for action in actions:
        column = action.get("column")
        if not column:
            continue
        step = str(action.get("step") or "")
        hint = "dtype" if step == "fix_dtypes" else step
        lookup.setdefault((str(column), hint), str(action["id"]))
    return lookup


def _action_impact(
    column: str | None,
    *,
    before: pd.DataFrame | None,
    after: pd.DataFrame | None,
) -> dict[str, Any]:
    if before is None and after is None:
        return {"before": {}, "after": {}}
    if column is None:
        return {
            "before": _frame_stats(before),
            "after": _frame_stats(after),
        }
    return {
        "before": _column_stats(before, column),
        "after": _column_stats(after, column),
    }


def _severity(issues: list[str], missing_pct: float) -> str:
    joined = " ".join(issues).lower()
    if missing_pct >= 50.0 or "constant column" in joined:
        return "high"
    if missing_pct >= 30.0 or "outlier" in joined or "mixed value types" in joined:
        return "medium"
    return "low"


def _frame_stats(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None:
        return {}
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
    }


def _column_stats(df: pd.DataFrame | None, column: str) -> dict[str, Any]:
    if df is None or column not in df.columns:
        return {}
    series = df[column]
    missing = int(series.isna().sum())
    return {
        "dtype": str(series.dtype),
        "missing": missing,
        "missing_pct": round(100.0 * missing / len(series), 2) if len(series) else 0.0,
        "unique": _safe_unique(series),
    }


def _safe_unique(series: pd.Series) -> int | None:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return None


def _clean_summary(report: CleanReport) -> dict[str, Any]:
    return {
        "rows_before": report.rows_before,
        "rows_after": report.rows_after,
        "cols_before": report.cols_before,
        "cols_after": report.cols_after,
        "missing_before": report.missing_before,
        "missing_after": report.missing_after,
        "missing_delta": report.missing_after - report.missing_before,
        "duplicates_removed": report.duplicates_removed,
        "outliers_handled": report.outliers_handled,
        "columns_dropped": list(report.columns_dropped),
        "columns_imputed": list(report.columns_imputed),
        "columns_preserved": list(report.columns_preserved),
    }


def _strategy_comparison(
    df: pd.DataFrame,
    config: CleanConfig,
    strategies: tuple[str, ...] | None,
) -> dict[str, Any] | None:
    if not strategies:
        return None
    from .plan import compare_plans  # noqa: PLC0415 - avoid importing planner unless requested

    frame = compare_plans(df, strategies=strategies, config=config)
    records = _frame_records(frame)
    present = {str(row.get("strategy")) for row in records}
    for strategy in strategies:
        if strategy not in present:
            records.append(
                {
                    "column": None,
                    "strategy": strategy,
                    "missing_model": None,
                    "outlier_action": None,
                    "n_outliers": 0,
                }
            )
    return {
        "source": "fd.compare_plans",
        "strategies": list(strategies),
        "command": f"fd.compare_plans(df, strategies={strategies!r})",
        "records": records,
    }


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in frame.to_dict("records"):
        records.append({str(k): _json_scalar(v) for k, v in raw.items()})
    return records


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _trust_from_enterprise(result: Any) -> dict[str, Any]:
    before = _trust_score_dict(getattr(result, "trust_before", None))
    after = _trust_score_dict(getattr(result, "trust_after", None))
    threshold = getattr(result, "fail_under_trust", None)
    before_overall = before.get("overall")
    after_overall = after.get("overall")
    delta = None
    if before_overall is not None and after_overall is not None:
        delta = float(after_overall) - float(before_overall)
    return {
        "before": before,
        "after": after,
        "delta": _round_or_none(delta),
        "gate": _gate_dict(float(after_overall) if after_overall is not None else None, threshold),
    }


def _trust_score_dict(score: Any) -> dict[str, Any]:
    if score is None:
        return {}
    if hasattr(score, "to_dict"):
        return score.to_dict()
    if isinstance(score, dict):
        return dict(score)
    return {"overall": float(score)}


def _gate_dict(score: float | None, threshold: float | None) -> dict[str, Any]:
    passed = None if threshold is None or score is None else score >= threshold
    return {
        "passed": passed,
        "threshold": threshold,
        "score": score,
    }


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)


def _lineage_from_tracker(tracker: Any) -> dict[str, Any]:
    if tracker is None:
        return {
            "openlineage_run_id": None,
            "input_datasets": [],
            "output_datasets": [],
            "impacted_assets": [],
        }
    payload = tracker.to_dict() if hasattr(tracker, "to_dict") else {}
    return {
        "openlineage_run_id": payload.get("run_id"),
        "input_datasets": [payload.get("input", "input")],
        "output_datasets": [payload.get("output", "output")],
        "impacted_assets": [],
        "events": payload.get("events", []),
    }


def _validation_issues(result: Any | None) -> list[dict[str, Any]]:
    if result is None:
        return []
    report = getattr(result, "validation_report", None)
    columns = getattr(report, "columns", {}) if report is not None else {}
    issues: list[dict[str, Any]] = []
    for column, validation in columns.items():
        if getattr(validation, "n_invalid", 0) <= 0:
            continue
        validator = str(getattr(validation, "validator", "semantic_validator"))
        samples = list(getattr(validation, "invalid_samples", ()))
        issues.append(
            {
                "id": f"semantic.{_slug(validator)}.{_slug(str(column))}",
                "column": str(column),
                "severity": "high",
                "finding": (
                    f"{getattr(validation, 'n_invalid', 0)} value(s) failed "
                    f"{validator}"
                ),
                "evidence": {
                    "validator": validator,
                    "n_checked": getattr(validation, "n_checked", 0),
                    "n_invalid": getattr(validation, "n_invalid", 0),
                    "valid_ratio": getattr(validation, "valid_ratio", None),
                    "invalid_samples": samples,
                },
                "inferred_role": "categorical",
                "recommended_action_id": None,
                "fix_code": (
                    "Review source values or update "
                    f"SemanticValidatorConfig(name={validator!r}, reference=(...))"
                ),
                "backend_requirement": "presentation_only",
            }
        )
    return issues


def _ci_summary(
    report_type: str,
    trust: dict[str, Any] | None,
    issues: list[dict[str, Any]],
) -> str:
    gate = (trust or {}).get("gate", {})
    if report_type != "trust_gate" or not gate:
        return ""
    passed = gate.get("passed")
    verdict = "PASSED" if passed is True else "FAILED" if passed is False else "NOT EVALUATED"
    lines = [
        f"freshdata trust gate {verdict}",
        f"  score: {gate.get('score')}",
        f"  required: {gate.get('threshold')}",
    ]
    if issues:
        lines.append("  violations:")
        for issue in issues:
            lines.append(
                f"    [{str(issue.get('severity', 'low')).upper()}] "
                f"{issue.get('id')}: {issue.get('finding')}"
            )
    return "\n".join(lines)


def to_pandas_safe(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value
    try:
        from .adapters.polars import to_pandas  # noqa: PLC0415

        return to_pandas(value)
    except Exception:  # noqa: BLE001 - display helpers must not make reports fail
        return None


def _action_hint(issues: list[str], role: str) -> str:
    joined = " ".join(issues).lower()
    if "outlier" in joined:
        return "outliers"
    if "missing" in joined:
        return "missing"
    if "convert" in joined:
        return "dtype"
    if role == "id":
        return "id"
    if role == "text":
        return "preserve"
    return "review"


def _fix_code(
    column: str,
    issues: list[str],
    role: str,
    config: CleanConfig,
    *,
    missing_pct: float,
    suggested_dtype: str | None,
) -> str:
    joined = " ".join(issues).lower()
    common = _common_args(config)
    if "outlier" in joined or (suggested_dtype or "").startswith("float"):
        return _clean_call([f'strategy="{config.strategy}"', 'outlier_method="iqr"', *common])
    if "missing" in joined and missing_pct >= config.missing_threshold_high * 100.0:
        return _clean_call(['strategy="aggressive"', *common])
    if role == "id":
        return _clean_call([f'strategy="{config.strategy}"', f"id_columns=({column!r},)", *common])
    if role == "target":
        return _clean_call([f'strategy="{config.strategy}"', f"target_column={column!r}", *common])
    if role == "text":
        return _clean_call(
            [f'strategy="{config.strategy}"', f"preserve_columns=({column!r},)", *common]
        )
    return _clean_call([f'strategy="{config.strategy}"', *common])


def _clean_call(args: list[str]) -> str:
    return f"fd.clean(df, {', '.join(args)}, return_report=True)"


def _common_args(config: CleanConfig) -> list[str]:
    args: list[str] = []
    if config.target_column is not None:
        args.append(f"target_column={config.target_column!r}")
    if config.id_columns:
        args.append(f"id_columns={tuple(config.id_columns)!r}")
    if config.preserve_columns:
        args.append(f"preserve_columns={tuple(config.preserve_columns)!r}")
    return args


def _highest_severity(issues: list[dict[str, Any]]) -> str:
    order = {"low": 1, "medium": 2, "high": 3}
    if not issues:
        return "none"
    return max((str(i["severity"]) for i in issues), key=lambda value: order.get(value, 0))


def _recommended_next_step(dataset_name: str, config: CleanConfig) -> str:
    return (
        f"Run fd.clean(df, strategy=\"{config.strategy}\", return_report=True) "
        f"and review high-risk preserved columns for {dataset_name}."
    )


def _slug(value: str) -> str:
    out = []
    for char in str(value).lower():
        out.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(out).split("_") if part)
