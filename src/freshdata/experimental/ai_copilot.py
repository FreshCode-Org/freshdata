"""FreshData AI Copilot — **experimental** explainable dataset analysis.

:func:`analyze_dataset` inspects a DataFrame and produces a
:class:`CopilotReport`: a plain-English summary, a ranked problem list, a
PII warning, context-policy violations, an ordered explainable cleaning
plan, and copy-ready freshdata code that implements the plan.

Design principles
-----------------
**Deterministic and offline by default.** The analysis is rule-based and
built entirely from freshdata's own primitives (profiling, PII detection,
context policies, value clustering, trust scoring). The same input always
produces the same report, no API key or network access is required, and
results are reproducible in CI.

**Privacy-first.** Raw cell values never enter the report's
``model_context`` (the payload an LLM provider *would* see). Samples are
masked with :class:`~freshdata.enterprise.MaskingRule` hashing plus
free-text PII scrubbing before inclusion, or omitted entirely with
``privacy="schema_only"``.

**Experimental.** This module lives under :mod:`freshdata.experimental`:
the report shape and rule vocabulary may evolve. The optional ``provider``
hook is a plain ``Callable[[str], str]`` so any LLM client can be plugged
in later; passing one emits a ``FutureWarning``. No built-in LLM
integration ships yet (TODO: first-party provider adapters).

Example
-------
>>> from freshdata.experimental.ai_copilot import analyze_dataset
>>> report = analyze_dataset(
...     df,
...     goal="Prepare this customer dataset for analytics and ML",
...     privacy="mask_pii_before_reasoning",
...     context_policy={
...         "email": "must_mask",
...         "age": "must_be_between_0_and_120",
...         "salary": "must_be_positive",
...         "city": "normalize_spelling",
...     },
... )
>>> print(report.summary)
>>> print(report.cleaning_plan)
>>> print(report.recommended_code)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..adapters.polars import to_pandas
from ..api import compile_context
from ..api import profile as _profile_frame
from ..api import validate as _validate_frame
from ..enterprise.cleaner import cluster_column
from ..enterprise.config import ClusterConfig, MaskingRule
from ..enterprise.metrics import TrustScore, compute_trust_score
from ..enterprise.privacy import PIIDetectionConfig, anonymize, detect_pii
from ..render.mixins import HtmlReprMixin

__all__ = [
    "CleaningPlan",
    "CopilotReport",
    "DetectedProblem",
    "PlanStep",
    "analyze_dataset",
]

_PRIVACY_MODES = ("mask_pii_before_reasoning", "schema_only")

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

#: ``context_policy`` rule vocabulary. ``must_be_between_<lo>_and_<hi>``
#: additionally accepts numeric bounds (int or float, negatives allowed).
SUPPORTED_POLICY_RULES = (
    "must_mask",
    "must_be_positive",
    "must_be_between_<lo>_and_<hi>",
    "must_be_unique",
    "normalize_spelling",
    "never_modify",
)

_BETWEEN_RE = re.compile(r"^must_be_between_(-?\d+(?:\.\d+)?)_and_(-?\d+(?:\.\d+)?)$")

#: Recognizable date layouts for the mixed-format check.
_DATE_FORMATS: tuple[tuple[str, str], ...] = (
    ("ISO (YYYY-MM-DD)", r"^\d{4}-\d{2}-\d{2}$"),
    ("slash D/M/Y", r"^\d{1,2}/\d{1,2}/\d{4}$"),
    ("slash YYYY/MM/DD", r"^\d{4}/\d{1,2}/\d{1,2}$"),
    ("dotted YYYY.MM.DD", r"^\d{4}\.\d{1,2}\.\d{1,2}$"),
    ("textual (Month D YYYY)", r"^[A-Za-z]{3,9}\.? \d{1,2},? \d{4}$"),
)


@dataclass(frozen=True)
class DetectedProblem:
    """One data-quality problem the copilot found, with severity and evidence."""

    kind: str
    severity: str
    detail: str
    column: str | None = None
    count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "detail": self.detail,
            "column": self.column,
            "count": self.count,
        }

    def __str__(self) -> str:
        where = f" [{self.column}]" if self.column else ""
        return f"({self.severity}){where} {self.detail}"


@dataclass(frozen=True)
class PlanStep:
    """One ordered step of the cleaning plan, with its rationale and the tool to use."""

    order: int
    action: str
    rationale: str
    tool: str
    columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "action": self.action,
            "rationale": self.rationale,
            "tool": self.tool,
            "columns": list(self.columns),
        }

    def __str__(self) -> str:
        cols = f" ({', '.join(self.columns)})" if self.columns else ""
        return (
            f"{self.order}. {self.action}{cols}"
            f"\n     why:  {self.rationale}"
            f"\n     tool: {self.tool}"
        )


@dataclass(frozen=True)
class CleaningPlan:
    """The ordered, explainable cleaning plan. ``print()`` it for a readable view."""

    steps: tuple[PlanStep, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}

    def __str__(self) -> str:
        if not self.steps:
            return "cleaning plan: nothing to do — the dataset already looks healthy."
        lines = ["cleaning plan (ordered, explainable):"]
        lines += [str(s) for s in self.steps]
        return "\n".join(lines)


@dataclass(frozen=True)
class CopilotReport(HtmlReprMixin):
    """Everything one :func:`analyze_dataset` run produced.

    ``summary``, ``cleaning_plan``, and ``recommended_code`` are all directly
    printable. ``model_context`` is the *only* payload an LLM provider ever
    sees — inspect it to verify no raw PII leaves your machine.
    """

    goal: str
    summary: str
    problems: tuple[DetectedProblem, ...]
    pii_warning: str | None
    policy_violations: tuple[Any, ...]
    cleaning_plan: CleaningPlan
    recommended_code: str
    trust: TrustScore
    model_context: dict[str, Any]
    audit: dict[str, Any]
    #: Free-form narrative from the optional ``provider`` hook (``None`` on the
    #: default deterministic path or when the provider call failed).
    narrative: str | None = None

    _render_kind = "copilot"

    @property
    def attention(self) -> tuple:
        """The ranked attention queue (privacy/policy first), as ``AttentionItem``\\s.

        Terminal, notebook, and programmatic callers all read the same order.
        """
        from ..render import normalize  # noqa: PLC0415

        return normalize.normalize(self).attention

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "summary": self.summary,
            "problems": [p.to_dict() for p in self.problems],
            "pii_warning": self.pii_warning,
            "policy_violations": [
                v.to_dict() if hasattr(v, "to_dict") else str(v) for v in self.policy_violations
            ],
            "cleaning_plan": self.cleaning_plan.to_dict(),
            "recommended_code": self.recommended_code,
            "trust_score": self.trust.to_dict(),
            "model_context": self.model_context,
            "narrative": self.narrative,
            "audit": self.audit,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __str__(self) -> str:
        return self.summary


@dataclass
class _PolicyIntent:
    """The parsed ``context_policy`` dict: which columns to mask/cluster/protect
    plus the natural-language sentences handed to :func:`freshdata.compile_context`."""

    mask_columns: list[str] = field(default_factory=list)
    cluster_columns: list[str] = field(default_factory=list)
    protected_columns: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)


def _parse_context_policy(context_policy: Mapping[str, Any] | None) -> _PolicyIntent:
    intent = _PolicyIntent()
    if not context_policy:
        return intent
    for column, raw in context_policy.items():
        rules = [raw] if isinstance(raw, str) else list(raw)
        for rule in rules:
            if not isinstance(rule, str):
                raise TypeError(
                    f"context_policy[{column!r}] entries must be strings, "
                    f"got {type(rule).__name__}"
                )
            if rule == "must_mask":
                intent.mask_columns.append(str(column))
            elif rule == "normalize_spelling":
                intent.cluster_columns.append(str(column))
            elif rule == "never_modify":
                intent.protected_columns.append(str(column))
                intent.sentences.append(f"never modify {column} values")
            elif rule == "must_be_positive":
                intent.sentences.append(f"{column} must be at least 0")
            elif rule == "must_be_unique":
                intent.sentences.append(f"{column} is unique")
            elif match := _BETWEEN_RE.match(rule):
                lo, hi = match.group(1), match.group(2)
                intent.sentences.append(f"{column} must be between {lo} and {hi}")
            else:
                supported = ", ".join(SUPPORTED_POLICY_RULES)
                raise ValueError(
                    f"unsupported context_policy rule {rule!r} for column {column!r}; "
                    f"supported rules: {supported}"
                )
    return intent


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if pd.isna(value):
        return None
    return str(value)


def _detect_mixed_date_formats(series: pd.Series) -> list[str]:
    """Return the distinct date layouts found in *series* (empty if not date-like)."""
    values = series.dropna().astype(str).head(200)
    if values.empty:
        return []
    hits: dict[str, int] = {}
    matched = 0
    for value in values:
        for label, pattern in _DATE_FORMATS:
            if re.match(pattern, value.strip()):
                hits[label] = hits.get(label, 0) + 1
                matched += 1
                break
    if len(hits) >= 2 and matched / len(values) >= 0.6:
        return sorted(hits, key=lambda k: (-hits[k], k))
    return []


def _category_noise(
    frame: pd.DataFrame,
    column: str,
    *,
    forced: bool,
) -> list[tuple[str, list[str]]]:
    """Return ``(canonical, variants)`` groups of near-duplicate spellings in *column*."""
    series = frame[column]
    if series.dtype != object and not isinstance(series.dtype, pd.StringDtype):
        return []
    n_unique = series.nunique(dropna=True)
    if n_unique < 2:
        return []
    # Unless the user forced clustering via ``normalize_spelling``, skip
    # near-unique columns (names, IDs, free text) where merging is unsafe.
    if not forced and (n_unique > 200 or n_unique > 0.5 * max(len(series), 1)):
        return []
    try:
        result = cluster_column(frame, column, config=ClusterConfig(), method="fingerprint")
    except ValueError:
        return []
    groups = []
    for cluster in result.clusters:
        variants = [v for v in cluster.variants if v != cluster.canonical]
        if variants:
            groups.append((cluster.canonical, sorted(variants)))
    return sorted(groups)


def _build_prompt(goal: str, model_context: dict[str, Any]) -> str:
    """The exact prompt a provider hook receives — built only from masked context."""
    return (
        "You are a data-quality assistant. Using ONLY the masked dataset "
        "context below (schema, aggregate statistics, and sample rows whose "
        "string values are hash-masked; numeric values pass through as-is), "
        "explain the main data-quality risks and how "
        "the proposed freshdata cleaning plan addresses them.\n\n"
        f"User goal: {goal}\n\n"
        f"Masked dataset context (JSON):\n{json.dumps(model_context, indent=2, default=str)}"
    )


def _is_stringlike(dtype: object) -> bool:
    return pd.api.types.is_object_dtype(dtype) or isinstance(
        dtype, (pd.StringDtype, pd.CategoricalDtype)
    )


def _sample_mask_columns(
    frame: pd.DataFrame, mask_columns: Sequence[str], allow_unmasked: Sequence[str]
) -> list[str]:
    """Columns to hash-mask in sample rows: every declared/detected PII column
    *and* every string-like column — regex PII detection cannot see names,
    addresses, or free text, so string-like columns are unsafe to send raw.
    ``allow_unmasked`` exempts specific string-like columns but never a
    declared or detected PII column. Numeric columns pass through as-is.
    """
    declared = {c for c in mask_columns if c in frame.columns}
    stringlike = {c for c in frame.columns if _is_stringlike(frame[c].dtype)}
    return sorted(declared | (stringlike - set(allow_unmasked)), key=str)


def _mask_sample(
    frame: pd.DataFrame, columns: Sequence[str], sample_rows: int
) -> list[dict[str, Any]]:
    sample = frame.head(sample_rows)
    rules = tuple(
        MaskingRule(name=f"copilot_mask_{c}", columns=(str(c),), strategy="hash") for c in columns
    )
    masked = anonymize(
        sample, rules=rules, detection_config=PIIDetectionConfig(), return_report=False
    )
    return [
        {str(k): _json_scalar(v) for k, v in record.items()}
        for record in masked.to_dict(orient="records")
    ]


def _problem_context_dict(problem: DetectedProblem) -> dict[str, Any]:
    """Serialize *problem* for ``model_context`` without raw cell values.

    ``category_noise`` details embed example spellings straight from the
    data; the local ``report.problems`` keeps those previews, but the model
    context must stay value-free in every privacy mode.
    """
    out = problem.to_dict()
    if problem.kind == "category_noise":
        out["detail"] = (
            f"{problem.count} groups of near-duplicate spellings detected "
            "(value preview withheld from model context)"
        )
    return out


def _generate_code(
    *,
    mask_columns: list[str],
    policy_sentences: list[str],
    cluster_columns: list[str],
    source_hint: str,
) -> str:
    """Render the copy-ready freshdata pipeline for this dataset."""
    lines = [
        "import pandas as pd",
        "import freshdata as fd",
        "from freshdata.enterprise import MaskingRule, compute_trust_score, merge_clusters",
        "",
        f"df = pd.read_csv({source_hint!r})",
        'print("trust before:", compute_trust_score(df).overall)',
        "",
    ]
    step = 1
    if mask_columns:
        rule_lines = ",\n".join(
            "    MaskingRule("
            f"name={f'mask_{c}'!r}, columns={(str(c),)!r}, strategy=\"hash\")"
            for c in mask_columns
        )
        lines += [
            f"# {step}) Mask PII before anything leaves your machine",
            f"rules = (\n{rule_lines},\n)",
            "df, privacy_report = fd.anonymize(df, rules=rules)",
            "print(privacy_report.summary())",
            "",
        ]
        step += 1
    if policy_sentences:
        text = " ".join(s.rstrip(".") + "." for s in policy_sentences)
        has_range_rule = any("between" in s or "at least" in s for s in policy_sentences)
        clean_call = "cleaned, report = fd.clean(df, policy=policy, return_report=True"
        clean_call += ', outliers="clip")' if has_range_rule else ")"
        lines += [
            f"# {step}) Encode domain rules as a reviewable, deterministic context policy",
            f"policy = fd.compile_context({text!r}, df=df)",
            "print(policy.summary())",
            "",
            f"# {step + 1}) Clean with the policy enforced; keep the full audit trail",
            clean_call,
        ]
        step += 2
    else:
        lines += [
            f"# {step}) Clean with safe defaults; keep the full audit trail",
            "cleaned, report = fd.clean(df, return_report=True)",
        ]
        step += 1
    lines += ["print(report.summary())", ""]
    if cluster_columns:
        cols = [str(c) for c in cluster_columns]
        lines += [
            f"# {step}) Merge near-duplicate category spellings",
            f"cleaned, cluster_results = merge_clusters(cleaned, columns={cols!r})",
            "",
        ]
        step += 1
    lines += [
        f"# {step}) Verify: re-score trust and inspect every action taken",
        'print("trust after:", compute_trust_score(cleaned).overall)',
        "for action in report.actions:",
        "    print(action.column, action.description, action.rationale)",
    ]
    return "\n".join(lines)


def _build_plan(
    *,
    mask_columns: list[str],
    pii_columns: list[str],
    duplicate_rows: int,
    policy_sentences: list[str],
    n_violations: int,
    noisy_columns: list[str],
    missing_columns: list[str],
) -> CleaningPlan:
    steps: list[PlanStep] = []

    def add(action: str, rationale: str, tool: str, columns: Sequence[str] = ()) -> None:
        steps.append(
            PlanStep(
                order=len(steps) + 1,
                action=action,
                rationale=rationale,
                tool=tool,
                columns=tuple(columns),
            )
        )

    to_mask = sorted(dict.fromkeys([*mask_columns, *pii_columns]))
    if to_mask:
        add(
            "Mask PII columns",
            "these columns hold personal data; masking first means nothing downstream "
            "(reports, models, logs) ever sees raw identifiers",
            "fd.anonymize(df, rules=(MaskingRule(...),))",
            to_mask,
        )
    if duplicate_rows:
        add(
            f"Remove {duplicate_rows} duplicate row(s)",
            "exact duplicates bias aggregates and leak between ML train/test splits",
            "fd.clean(df)  # duplicate handling is on by default",
        )
    if policy_sentences:
        rationale = "encode business rules once, then enforce them on every run"
        if n_violations:
            rationale = (
                f"{n_violations} value(s) currently violate your rules; the policy "
                "makes the fix explicit and repeatable"
            )
        add(
            "Compile and enforce the context policy",
            rationale,
            'fd.clean(df, policy=fd.compile_context("...", df=df))',
        )
    if noisy_columns:
        add(
            "Normalize near-duplicate category spellings",
            "variant spellings fragment group-bys and one-hot encodings",
            "freshdata.enterprise.merge_clusters(df, columns=[...])",
            sorted(dict.fromkeys(noisy_columns)),
        )
    if missing_columns:
        add(
            "Impute or flag missing values",
            "the decision engine picks a per-column strategy and records its rationale",
            "fd.clean(df, return_report=True)",
            sorted(missing_columns),
        )
    add(
        "Verify and keep the audit trail",
        "re-score trust, review every action, and export the report for reviewers",
        "compute_trust_score(cleaned); report.summary(); report.to_json()",
    )
    return CleaningPlan(steps=tuple(steps))


def _build_summary(
    *,
    goal: str,
    frame: pd.DataFrame,
    trust: TrustScore,
    problems: Sequence[DetectedProblem],
    pii_warning: str | None,
    engine: str,
) -> str:
    counts = {"high": 0, "medium": 0, "low": 0}
    for p in problems:
        counts[p.severity] = counts.get(p.severity, 0) + 1
    lines = [
        "FreshData AI Copilot report (experimental)",
        f"  engine:  {engine}",
        f"  goal:    {goal}",
        f"  shape:   {len(frame)} rows x {frame.shape[1]} columns",
        f"  trust:   {trust.overall:.1f}/100 (grade {trust.grade})",
        (
            f"  problems: {len(problems)} found — "
            f"{counts['high']} high, {counts['medium']} medium, {counts['low']} low"
        ),
    ]
    for p in problems[:8]:
        lines.append(f"    - {p}")
    if len(problems) > 8:
        lines.append(f"    … and {len(problems) - 8} more (see report.problems)")
    if pii_warning:
        lines.append(f"  privacy: {pii_warning}")
    lines.append("  next:    print(report.cleaning_plan), then run report.recommended_code")
    return "\n".join(lines)


@dataclass
class _Findings:
    """Everything the deterministic analysis pass collected."""

    problems: list[DetectedProblem] = field(default_factory=list)
    date_layouts: dict[str, list[str]] = field(default_factory=dict)
    pii_columns: list[str] = field(default_factory=list)
    pii_suppressed: list[str] = field(default_factory=list)
    pii_warning: str | None = None
    violations: tuple[Any, ...] = ()
    compiled_policy_summary: str | None = None
    duplicate_rows: int = 0
    missing_columns: list[str] = field(default_factory=list)
    noisy_columns: list[str] = field(default_factory=list)


def _scan_pii(scan: Any, out: _Findings) -> None:
    pii_by_column = scan.by_column()
    # Dotted/slashed dates match the generic phone pattern; a column whose
    # only PII hits are PHONE but which parses as dates is a false positive.
    out.pii_suppressed = sorted(
        col
        for col, entities in pii_by_column.items()
        if col in out.date_layouts and {e.entity_type for e in entities} == {"PHONE"}
    )
    for col in out.pii_suppressed:
        del pii_by_column[col]
    out.pii_columns = sorted(pii_by_column)
    if not out.pii_columns:
        return
    parts = []
    for col in out.pii_columns:
        types = sorted({e.entity_type for e in pii_by_column[col]})
        parts.append(f"{col} ({'/'.join(types)})")
        out.problems.append(
            DetectedProblem(
                kind="pii",
                severity="high",
                column=col,
                count=len(pii_by_column[col]),
                detail=f"contains {'/'.join(types)} personal data — mask before sharing",
            )
        )
    out.pii_warning = (
        f"PII detected in {len(out.pii_columns)} column(s): {', '.join(parts)}. "
        "Mask before sharing, training, or LLM reasoning — this report's own "
        "model context is already masked."
    )


def _check_policy(frame: pd.DataFrame, intent: _PolicyIntent, out: _Findings) -> None:
    if not intent.sentences:
        return
    text = " ".join(s.rstrip(".") + "." for s in intent.sentences)
    policy = compile_context(text, df=frame)
    out.compiled_policy_summary = policy.summary()
    out.violations = tuple(_validate_frame(frame, policy=policy))
    for finding in out.violations:
        severity = "high" if getattr(finding, "severity", "") == "error" else "medium"
        out.problems.append(
            DetectedProblem(
                kind="policy_violation",
                severity=severity,
                column=getattr(finding, "column", None),
                count=(getattr(finding, "extra", None) or {}).get("n_violations"),
                detail=str(getattr(finding, "message", finding)),
            )
        )


def _scan_structure(frame: pd.DataFrame, prof: Any, intent: _PolicyIntent, out: _Findings) -> None:
    out.duplicate_rows = prof.duplicate_rows or 0
    if out.duplicate_rows:
        out.problems.append(
            DetectedProblem(
                kind="duplicate_rows",
                severity="medium",
                count=out.duplicate_rows,
                detail=f"{out.duplicate_rows} exact duplicate row(s)",
            )
        )
    for col in prof.columns:
        if col.missing:
            out.missing_columns.append(col.name)
            out.problems.append(
                DetectedProblem(
                    kind="missing_values",
                    severity="medium" if col.missing_pct >= 20 else "low",
                    column=col.name,
                    count=col.missing,
                    detail=f"{col.missing} missing value(s) ({col.missing_pct:.1f}%)",
                )
            )
    for col, layouts in out.date_layouts.items():
        out.problems.append(
            DetectedProblem(
                kind="mixed_date_formats",
                severity="medium",
                column=col,
                count=len(layouts),
                detail=f"mixed date layouts: {', '.join(layouts)}",
            )
        )
    skip = set(out.pii_columns) | set(intent.mask_columns)
    candidates = list(dict.fromkeys(intent.cluster_columns)) + [
        str(c)
        for c in frame.columns
        if str(c) not in skip and str(c) not in intent.cluster_columns
    ]
    for col in candidates:
        if col not in frame.columns or col in skip:
            continue
        groups = _category_noise(frame, col, forced=col in intent.cluster_columns)
        if not groups:
            continue
        out.noisy_columns.append(col)
        preview = "; ".join(
            f"{canonical!r} ~ {', '.join(repr(v) for v in variants[:3])}"
            for canonical, variants in groups[:3]
        )
        out.problems.append(
            DetectedProblem(
                kind="category_noise",
                severity="medium",
                column=col,
                count=len(groups),
                detail=f"near-duplicate spellings: {preview}",
            )
        )


def _collect_findings(
    frame: pd.DataFrame, prof: Any, scan: Any, intent: _PolicyIntent
) -> _Findings:
    out = _Findings()
    for col in frame.columns:
        if frame[col].dtype == object:
            layouts = _detect_mixed_date_formats(frame[col])
            if layouts:
                out.date_layouts[str(col)] = layouts
    _scan_pii(scan, out)
    _check_policy(frame, intent, out)
    _scan_structure(frame, prof, intent, out)
    out.problems.sort(key=lambda p: (_SEVERITY_RANK.get(p.severity, 9), p.kind, p.column or ""))
    return out


def analyze_dataset(
    df: Any,
    *,
    goal: str = "Prepare this dataset for analytics and ML",
    privacy: str = "mask_pii_before_reasoning",
    context_policy: Mapping[str, Any] | None = None,
    provider: Callable[[str], str] | None = None,
    sample_rows: int = 5,
    source_hint: str = "your_data.csv",
    allow_unmasked_columns: Sequence[str] = (),
    sensitive_columns: Sequence[str] = (),
) -> CopilotReport:
    """Analyze *df* and return an explainable, privacy-safe :class:`CopilotReport`.

    Deterministic and fully offline by default: the analysis is rule-based
    (freshdata profiling, PII detection, context-policy validation, value
    clustering, trust scoring) and needs no API key.

    Parameters
    ----------
    df:
        A pandas (or polars) DataFrame. Never modified.
    goal:
        What you want the dataset ready for; echoed into the report and the
        provider prompt.
    privacy:
        ``"mask_pii_before_reasoning"`` (default) includes ``sample_rows``
        sample rows in ``report.model_context`` with every declared/detected
        PII column *and* every string-like column hash-masked (regex PII
        detection cannot see names, addresses, or free text, so string
        values are never sent raw). Numeric values pass through as-is —
        numeric quasi-identifiers are the residual risk; drop such columns
        first or use ``"schema_only"``, which includes no sample rows at
        all.
    context_policy:
        Optional ``{column: rule}`` mapping (rule may also be a list of
        rules). Supported rules: ``must_mask``,
        ``must_be_between_<lo>_and_<hi>``, ``must_be_positive``,
        ``must_be_unique``, ``normalize_spelling``, ``never_modify``.
        Range/positive/unique/protected rules are compiled through
        :func:`freshdata.compile_context` and validated against *df*.
    provider:
        **Experimental.** Optional ``Callable[[str], str]`` LLM hook. It
        receives one prompt built *only* from the masked ``model_context``
        and its return value lands in ``report.narrative``. Provider
        failures are recorded in ``report.audit`` and never break the
        deterministic report.
    sample_rows:
        How many masked sample rows to include in ``model_context``.
    source_hint:
        Filename used in the generated ``recommended_code``.
    allow_unmasked_columns:
        Explicit opt-out: string-like columns listed here are sent unmasked
        in the sample rows. Declared (``must_mask``) and regex-detected PII
        columns are always masked regardless. Unknown column names raise
        ``ValueError``.
    """
    if privacy not in _PRIVACY_MODES:
        raise ValueError(f"privacy must be one of {_PRIVACY_MODES}, got {privacy!r}")
    frame = to_pandas(df)
    unknown = [str(c) for c in allow_unmasked_columns if c not in frame.columns]
    if unknown:
        raise ValueError(f"allow_unmasked_columns contains unknown column(s): {unknown}")
    intent = _parse_context_policy(context_policy)

    prof = _profile_frame(frame)
    trust = compute_trust_score(frame)
    scan = detect_pii(frame)

    found = _collect_findings(frame, prof, scan, intent)
    problems = found.problems
    pii_columns = found.pii_columns
    pii_warning = found.pii_warning

    # --- plan + code -------------------------------------------------------------
    plan = _build_plan(
        mask_columns=intent.mask_columns,
        pii_columns=pii_columns,
        duplicate_rows=found.duplicate_rows,
        policy_sentences=intent.sentences,
        n_violations=sum(1 for p in problems if p.kind == "policy_violation"),
        noisy_columns=found.noisy_columns,
        missing_columns=found.missing_columns,
    )
    # Declared-sensitive columns always join the mask set: pattern-based PII
    # detection cannot recognise every sensitive token (an internal case ID,
    # a synthetic SSN), so the caller's declaration is authoritative.
    declared_sensitive = [str(c) for c in sensitive_columns if str(c) in df.columns]
    mask_for_code = sorted(
        dict.fromkeys([*intent.mask_columns, *pii_columns, *declared_sensitive])
    )
    recommended_code = _generate_code(
        mask_columns=mask_for_code,
        policy_sentences=intent.sentences,
        cluster_columns=sorted(dict.fromkeys([*intent.cluster_columns, *found.noisy_columns])),
        source_hint=source_hint,
    )

    # --- model context (the ONLY payload a provider ever sees) ----------------
    model_context: dict[str, Any] = {
        "privacy": privacy,
        "n_rows": prof.n_rows,
        "n_cols": prof.n_cols,
        "duplicate_rows": found.duplicate_rows,
        "trust_score": trust.overall,
        "schema": [
            {
                "name": c.name,
                "dtype": c.dtype,
                "missing_pct": round(c.missing_pct, 2),
                "unique": c.unique,
                "pii": c.name in pii_columns,
            }
            for c in prof.columns
        ],
        "problems": [_problem_context_dict(p) for p in problems],
    }
    sample_mask: list[str] = []
    if privacy == "mask_pii_before_reasoning" and sample_rows > 0:
        sample_mask = _sample_mask_columns(frame, mask_for_code, allow_unmasked_columns)
        model_context["sample_rows_masked"] = _mask_sample(frame, sample_mask, sample_rows)

    # --- optional provider hook (experimental) ----------------------------------
    engine = "deterministic-local"
    narrative: str | None = None
    provider_error: str | None = None
    if provider is not None:
        warnings.warn(
            "ai_copilot provider hooks are experimental; the prompt contract "
            "may change in future releases",
            FutureWarning,
            stacklevel=2,
        )
        try:
            narrative = str(provider(_build_prompt(goal, model_context)))
            engine = f"provider:{getattr(provider, '__name__', type(provider).__name__)}"
        except Exception as exc:  # noqa: BLE001 - provider failures must not break the report
            provider_error = f"{type(exc).__name__}: {exc}"

    summary = _build_summary(
        goal=goal,
        frame=frame,
        trust=trust,
        problems=problems,
        pii_warning=pii_warning,
        engine=engine,
    )

    context_sha = hashlib.sha256(
        json.dumps(model_context, sort_keys=True, default=str).encode()
    ).hexdigest()
    audit: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": engine,
        "privacy_mode": privacy,
        "pii_entities_found": len(scan.entities),
        "pii_columns": pii_columns,
        "pii_suppressed_date_like": found.pii_suppressed,
        "masked_columns": mask_for_code,
        "sample_masked_columns": sample_mask,
        "allow_unmasked_columns": sorted(str(c) for c in allow_unmasked_columns),
        "policy_sentences": list(intent.sentences),
        "compiled_policy": found.compiled_policy_summary,
        "policy_violations": len(found.violations),
        "problems_found": len(problems),
        "model_context_sha256": context_sha,
    }
    if provider_error:
        audit["provider_error"] = provider_error

    return CopilotReport(
        goal=goal,
        summary=summary,
        problems=tuple(problems),
        pii_warning=pii_warning,
        policy_violations=found.violations,
        cleaning_plan=plan,
        recommended_code=recommended_code,
        trust=trust,
        model_context=model_context,
        audit=audit,
        narrative=narrative,
    )
