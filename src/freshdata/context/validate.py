"""Non-mutating policy validation: check a frame against a context, report findings.

Powers :func:`freshdata.validate`. The frame is never modified; every check
projects into the shared :class:`~freshdata.findings.QualityFinding` shape so
the quality-ops exporters downstream speak one language.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..findings import FindingList, QualityFinding
from .compiler import compile_context, effective_columns, resolve_policy
from .types import ColumnConstraint, ContextPolicy

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from ..config import CleanConfig

_STEP = "context"


def _policy_for(df: pd.DataFrame, cfg: CleanConfig) -> ContextPolicy:
    """Compile ``cfg.context`` or resolve a supplied ``cfg.policy`` for *df*."""
    if cfg.context is not None:
        return compile_context(cfg.context, df=df, config=cfg, strict=cfg.strict)
    policy = cfg.policy
    if not isinstance(policy, ContextPolicy):
        raise TypeError(
            f"policy must be a freshdata.ContextPolicy, got {type(policy).__name__}"
        )
    schema = effective_columns(df, None, cfg)
    assert schema is not None
    return resolve_policy(policy, schema)


def _series_for(df: pd.DataFrame, cfg: CleanConfig, column: str) -> pd.Series | None:
    """The raw series behind an effective (post-normalization) column name.

    ``fd.validate`` never renames anything, so the policy's resolved names are
    translated back to the frame's actual labels positionally.
    """
    schema = effective_columns(df, None, cfg)
    assert schema is not None
    for actual, effective in zip(df.columns, schema):
        if effective == column:
            return df[actual]
    return None


def _check_unique(series: pd.Series, c: ColumnConstraint) -> QualityFinding | None:
    dupes = series[series.notna() & series.duplicated(keep=False)]
    if dupes.empty:
        return None
    return QualityFinding.create(
        severity="error",
        step=_STEP,
        rule_name="context.unique",
        column=c.column,
        message=f"{c.column!r} declared unique but has {len(dupes)} duplicated value(s)",
        row_selector=f"rows: {sorted(map(str, dupes.index[:10]))}",
        expected_condition="all values unique",
        extra={"constraint_id": c.id, "n_violations": int(len(dupes))},
    )


def _check_allowed_values(series: pd.Series, c: ColumnConstraint) -> QualityFinding | None:
    values = [str(v) for v in c.params.get("values", ())]
    if not values:
        return None
    observed = series.dropna().astype(str)
    bad = observed[~observed.isin(values)]
    if bad.empty:
        return None
    examples = sorted(set(bad))[:5]
    return QualityFinding.create(
        severity="error",
        step=_STEP,
        rule_name="context.allowed_values",
        column=c.column,
        message=f"{c.column!r} has {len(bad)} value(s) outside the allowed set",
        observed_value=examples,
        expected_condition=f"one of {values}",
        extra={"constraint_id": c.id, "value_set": values, "n_violations": int(len(bad))},
    )


def _check_range(series: pd.Series, c: ColumnConstraint) -> QualityFinding | None:
    import pandas as pd  # noqa: PLC0415 - keep the context package import-light

    numeric = pd.to_numeric(series, errors="coerce")
    lo, hi = c.params.get("lo"), c.params.get("hi")
    mask = pd.Series(False, index=series.index)
    if lo is not None:
        mask |= numeric < lo
    if hi is not None:
        mask |= numeric > hi
    mask &= numeric.notna()
    n_bad = int(mask.sum())
    if not n_bad:
        return None
    bounds = f"[{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}]"
    return QualityFinding.create(
        severity="error",
        step=_STEP,
        rule_name="context.range",
        column=c.column,
        message=f"{c.column!r} has {n_bad} value(s) outside {bounds}",
        observed_value=sorted(numeric[mask].head(5).tolist()),
        expected_condition=f"value in {bounds}",
        extra={
            "constraint_id": c.id,
            "min_value": lo,
            "max_value": hi,
            "n_violations": n_bad,
        },
    )


def validate_frame(df: pd.DataFrame, cfg: CleanConfig) -> FindingList:
    """Check *df* against the compiled policy without mutating anything.

    Returns a :class:`~freshdata.findings.FindingList` covering unresolved
    references, compile issues, protected columns, and (where checkable in
    Phase 1) unique / allowed-values / range violations.
    """
    policy = _policy_for(df, cfg)
    findings = FindingList()

    for u in policy.unresolved:
        findings.append(
            QualityFinding.create(
                severity="warning",
                step=_STEP,
                rule_name="context.unresolved_reference",
                message=f"could not resolve column reference {u.ref!r}: {u.reason}",
                expected_condition="reference resolves to exactly one column",
                extra={
                    "ref": u.ref,
                    "sentence": u.sentence,
                    "candidates": [[c, s] for c, s in u.candidates],
                },
            )
        )
    for issue in policy.issues:
        findings.append(
            QualityFinding.create(
                severity=issue.severity,
                step=_STEP,
                rule_name=f"context.{issue.kind}",
                column=issue.columns[0] if issue.columns else None,
                message=issue.message,
                extra={"sentences": list(issue.sentences)},
            )
        )

    for c in policy.constraints:
        if c.rule == "protected" and c.column is not None:
            findings.append(
                QualityFinding.create(
                    severity="info",
                    step=_STEP,
                    rule_name="context.protected",
                    column=c.column,
                    message=f"{c.column!r} is protected by policy {c.id} (never modified)",
                    expected_condition="column left byte-identical by any clean",
                    extra={"constraint_id": c.id, "enforcement": c.enforcement},
                )
            )
        if c.column is None:
            continue
        series = _series_for(df, cfg, c.column)
        if series is None:
            continue
        finding: QualityFinding | None = None
        if c.rule == "unique":
            finding = _check_unique(series, c)
        elif c.rule == "allowed_values":
            finding = _check_allowed_values(series, c)
        elif c.rule == "range":
            finding = _check_range(series, c)
        if finding is not None:
            findings.append(finding)
    return findings
