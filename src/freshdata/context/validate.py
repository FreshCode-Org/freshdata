"""Non-mutating policy validation: check a frame against a context, report findings.

Powers :func:`freshdata.validate`. The frame is never modified; every check
projects into the shared :class:`~freshdata.findings.QualityFinding` shape so
the quality-ops exporters downstream speak one language.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

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


#: Case-insensitive spellings accepted for a boolean column's allowed values.
_BOOL_WORDS = {
    "true": True,
    "t": True,
    "yes": True,
    "y": True,
    "1": True,
    "false": False,
    "f": False,
    "no": False,
    "n": False,
    "0": False,
}


def _as_number(value: object) -> int | float | None:
    """*value* as a finite Python number, or ``None`` when it is not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number: int | float = value
    else:
        text = str(value).strip()
        try:
            number = int(text)
        except ValueError:
            try:
                number = float(text)
            except ValueError:
                return None
    return number if math.isfinite(number) else None


def _as_bool(value: object) -> bool | None:
    """*value* as a bool (true/false/yes/no/1/0/t/f/y/n, any case), else ``None``."""
    if isinstance(value, bool):
        return value
    return _BOOL_WORDS.get(str(value).strip().lower())


def _dedupe(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for v in values:
        if not any(type(v) is type(o) and v == o for o in out):
            out.append(v)
    return out


def _typed_allowed(series: pd.Series, raw: list[Any]) -> tuple[list[Any], list[Any], pd.Series]:
    """Return ``(value_set, comparable, observed)`` for the column's dtype.

    * boolean columns (``bool`` / nullable ``boolean``): allowed entries map
      case-insensitively via :data:`_BOOL_WORDS`;
    * other numeric columns: allowed entries are parsed as numbers and compared
      with exact numeric equality (so ``"1"`` matches ``1.0``);
    * everything else: today's string comparison, unchanged.

    ``value_set`` keeps entries that do not convert as their original string so
    nothing declared is dropped from the exported set; ``comparable`` holds only
    the converted entries, since an unconvertible one can never match a value of
    that dtype. Missing values are excluded from ``observed`` in every case.
    """
    from pandas.api.types import is_bool_dtype, is_numeric_dtype  # noqa: PLC0415

    observed = series.dropna()
    if is_bool_dtype(series.dtype) or is_numeric_dtype(series.dtype):
        convert = _as_bool if is_bool_dtype(series.dtype) else _as_number
        converted = [convert(v) for v in raw]
        comparable = _dedupe([v for v in converted if v is not None])
        value_set = _dedupe([str(v) if t is None else t for v, t in zip(raw, converted)])
        if is_bool_dtype(series.dtype):
            observed = observed.astype(bool)
        return value_set, comparable, observed
    values = [str(v) for v in raw]
    return values, values, observed.astype(str)


def _check_allowed_values(series: pd.Series, c: ColumnConstraint) -> QualityFinding | None:
    raw = list(c.params.get("values", ()))
    if not raw:
        return None
    values, comparable, observed = _typed_allowed(series, raw)
    bad = observed[~observed.isin(comparable)]
    if bad.empty:
        return None
    examples = sorted(set(bad.tolist()))[:5]
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

    from .._numeric import safe_to_numeric  # noqa: PLC0415

    numeric = safe_to_numeric(series, errors="coerce")
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

    _run_plugin_validators(df, policy, cfg, findings)
    return findings


def _run_plugin_validators(
    df: pd.DataFrame, policy: ContextPolicy, cfg: CleanConfig, findings: FindingList
) -> None:
    """Append read-only findings from any active plugin validators.

    Plugin validators receive the frame, the compiled policy, and the config;
    they may only *append findings* (each isolated so a bug cannot break
    validation) — they never mutate the frame or the existing findings.
    """
    from ..plugins import active_validators  # noqa: PLC0415 - avoid import cycle

    for validator in active_validators():
        for finding in validator.validate(df, policy, cfg):
            findings.append(finding)
