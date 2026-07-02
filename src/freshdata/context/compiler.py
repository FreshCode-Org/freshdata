"""Compile parsed intents into a :class:`ContextPolicy` and lower it into config.

The compiler is the only place where parsing, column resolution, conflict
detection, and lowering meet. It never invents columns, never guesses between
ambiguous candidates, and in strict mode refuses to produce a policy with
unresolved references, unparsed sentences, or protection conflicts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .parser import parse_context
from .resolve import DEFAULT_THRESHOLD, Resolution, resolve_reference
from .types import (
    MUTATING_ACTIONS,
    POLICY_VERSION,
    ColumnConstraint,
    ContextPolicy,
    IntentCandidate,
    PolicyError,
    PolicyIssue,
    UnresolvedRef,
    sha256_text,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from ..config import CleanConfig

#: intent -> (rule, action, enforcement)
_INTENT_TABLE: dict[str, tuple[str, str, str]] = {
    "unique": ("unique", "validate_only", "soft"),
    "valid_format": ("valid_format", "repair_or_flag", "soft"),
    "locale_format": ("locale_format", "normalize_or_flag", "soft"),
    "protected": ("protected", "never_modify", "hard"),
    "impute_if": ("impute_missing", "impute_if_confident_else_preserve", "soft"),
    "allowed_values": ("allowed_values", "validate_only", "soft"),
    "range": ("range", "validate_only", "soft"),
    "dedup_key": ("dedup_key", "custom", "soft"),
    "drop_if": ("custom", "custom", "soft"),
    "rename": ("custom", "custom", "soft"),
    "map": ("custom", "custom", "soft"),
}

#: Intents folded into ``params["kind"]`` when the rule collapses to "custom".
_CUSTOM_KINDS = frozenset({"drop_if", "rename", "map"})


def effective_columns(
    df: pd.DataFrame | None,
    columns: Sequence[str] | None,
    config: CleanConfig | None,
) -> list[str] | None:
    """The schema the resolver should see: post-``column_names`` labels.

    With ``column_names=True`` (the default) the pipeline snake_cases labels
    before any policy could act on them, so references are resolved against the
    renamed schema — exactly the names ``preserve_columns``/``id_columns`` use.
    """
    if columns is not None:
        raw: list[object] = list(columns)
    elif df is not None:
        raw = list(df.columns)
    else:
        return None
    normalize = config.column_names if config is not None else True
    if normalize:
        from ..steps.columns import normalized_column_labels  # noqa: PLC0415

        return [str(c) for c in normalized_column_labels(raw)]
    return [str(c) for c in raw]


def _constraint_params(candidate: IntentCandidate) -> dict[str, Any]:
    params = dict(candidate.params)
    if candidate.intent in _CUSTOM_KINDS:
        params = {"kind": candidate.intent, **params}
    return params


def _resolve(
    ref: str, schema: list[str] | None, sentence: str
) -> tuple[Resolution | None, UnresolvedRef | None]:
    """Resolve *ref*, or explain why it could not be resolved."""
    if schema is None:
        return None, None  # schema-free compile: defer resolution
    resolution = resolve_reference(ref, schema, threshold=DEFAULT_THRESHOLD)
    if resolution.resolved:
        return resolution, None
    return None, UnresolvedRef(
        ref=ref,
        sentence=sentence,
        reason=resolution.reason,
        candidates=resolution.candidates,
    )


def compile_context(
    text: str,
    df: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
    config: CleanConfig | None = None,
    strict: bool = False,
) -> ContextPolicy:
    """Compile natural-language *text* into an inspectable :class:`ContextPolicy`.

    Deterministic, model-free, offline. With a schema (*df* or *columns*),
    column phrases are resolved through the exact/snake_case/alias/difflib
    ladder; without one, constraints keep ``column=None`` and are resolved
    later (e.g. when the policy meets a frame inside ``fd.clean``).

    In strict mode any unresolved reference, unparsed sentence, or protection
    conflict raises :class:`PolicyError` before any data is touched.
    """
    parsed = parse_context(text)
    schema = effective_columns(df, columns, config)

    constraints: list[ColumnConstraint] = []
    unresolved: list[UnresolvedRef] = []
    issues: list[PolicyIssue] = []
    dataset_domain: str | None = None

    for candidate in parsed.candidates:
        assert isinstance(candidate, IntentCandidate)
        sentence = candidate.provenance.sentence

        if candidate.intent == "domain":
            name = str(candidate.params.get("name") or "") or None
            if dataset_domain is not None and name is not None and name != dataset_domain:
                issues.append(
                    PolicyIssue(
                        kind="domain_overridden",
                        message=f"dataset domain restated: {dataset_domain!r} -> {name!r} "
                        "(last statement wins)",
                        severity="warning",
                        sentences=(sentence,),
                    )
                )
            dataset_domain = name or dataset_domain
            continue

        rule, action, enforcement = _INTENT_TABLE[candidate.intent]
        params = _constraint_params(candidate)

        if candidate.intent == "dedup_key":
            resolved_cols: list[str] = []
            failed = False
            for ref in candidate.column_refs:
                resolution, miss = _resolve(ref, schema, sentence)
                if miss is not None:
                    unresolved.append(miss)
                    failed = True
                elif resolution is not None:
                    resolved_cols.append(str(resolution.column))
                else:  # schema-free: keep the raw reference for later resolution
                    resolved_cols.append(ref)
            if failed:
                continue
            constraints.append(
                ColumnConstraint(
                    id="",  # assigned below
                    column=None,
                    resolved_from=", ".join(candidate.column_refs),
                    resolution_confidence=1.0 if schema is not None else 0.0,
                    rule=rule,
                    action=action,
                    params={**params, "columns": resolved_cols},
                    enforcement=enforcement,
                    provenance=candidate.provenance,
                )
            )
            continue

        ref = candidate.column_refs[0]
        resolution, miss = _resolve(ref, schema, sentence)
        if miss is not None:
            unresolved.append(miss)
            continue
        constraints.append(
            ColumnConstraint(
                id="",
                column=(str(resolution.column) if resolution is not None else None),
                resolved_from=ref,
                resolution_confidence=(resolution.confidence if resolution is not None else 0.0),
                rule=rule,
                action=action,
                params=params,
                enforcement=enforcement,
                provenance=candidate.provenance,
            )
        )

    constraints = _drop_superseded(constraints, issues)
    constraints = _enforce_protection(constraints, issues)
    constraints = [
        ColumnConstraint(
            id=f"c{i}",
            column=c.column,
            resolved_from=c.resolved_from,
            resolution_confidence=c.resolution_confidence,
            rule=c.rule,
            action=c.action,
            params=c.params,
            enforcement=c.enforcement,
            provenance=c.provenance,
        )
        for i, c in enumerate(constraints, start=1)
    ]

    for sentence_obj in parsed.unparsed:
        issues.append(
            PolicyIssue(
                kind="unparsed_sentence",
                message=f"no tier-0 pattern matched: {sentence_obj.sentence!r}",  # type: ignore[attr-defined]
                severity="warning",
                sentences=(sentence_obj.sentence,),  # type: ignore[attr-defined]
            )
        )

    policy = ContextPolicy(
        policy_version=POLICY_VERSION,
        dataset_domain=dataset_domain,
        constraints=tuple(constraints),
        unresolved=tuple(unresolved),
        issues=tuple(issues),
        source_text_sha256=sha256_text(text),
        strict=strict,
    )
    if strict:
        _raise_if_dirty(policy)
    return policy


def _drop_superseded(
    constraints: list[ColumnConstraint], issues: list[PolicyIssue]
) -> list[ColumnConstraint]:
    """Same (rule, column) stated twice: the last statement wins, with a warning."""
    keyed: dict[tuple[str, str | None, str], ColumnConstraint] = {}
    order: list[tuple[str, str | None, str]] = []
    for c in constraints:
        key = (c.rule, c.column, c.resolved_from if c.column is None else "")
        if key in keyed and keyed[key].params != c.params:
            issues.append(
                PolicyIssue(
                    kind="superseded",
                    message=f"{c.rule} for {c.column or c.resolved_from!r} restated; "
                    "the last statement wins",
                    severity="warning",
                    sentences=tuple(
                        p.sentence for p in (keyed[key].provenance, c.provenance) if p
                    ),
                    columns=(c.column,) if c.column else (),
                )
            )
        if key not in keyed:
            order.append(key)
        keyed[key] = c
    return [keyed[key] for key in order]


def _enforce_protection(
    constraints: list[ColumnConstraint], issues: list[PolicyIssue]
) -> list[ColumnConstraint]:
    """Protection wins ties: mutating intents on protected columns are demoted."""
    protected: dict[str, ColumnConstraint] = {
        str(c.column): c
        for c in constraints
        if c.rule == "protected" and c.column is not None
    }
    if not protected:
        return constraints
    out: list[ColumnConstraint] = []
    for c in constraints:
        if c.column is None or c.column not in protected or c.action not in MUTATING_ACTIONS:
            out.append(c)
            continue
        guard = protected[c.column]
        issues.append(
            PolicyIssue(
                kind="protection_conflict",
                message=f"column {c.column!r} is both protected and asked to be "
                f"repaired ({c.rule}); protection wins — the {c.rule} constraint "
                "was demoted to validate_only",
                severity="error",
                sentences=tuple(
                    p.sentence for p in (guard.provenance, c.provenance) if p
                ),
                columns=(c.column,),
            )
        )
        out.append(
            ColumnConstraint(
                id=c.id,
                column=c.column,
                resolved_from=c.resolved_from,
                resolution_confidence=c.resolution_confidence,
                rule=c.rule,
                action="validate_only",
                params=c.params,
                enforcement=c.enforcement,
                provenance=c.provenance,
            )
        )
    return out


def _raise_if_dirty(policy: ContextPolicy) -> None:
    """Strict mode: refuse to hand back a policy with unresolved/unparsed parts."""
    problems: list[str] = []
    items: list[Any] = []
    for u in policy.unresolved:
        problems.append(f"unresolved column reference {u.ref!r}: {u.reason}")
        items.append(u)
    for issue in policy.issues:
        if issue.kind == "unparsed_sentence" or issue.severity == "error":
            problems.append(f"{issue.kind}: {issue.message}")
            items.append(issue)
    if problems:
        raise PolicyError(
            "strict context compile failed:\n  - " + "\n  - ".join(problems),
            items=tuple(items),
        )


def resolve_policy(policy: ContextPolicy, columns: Sequence[str]) -> ContextPolicy:
    """Resolve any still-unresolved constraint columns against *columns*.

    Used when a schema-free policy (compiled without a frame) finally meets a
    real frame inside ``fd.clean``/``fd.suggest_plan``. Already-resolved
    constraints are left untouched; strictness is honoured by the caller.
    """
    if all(c.column is not None or c.rule == "dedup_key" for c in policy.constraints):
        needs_dedup = any(
            c.rule == "dedup_key" and c.resolution_confidence == 0.0
            for c in policy.constraints
        )
        if not needs_dedup:
            return policy

    schema = [str(c) for c in columns]
    constraints: list[ColumnConstraint] = []
    unresolved = list(policy.unresolved)
    for c in policy.constraints:
        if c.rule == "dedup_key" and c.resolution_confidence == 0.0:
            resolved_cols: list[str] = []
            failed = False
            for ref in c.params.get("columns", ()):  # raw refs from schema-free compile
                resolution = resolve_reference(str(ref), schema)
                if not resolution.resolved:
                    unresolved.append(
                        UnresolvedRef(
                            ref=str(ref),
                            sentence=c.provenance.sentence if c.provenance else "",
                            reason=resolution.reason,
                            candidates=resolution.candidates,
                        )
                    )
                    failed = True
                else:
                    resolved_cols.append(str(resolution.column))
            if failed:
                continue
            constraints.append(
                ColumnConstraint(
                    id=c.id,
                    column=None,
                    resolved_from=c.resolved_from,
                    resolution_confidence=1.0,
                    rule=c.rule,
                    action=c.action,
                    params={**c.params, "columns": resolved_cols},
                    enforcement=c.enforcement,
                    provenance=c.provenance,
                )
            )
            continue
        if c.column is not None:
            constraints.append(c)
            continue
        resolution = resolve_reference(c.resolved_from, schema)
        if not resolution.resolved:
            unresolved.append(
                UnresolvedRef(
                    ref=c.resolved_from,
                    sentence=c.provenance.sentence if c.provenance else "",
                    reason=resolution.reason,
                    candidates=resolution.candidates,
                )
            )
            continue
        constraints.append(
            ColumnConstraint(
                id=c.id,
                column=str(resolution.column),
                resolved_from=c.resolved_from,
                resolution_confidence=resolution.confidence,
                rule=c.rule,
                action=c.action,
                params=c.params,
                enforcement=c.enforcement,
                provenance=c.provenance,
            )
        )
    return ContextPolicy(
        policy_version=policy.policy_version,
        dataset_domain=policy.dataset_domain,
        constraints=tuple(constraints),
        unresolved=tuple(unresolved),
        issues=policy.issues,
        source_text_sha256=policy.source_text_sha256,
        strict=policy.strict,
    )


def apply_policy_to_config(
    cfg: CleanConfig,
    *,
    df: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
    report: Any | None = None,
) -> CleanConfig:
    """Turn ``cfg.context``/``cfg.policy`` into a concrete, lowered config.

    The single entry point used by ``run_pipeline`` and ``suggest_plan``:

    - ``context`` text is compiled against the effective schema;
    - a supplied ``policy`` skips parsing (schema-free ones get their columns
      resolved now);
    - the policy is lowered into ``preserve_columns`` / ``id_columns`` /
      ``semantic_context`` and attached as ``cfg.policy``;
    - unresolved references and issues are surfaced on the *report*.

    With ``cfg.context`` and ``cfg.policy`` both ``None`` the config is
    returned unchanged — the zero-behaviour-change guarantee.
    """
    if cfg.context is None and cfg.policy is None:
        return cfg

    if cfg.context is not None:
        policy = compile_context(
            cfg.context, df=df, columns=columns, config=cfg, strict=cfg.strict
        )
    else:
        policy = cfg.policy  # type: ignore[assignment]
        if not isinstance(policy, ContextPolicy):
            raise TypeError(
                f"policy must be a freshdata.ContextPolicy, got {type(policy).__name__}"
            )
        schema = effective_columns(df, columns, cfg)
        if schema is not None:
            policy = resolve_policy(policy, schema)
        if cfg.strict:
            _raise_if_dirty(policy)

    lowered = policy.lower(cfg)

    if report is not None:
        report.add(
            step="context",
            description=f"compiled context policy: {len(policy.constraints)} constraint(s), "
            f"{len(policy.unresolved)} unresolved, {len(policy.issues)} issue(s)",
            rationale="deterministic tier-0 context compiler",
            metadata={"policy": policy.to_dict()},
        )
        for u in policy.unresolved:
            cands = ", ".join(f"{c} ({s:.2f})" for c, s in u.candidates[:3])
            report.add_warning(
                f"[context] unresolved column reference {u.ref!r}: {u.reason}"
                + (f" — candidates: {cands}" if cands else "")
            )
        for issue in policy.issues:
            report.add_warning(f"[context] {issue.kind}: {issue.message}")
        for col in policy.protected_columns:
            report.add(
                step="context",
                description=f"column {col!r} protected by policy (never modified)",
                column=col,
                rationale="context policy: never_modify",
                metadata={"enforcement": "hard"},
            )
    return lowered
