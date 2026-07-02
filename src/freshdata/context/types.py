"""Typed, JSON-round-trippable building blocks of a compiled context policy.

Everything here is deterministic data: the parser emits :class:`IntentCandidate`
and :class:`UnparsedSentence`, the resolver emits :class:`UnresolvedRef`, and the
compiler assembles them into a :class:`ContextPolicy` — the reviewable contract
that :meth:`ContextPolicy.lower` turns into a plain :class:`~freshdata.CleanConfig`.
No model, no I/O beyond the optional ``to_json(path=...)`` convenience, and no
imports heavier than the stdlib, so ``import freshdata.context`` stays cheap.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import CleanConfig

#: Serialization format version; bump when the JSON layout changes.
POLICY_VERSION = "1"

#: Constraint rules the compiler can emit (mirrors the intent lexicon).
CONSTRAINT_RULES = (
    "unique",
    "valid_format",
    "locale_format",
    "protected",
    "impute_missing",
    "allowed_values",
    "range",
    "dedup_key",
    "custom",
)

#: What the engine is asked to do about a constraint.
CONSTRAINT_ACTIONS = (
    "validate_only",
    "repair_or_flag",
    "normalize_or_flag",
    "never_modify",
    "impute_if_confident_else_preserve",
    "custom",
)

#: Rules that request a mutation of the column's values (protection conflicts).
MUTATING_ACTIONS = frozenset(
    {"repair_or_flag", "normalize_or_flag", "impute_if_confident_else_preserve"}
)


class PolicyError(ValueError):
    """Raised in strict mode when a context cannot be compiled cleanly.

    Carries the offending :class:`UnresolvedRef` / :class:`PolicyIssue` /
    :class:`UnparsedSentence` items so callers (and the CLI) can show exactly
    which sentences need fixing before any data is touched.
    """

    def __init__(self, message: str, *, items: tuple[Any, ...] = ()) -> None:
        super().__init__(message)
        self.items = items


@dataclass(frozen=True)
class Provenance:
    """Where a constraint came from: the raw sentence and the parser tier."""

    sentence: str
    tier: int = 0
    parse_confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        return cls(
            sentence=str(data["sentence"]),
            tier=int(data.get("tier", 0)),
            parse_confidence=float(data.get("parse_confidence", 1.0)),
        )


@dataclass(frozen=True)
class IntentCandidate:
    """One parsed intent, before column resolution.

    ``column_refs`` holds the raw column phrases exactly as written by the user
    (``"CustomerID"``, ``"Phone numbers"``); most intents have one, ``dedup_key``
    may have several, ``domain`` has none.
    """

    intent: str
    column_refs: tuple[str, ...]
    params: dict[str, Any]
    provenance: Provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "column_refs": list(self.column_refs),
            "params": dict(self.params),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class UnparsedSentence:
    """A sentence no tier-0 pattern matched — surfaced, never silently dropped."""

    sentence: str
    reason: str = "no tier-0 intent pattern matched"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class UnresolvedRef:
    """A column phrase the resolver could not map to exactly one schema column."""

    ref: str
    sentence: str
    reason: str
    #: Top scoring candidates as ``(column, score)``, best first.
    candidates: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "sentence": self.sentence,
            "reason": self.reason,
            "candidates": [[c, round(s, 4)] for c, s in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnresolvedRef:
        return cls(
            ref=str(data["ref"]),
            sentence=str(data.get("sentence", "")),
            reason=str(data.get("reason", "")),
            candidates=tuple((str(c), float(s)) for c, s in data.get("candidates", ())),
        )


@dataclass(frozen=True)
class PolicyIssue:
    """A compile-time problem worth surfacing (conflict, unparsed line, override)."""

    kind: str
    message: str
    severity: str = "warning"  # "warning" | "error"
    sentences: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "severity": self.severity,
            "sentences": list(self.sentences),
            "columns": list(self.columns),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyIssue:
        return cls(
            kind=str(data["kind"]),
            message=str(data["message"]),
            severity=str(data.get("severity", "warning")),
            sentences=tuple(str(s) for s in data.get("sentences", ())),
            columns=tuple(str(c) for c in data.get("columns", ())),
        )


@dataclass(frozen=True)
class ColumnConstraint:
    """One compiled rule about one column (or a column group for ``dedup_key``)."""

    id: str
    column: str | None
    resolved_from: str
    resolution_confidence: float
    rule: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    enforcement: str = "soft"  # "soft" | "hard"
    provenance: Provenance | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "column": self.column,
            "resolved_from": self.resolved_from,
            "resolution_confidence": round(self.resolution_confidence, 4),
            "rule": self.rule,
            "action": self.action,
            "params": dict(self.params),
            "enforcement": self.enforcement,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnConstraint:
        prov = data.get("provenance")
        return cls(
            id=str(data["id"]),
            column=(str(data["column"]) if data.get("column") is not None else None),
            resolved_from=str(data.get("resolved_from", "")),
            resolution_confidence=float(data.get("resolution_confidence", 0.0)),
            rule=str(data["rule"]),
            action=str(data["action"]),
            params=dict(data.get("params", {})),
            enforcement=str(data.get("enforcement", "soft")),
            provenance=Provenance.from_dict(prov) if prov else None,
        )


@dataclass(frozen=True)
class Thresholds:
    """Effective decision thresholds for one (column, kind) pair."""

    auto: float
    review: float
    floor: float = 0.50
    #: True when a context constraint (not the global config) set ``auto``.
    from_policy: bool = False


def sha256_text(text: str) -> str:
    """Stable fingerprint of the source context text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fold_constraint(
    c: ColumnConstraint,
    meta: dict[str, Any],
    preserve: list[str],
    id_cols: list[str],
) -> None:
    """Apply one resolved constraint's lowering onto per-column config state.

    "custom" rules (drop_if / rename / map) carry no Phase-1 lowering; they stay
    in the policy for the executor phases and for :func:`freshdata.validate`.
    """
    assert c.column is not None
    if c.rule == "protected":
        if c.column not in preserve:
            preserve.append(c.column)
        meta["mutable"] = False
    elif c.rule == "unique":
        if c.column not in id_cols:
            id_cols.append(c.column)
        meta["unique"] = True
    elif c.rule == "valid_format":
        meta.setdefault("semantic_type", c.params.get("format"))
    elif c.rule == "locale_format":
        meta.setdefault("semantic_type", c.params.get("format"))
        if c.params.get("region"):
            meta["region"] = c.params["region"]
    elif c.rule == "impute_missing":
        if c.params.get("min_confidence") is not None:
            meta["impute_min_confidence"] = c.params["min_confidence"]
    elif c.rule == "allowed_values":
        meta.setdefault("allowed_values", list(c.params.get("values", ())))
    elif c.rule == "range":
        if c.params.get("lo") is not None:
            meta["min_value"] = c.params["lo"]
        if c.params.get("hi") is not None:
            meta["max_value"] = c.params["hi"]


@dataclass(frozen=True)
class ContextPolicy:
    """The compiled, inspectable contract between the user's prose and the engine.

    Immutable and JSON-round-trippable (:meth:`to_json` / :meth:`from_json`), so a
    policy can be reviewed in a pull request like code and passed back verbatim via
    ``fd.clean(df, policy=...)``.
    """

    policy_version: str = POLICY_VERSION
    dataset_domain: str | None = None
    constraints: tuple[ColumnConstraint, ...] = ()
    unresolved: tuple[UnresolvedRef, ...] = ()
    issues: tuple[PolicyIssue, ...] = ()
    source_text_sha256: str | None = None
    strict: bool = False

    # -- introspection ------------------------------------------------------

    def constraints_for(self, column: str) -> tuple[ColumnConstraint, ...]:
        """All constraints that apply to *column* (post-normalization name)."""
        from .normalize import snake_ref  # noqa: PLC0415 - avoid import cycle at load

        matched = []
        for c in self.constraints:
            if c.column == column or (
                c.column is None and snake_ref(c.resolved_from) == snake_ref(column)
            ):
                matched.append(c)
        return tuple(matched)

    def is_protected(self, column: str) -> bool:
        """True when the policy forbids any modification of *column*."""
        return any(c.rule == "protected" for c in self.constraints_for(column))

    def thresholds(self, column: str, kind: str, cfg: CleanConfig) -> Thresholds:
        """Effective (auto, review, floor) thresholds for *column* and *kind*.

        ``impute_missing`` constraints raise the auto threshold for
        ``kind="impute"`` (or ``"missing"``); everything else falls back to the
        global config thresholds.
        """
        auto = cfg.semantic_auto_threshold
        review = cfg.semantic_review_threshold
        from_policy = False
        if kind in ("impute", "missing"):
            for c in self.constraints_for(column):
                if c.rule == "impute_missing":
                    min_conf = c.params.get("min_confidence")
                    if isinstance(min_conf, (int, float)):
                        auto = max(auto, float(min_conf))
                        from_policy = True
        return Thresholds(
            auto=auto, review=min(review, auto), floor=0.50, from_policy=from_policy
        )

    @property
    def protected_columns(self) -> tuple[str, ...]:
        """Resolved columns under a ``protected`` constraint, in policy order."""
        out = []
        for c in self.constraints:
            if c.rule == "protected" and c.column is not None and c.column not in out:
                out.append(c.column)
        return tuple(out)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "dataset_domain": self.dataset_domain,
            "constraints": [c.to_dict() for c in self.constraints],
            "unresolved": [u.to_dict() for u in self.unresolved],
            "issues": [i.to_dict() for i in self.issues],
            "source_text_sha256": self.source_text_sha256,
            "strict": self.strict,
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize to JSON; optionally also write it to *path*."""
        text = json.dumps(self.to_dict(), indent=2, sort_keys=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextPolicy:
        return cls(
            policy_version=str(data.get("policy_version", POLICY_VERSION)),
            dataset_domain=(
                str(data["dataset_domain"]) if data.get("dataset_domain") is not None else None
            ),
            constraints=tuple(
                ColumnConstraint.from_dict(c) for c in data.get("constraints", ())
            ),
            unresolved=tuple(UnresolvedRef.from_dict(u) for u in data.get("unresolved", ())),
            issues=tuple(PolicyIssue.from_dict(i) for i in data.get("issues", ())),
            source_text_sha256=(
                str(data["source_text_sha256"])
                if data.get("source_text_sha256") is not None
                else None
            ),
            strict=bool(data.get("strict", False)),
        )

    @classmethod
    def from_json(cls, data: str | Path) -> ContextPolicy:
        """Load a policy from a JSON string or a path to a ``.json`` file."""
        if isinstance(data, Path) or (isinstance(data, str) and data.lstrip()[:1] != "{"):
            text = Path(data).read_text(encoding="utf-8")
        else:
            text = data
        return cls.from_dict(json.loads(text))

    # -- presentation -------------------------------------------------------

    def summary(self) -> str:
        """Human-readable one-screen description of the compiled policy."""
        lines = [f"freshdata context policy (v{self.policy_version})"]
        if self.dataset_domain:
            lines.append(f"  domain: {self.dataset_domain}")
        lines.append(f"  constraints: {len(self.constraints)}")
        for c in self.constraints:
            col = c.column if c.column is not None else f"?{c.resolved_from!r}"
            detail = ""
            if c.rule in ("valid_format", "locale_format"):
                detail = f" format={c.params.get('format')}"
                if c.params.get("region"):
                    detail += f" region={c.params['region']}"
            elif c.rule == "impute_missing":
                detail = f" min_confidence={c.params.get('min_confidence')}"
            elif c.rule == "allowed_values":
                detail = f" values={list(c.params.get('values', ()))}"
            elif c.rule == "range":
                detail = f" [{c.params.get('lo')}, {c.params.get('hi')}]"
            elif c.rule == "dedup_key":
                detail = f" columns={list(c.params.get('columns', ()))}"
            elif c.rule == "custom":
                detail = f" kind={c.params.get('kind')}"
            hard = " [hard]" if c.enforcement == "hard" else ""
            src = ""
            if c.resolved_from and c.resolved_from != c.column:
                src = f"  (from {c.resolved_from!r}, {c.resolution_confidence:.2f})"
            lines.append(f"    {c.id}: {c.rule:<14} {col}{detail}{hard}{src}")
        if self.unresolved:
            lines.append(f"  unresolved references: {len(self.unresolved)}")
            for u in self.unresolved:
                cands = ", ".join(f"{c} ({s:.2f})" for c, s in u.candidates[:3])
                suffix = f" — candidates: {cands}" if cands else ""
                lines.append(f"    {u.ref!r}: {u.reason}{suffix}")
        if self.issues:
            lines.append(f"  issues: {len(self.issues)}")
            for issue in self.issues:
                lines.append(f"    [{issue.severity}] {issue.kind}: {issue.message}")
        return "\n".join(lines)

    # -- lowering -----------------------------------------------------------

    def lower(self, cfg: CleanConfig) -> CleanConfig:
        """Return a new :class:`CleanConfig` with this policy folded in.

        Pure: *cfg* is never mutated. Protected columns are appended to
        ``preserve_columns``, unique columns to ``id_columns``, per-column
        semantic hints (semantic_type, region, allowed_values, range,
        impute confidence, mutability) are merged into ``semantic_context``,
        and the policy itself is attached as ``cfg.policy`` (with ``context``
        cleared so the text is never recompiled downstream).
        """
        from ..config import CleanConfig  # noqa: PLC0415 - runtime import, cycle-safe

        if not isinstance(cfg, CleanConfig):
            raise TypeError(f"lower() expects a CleanConfig, got {type(cfg).__name__}")

        preserve = list(cfg.preserve_columns)
        id_cols = list(cfg.id_columns)
        duplicate_subset = cfg.duplicate_subset

        semantic: dict[str, Any] = {}
        if isinstance(cfg.semantic_context, dict):
            semantic = dict(cfg.semantic_context)
        columns_meta: dict[str, dict[str, Any]] = {
            str(k): dict(v)
            for k, v in (semantic.get("columns") or {}).items()
            if isinstance(v, dict)
        }

        if self.dataset_domain and not semantic.get("dataset"):
            semantic["dataset"] = self.dataset_domain

        protected = set(self.protected_columns)
        for c in self.constraints:
            if c.rule == "dedup_key":
                # Group constraint (column=None by design): resolved members
                # live in params["columns"].
                cols = tuple(str(x) for x in c.params.get("columns", ()))
                if cols and duplicate_subset is None and c.resolution_confidence > 0.0:
                    duplicate_subset = cols
                continue
            if c.column is None:
                continue
            meta = columns_meta.setdefault(c.column, {})
            _fold_constraint(c, meta, preserve, id_cols)
            if c.rule != "protected" and c.column in protected:
                # Protection wins ties: a protected column never gains a hint
                # that could be read as permission to mutate.
                meta.pop("impute_min_confidence", None)
                meta["mutable"] = False

        if columns_meta:
            semantic["columns"] = columns_meta

        return dataclasses.replace(
            cfg,
            preserve_columns=tuple(preserve),
            id_columns=tuple(id_cols),
            duplicate_subset=duplicate_subset,
            semantic_context=semantic if semantic else cfg.semantic_context,
            policy=self,
            context=None,
        )
