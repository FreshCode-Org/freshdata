"""Executable, reviewable repair plans (Phase 2).

:func:`freshdata.suggest_plan` proposes; a human (or calling system) approves,
rejects, or overrides; :func:`freshdata.apply_plan` executes **exactly** the
approved actions — no re-profiling, no re-deciding — under the physical
protected-column guard, against the same frame the plan was built for
(:class:`FrameSignature` drift refusal), with an optional compact undo log and
a deterministic :attr:`~freshdata.CleanReport.decisions_hash` for audit.

Everything here is deterministic and offline. Dataclasses are plain (no
``slots``) because the package still supports Python 3.9.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CleanConfig, merge_options
from .report import CleanReport

#: Ordering used by ``approve_all(max_risk=...)``.
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

#: Rows hashed (from the head) for the cheap frame-content fingerprint.
_SIGNATURE_SAMPLE_ROWS = 512


class PlanDriftError(ValueError):
    """The frame no longer matches the one the plan was suggested for.

    Raised by :func:`freshdata.apply_plan` before touching any data. Pass
    ``allow_drift=True`` to apply anyway (actions whose columns or values no
    longer exist are skipped and recorded).
    """


@dataclass(frozen=True)
class FrameSignature:
    """A cheap, deterministic fingerprint of a frame's shape and content."""

    n_rows: int
    columns_hash: str
    sample_hash: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameSignature:
        return cls(
            n_rows=int(data["n_rows"]),
            columns_hash=str(data["columns_hash"]),
            sample_hash=str(data["sample_hash"]),
        )


def compute_frame_signature(df: pd.DataFrame) -> FrameSignature:
    """Fingerprint *df*: row count, column names+dtypes, and a head sample."""
    columns_payload = json.dumps(
        [[str(c), str(df[c].dtype)] for c in df.columns], ensure_ascii=False
    )
    columns_hash = hashlib.sha256(columns_payload.encode("utf-8")).hexdigest()
    sample = df.iloc[:_SIGNATURE_SAMPLE_ROWS]
    try:
        hashed = pd.util.hash_pandas_object(sample, index=True)
        sample_hash = hashlib.sha256(hashed.values.tobytes()).hexdigest()
    except TypeError:  # unhashable payloads (lists/dicts in cells)
        fallback = json.dumps([len(sample), [str(c) for c in sample.columns]])
        sample_hash = hashlib.sha256(fallback.encode("utf-8")).hexdigest()
    return FrameSignature(
        n_rows=len(df), columns_hash=columns_hash, sample_hash=sample_hash
    )


@dataclass(frozen=True)
class RepairProposal:
    """One scored candidate repair, as surfaced by a semantic expert."""

    column: str
    raw_value: object
    proposed_value: object
    issue_type: str
    backend: str
    raw_score: float
    evidence: Mapping[str, object]
    risk: str  # "low" | "medium" | "high"


@dataclass
class PlannedAction:
    """One reviewable unit of work in a :class:`RepairPlan`."""

    id: str
    column: str | None
    kind: str
    params: dict[str, Any]
    source: str
    confidence: float
    risk: str  # "low" | "medium" | "high"
    n_affected: int
    examples: tuple[tuple[object, object], ...]
    decision: str  # "auto" | "suggest" | "skip" | "blocked"
    approval: str = "pending"  # "pending" | "approved" | "rejected"
    reversible: bool | None = None
    policy_rule_id: str | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "column": self.column,
            "kind": self.kind,
            "params": _json_safe(self.params),
            "source": self.source,
            "confidence": self.confidence,
            "risk": self.risk,
            "n_affected": self.n_affected,
            "examples": [[_json_safe(a), _json_safe(b)] for a, b in self.examples],
            "decision": self.decision,
            "approval": self.approval,
            "reversible": self.reversible,
            "policy_rule_id": self.policy_rule_id,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlannedAction:
        return cls(
            id=str(data["id"]),
            column=(str(data["column"]) if data.get("column") is not None else None),
            kind=str(data["kind"]),
            params=dict(data.get("params", {})),
            source=str(data.get("source", "")),
            confidence=float(data.get("confidence", 0.0)),
            risk=str(data.get("risk", "high")),
            n_affected=int(data.get("n_affected", 0)),
            examples=tuple(
                (a, b) for a, b in (tuple(e) for e in data.get("examples", ()))
            ),
            decision=str(data.get("decision", "suggest")),
            approval=str(data.get("approval", "pending")),
            reversible=data.get("reversible"),
            policy_rule_id=(
                str(data["policy_rule_id"])
                if data.get("policy_rule_id") is not None
                else None
            ),
            rationale=str(data.get("rationale", "")),
        )


def _json_safe(value: object) -> object:
    """Best-effort conversion of audit payloads to JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):  # NaN / NaT scalars
        return None
    return repr(value)


@dataclass
class RepairPlan:
    """A reviewable, serializable, executable set of planned repairs.

    Approve / reject / override actions by selector (an action ID, a column
    name, an action kind, an iterable of those, or a predicate), then execute
    with :func:`freshdata.apply_plan`. Blocked actions (protected columns,
    identifier vetoes) never execute regardless of approval.
    """

    actions: list[PlannedAction]
    policy: Any  # ContextPolicy | None — typed loosely to keep imports light
    config: CleanConfig
    frame_signature: FrameSignature
    schema_diff: Any = None
    #: Serialization format version.
    plan_version: str = "1"
    #: Free-form reasons recorded by :meth:`reject`.
    rejection_reasons: dict[str, str] = field(default_factory=dict)

    # -- selection ----------------------------------------------------------

    def _select(self, selector: object) -> list[PlannedAction]:
        if selector is None:
            return list(self.actions)
        if callable(selector) and not isinstance(selector, str):
            return [a for a in self.actions if selector(a)]
        if isinstance(selector, PlannedAction):
            keys: set[str] = {selector.id}
        elif isinstance(selector, str):
            keys = {selector}
        elif isinstance(selector, Iterable):
            keys = {s.id if isinstance(s, PlannedAction) else str(s) for s in selector}
        else:
            raise TypeError(
                "selector must be an action id / column / kind string, an "
                f"iterable of those, or a predicate; got {type(selector).__name__}"
            )
        matched = [
            a
            for a in self.actions
            if a.id in keys or (a.column is not None and a.column in keys) or a.kind in keys
        ]
        if not matched:
            raise KeyError(
                f"selector {sorted(keys)!r} matched no planned action; known ids: "
                f"{[a.id for a in self.actions]}"
            )
        return matched

    # -- review workflow ----------------------------------------------------

    def approve(self, selector: object) -> RepairPlan:
        """Approve the selected actions (blocked actions stay blocked)."""
        for action in self._select(selector):
            if action.decision == "blocked":
                continue
            action.approval = "approved"
        return self

    def reject(self, selector: object, reason: str = "") -> RepairPlan:
        """Reject the selected actions; they will never execute."""
        for action in self._select(selector):
            action.approval = "rejected"
            if reason:
                self.rejection_reasons[action.id] = reason
        return self

    def override(self, selector: object, params: Mapping[str, object]) -> RepairPlan:
        """Replace action parameters (e.g. ``proposed_value``) before approval.

        Overridden actions are marked (``source`` gains ``+override``) so the
        audit trail shows a human changed the machine proposal.
        """
        for action in self._select(selector):
            if action.decision == "blocked":
                continue
            action.params = {**action.params, **dict(params)}
            if "proposed_value" in params:
                raw = action.params.get("raw_value")
                action.examples = ((raw, params["proposed_value"]),)
            if not action.source.endswith("+override"):
                action.source = f"{action.source}+override"
        return self

    def approve_all(self, max_risk: str = "low") -> RepairPlan:
        """Approve every non-blocked, non-rejected action at or below *max_risk*."""
        if max_risk not in _RISK_ORDER:
            raise ValueError(f"max_risk must be one of {sorted(_RISK_ORDER)}, got {max_risk!r}")
        ceiling = _RISK_ORDER[max_risk]
        for action in self.actions:
            if action.decision == "blocked" or action.approval == "rejected":
                continue
            if action.decision == "skip":
                continue
            if _RISK_ORDER.get(action.risk, 2) <= ceiling:
                action.approval = "approved"
        return self

    # -- presentation ---------------------------------------------------------

    def summary(self) -> str:
        """Stable, human-readable one-screen review of the plan."""
        lines = [
            f"freshdata repair plan (v{self.plan_version})",
            f"  frame: {self.frame_signature.n_rows} row(s), "
            f"columns {self.frame_signature.columns_hash[:12]}",
            f"  actions: {len(self.actions)}",
        ]
        for a in self.actions:
            raw = a.params.get("raw_value")
            proposed = a.params.get("proposed_value")
            change = f"{raw!r} -> {proposed!r}" if proposed is not None else f"{raw!r} flagged"
            lines.append(
                f"    {a.id}: [{a.decision}/{a.approval}] {a.column} {a.kind} "
                f"{change} (n={a.n_affected}, conf={a.confidence:.2f}, risk={a.risk})"
            )
        if not self.actions:
            lines.append("    (nothing to do)")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One planned action per row, for notebook review."""
        return pd.DataFrame(
            [
                {
                    "id": a.id,
                    "column": a.column,
                    "kind": a.kind,
                    "raw_value": a.params.get("raw_value"),
                    "proposed_value": a.params.get("proposed_value"),
                    "n_affected": a.n_affected,
                    "confidence": a.confidence,
                    "risk": a.risk,
                    "decision": a.decision,
                    "approval": a.approval,
                    "reversible": a.reversible,
                    "source": a.source,
                    "policy_rule_id": a.policy_rule_id,
                }
                for a in self.actions
            ]
        )

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        n_approved = sum(1 for a in self.actions if a.approval == "approved")
        return (
            f"<RepairPlan: {len(self.actions)} action(s), {n_approved} approved, "
            f"{self.frame_signature.n_rows} row(s)>"
        )

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "actions": [a.to_dict() for a in self.actions],
            "policy": self.policy.to_dict() if self.policy is not None else None,
            "config": _config_overrides(self.config),
            "frame_signature": self.frame_signature.to_dict(),
            "rejection_reasons": dict(self.rejection_reasons),
        }

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RepairPlan:
        policy = None
        if data.get("policy") is not None:
            from .context.types import ContextPolicy  # noqa: PLC0415 — lazy, cycle-safe

            policy = ContextPolicy.from_dict(dict(data["policy"]))
        config = merge_options(None, **dict(data.get("config", {})))
        if policy is not None:
            config = dataclasses.replace(config, policy=policy)
        return cls(
            actions=[PlannedAction.from_dict(a) for a in data.get("actions", ())],
            policy=policy,
            config=config,
            frame_signature=FrameSignature.from_dict(data["frame_signature"]),
            plan_version=str(data.get("plan_version", "1")),
            rejection_reasons=dict(data.get("rejection_reasons", {})),
        )

    @classmethod
    def from_json(cls, data: str | Path) -> RepairPlan:
        if isinstance(data, Path) or (isinstance(data, str) and data.lstrip()[:1] != "{"):
            text = Path(data).read_text(encoding="utf-8")
        else:
            text = data
        return cls.from_dict(json.loads(text))

    # -- audit ----------------------------------------------------------------

    def decisions_hash(self) -> str:
        """Deterministic digest of what was decided (see report.decisions_hash)."""
        return compute_decisions_hash(self)


def _config_overrides(config: CleanConfig) -> dict[str, Any]:
    """JSON-safe non-default config fields (``policy`` is serialized separately)."""
    defaults = CleanConfig()
    overrides: dict[str, Any] = {}
    for f in dataclasses.fields(CleanConfig):
        if f.name in ("policy", "context"):
            continue
        value = getattr(config, f.name)
        if value == getattr(defaults, f.name):
            continue
        safe = _json_safe(value)
        try:
            json.dumps(safe)
        except (TypeError, ValueError):
            continue
        overrides[f.name] = safe
    return overrides


def compute_decisions_hash(plan: RepairPlan) -> str:
    """SHA-256 over action identities, params, decisions, policy, thresholds.

    Deliberately excludes anything unstable (timestamps, object ids, row
    counts) so the same reviewed plan always hashes identically.
    """
    payload = {
        "actions": [
            {
                "id": a.id,
                "kind": a.kind,
                "column": a.column,
                "params": _json_safe(a.params),
                "decision": a.decision,
                "approval": a.approval,
            }
            for a in sorted(plan.actions, key=lambda a: a.id)
        ],
        "policy_sha256": getattr(plan.policy, "source_text_sha256", None),
        "thresholds": {
            "semantic_auto_threshold": plan.config.semantic_auto_threshold,
            "semantic_review_threshold": plan.config.semantic_review_threshold,
            "impute_min_confidence": _impute_thresholds(plan.config),
        },
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _impute_thresholds(config: CleanConfig) -> dict[str, float]:
    semantic = config.semantic_context
    out: dict[str, float] = {}
    if isinstance(semantic, Mapping):
        columns = semantic.get("columns")
        if isinstance(columns, Mapping):
            for name, meta in columns.items():
                if isinstance(meta, Mapping):
                    value = meta.get("impute_min_confidence")
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        out[str(name)] = float(value)
    return out


# --------------------------------------------------------------------------- #
# Building a plan
# --------------------------------------------------------------------------- #

_BLOCKED_MARKERS = (
    "protected",
    "preserve_columns",
    "never modified",
    "mutable",
    "identifier",
)


def _classify_decision(action: str, reason: str, issue_type: str) -> str:
    if action == "apply":
        return "auto"
    if action == "suggest":
        return "suggest"
    if issue_type == "identifier_like":
        return "blocked"
    reason_l = reason.lower()
    if any(marker in reason_l for marker in _BLOCKED_MARKERS):
        return "blocked"
    return "skip"


def _policy_rule_for(policy: Any, column: str | None) -> str | None:
    if policy is None or column is None:
        return None
    for constraint in getattr(policy, "constraints_for", lambda _c: ())(column):
        return constraint.id
    return None


def _action_params(proposal: Any, decision: Any) -> dict[str, Any]:
    """Audit params for one planned action; model provenance rides along."""
    params: dict[str, Any] = {
        "raw_value": proposal.raw_value,
        "proposed_value": proposal.proposed_value,
        "expert": proposal.expert,
        "gate_reason": decision.reason,
    }
    backend = getattr(proposal, "backend", "deterministic")
    if backend != "deterministic":
        params["backend"] = backend
    calibration = getattr(proposal, "calibration", None)
    if calibration is not None and getattr(calibration, "calibration_version", "") not in (
        "",
        "uncalibrated",
    ):
        params["calibration_version"] = calibration.calibration_version
        params["raw_score"] = calibration.raw
        params["features_hash"] = calibration.features_hash
    return params


def build_repair_plan(df: pd.DataFrame, config: CleanConfig) -> RepairPlan:
    """Profile *df* under *config* and assemble a reviewable :class:`RepairPlan`.

    Proposals are generated by the deterministic semantic experts and gated by
    the same policy logic :func:`freshdata.clean` uses, so a plan's ``auto``
    actions are exactly what ``semantic_mode="auto"`` would have applied.
    """
    from .semantic.backends import gather_proposals  # noqa: PLC0415 — lazy
    from .semantic.context import build_semantic_context  # noqa: PLC0415 — lazy
    from .semantic.policy import decide  # noqa: PLC0415
    from .semantic.scoring import calibrate_proposals  # noqa: PLC0415

    signature = compute_frame_signature(df)
    actions: list[PlannedAction] = []

    if config.semantic_enabled or config.policy is not None:
        # A compiled policy implies the user wants proposals even when they
        # didn't set semantic_mode; plan in review posture in that case.
        plan_config = config
        if not config.semantic_enabled:
            plan_config = dataclasses.replace(config, semantic_mode="review")
        ctx = build_semantic_context(df, plan_config)
        proposals = sorted(
            calibrate_proposals(gather_proposals(df, ctx, plan_config), plan_config, ctx),
            key=lambda p: (p.column, p.issue_type, str(p.raw_value)),
        )
        for i, proposal in enumerate(proposals, start=1):
            decision = decide(proposal, plan_config, ctx)
            kind = proposal.issue_type
            classified = _classify_decision(decision.action, decision.reason, kind)
            actions.append(
                PlannedAction(
                    id=f"a{i}",
                    column=proposal.column,
                    kind=kind,
                    params=_action_params(proposal, decision),
                    source=f"expert:{proposal.expert}",
                    confidence=proposal.confidence,
                    risk=decision.risk,
                    n_affected=proposal.count,
                    examples=((proposal.raw_value, proposal.proposed_value),),
                    decision=classified,
                    approval="approved" if classified == "auto" else "pending",
                    reversible=(
                        True if proposal.proposed_value is not None else None
                    ),
                    policy_rule_id=_policy_rule_for(config.policy, proposal.column),
                    rationale=proposal.rationale,
                )
            )

    return RepairPlan(
        actions=actions,
        policy=config.policy,
        config=config,
        frame_signature=signature,
    )


# --------------------------------------------------------------------------- #
# Executing a plan
# --------------------------------------------------------------------------- #


def _executable(action: PlannedAction) -> bool:
    return (
        action.approval == "approved"
        and action.decision in ("auto", "suggest")
        and action.column is not None
        and action.params.get("proposed_value") is not None
    )


def _check_drift(df: pd.DataFrame, plan: RepairPlan) -> None:
    current = compute_frame_signature(df)
    if current == plan.frame_signature:
        return
    drift = []
    if current.n_rows != plan.frame_signature.n_rows:
        drift.append(f"row count {plan.frame_signature.n_rows} -> {current.n_rows}")
    if current.columns_hash != plan.frame_signature.columns_hash:
        drift.append("column names/dtypes changed")
    if current.sample_hash != plan.frame_signature.sample_hash:
        drift.append("cell contents changed")
    raise PlanDriftError(
        "the frame no longer matches the one this plan was suggested for "
        f"({'; '.join(drift) or 'content drift'}). Re-run suggest_plan, or "
        "pass allow_drift=True to apply anyway."
    )


def _collect_mappings(
    out: pd.DataFrame, plan: RepairPlan, report: CleanReport
) -> tuple[dict[str, dict[object, object]], list[PlannedAction]]:
    """Group executable value repairs per column; record stale ones."""
    mappings: dict[str, dict[object, object]] = {}
    executed: list[PlannedAction] = []
    for action in plan.actions:
        if not _executable(action):
            continue
        column = str(action.column)
        if column not in out.columns:
            report.add(
                "apply_plan",
                f"skipped {action.id}: column {column!r} not present (frame drift)",
                column=column,
                status="skipped",
                risk=action.risk,
                metadata={"action_id": action.id},
            )
            continue
        mappings.setdefault(column, {})[action.params.get("raw_value")] = (
            action.params.get("proposed_value")
        )
        executed.append(action)
    return mappings, executed


def _capture_undo(
    out: pd.DataFrame, executed: list[PlannedAction], undo_cell_limit: int
) -> dict[str, Any]:
    """Record pre-apply cell positions/values, honestly capped by the limit."""
    entries: list[dict[str, Any]] = []
    dtypes: dict[str, str] = {}
    budget = max(0, int(undo_cell_limit))
    for action in executed:
        column = str(action.column)
        raw = action.params.get("raw_value")
        indices = out.index[out[column] == raw].tolist()
        if len(indices) <= budget:
            budget -= len(indices)
            dtypes.setdefault(column, str(out[column].dtype))
            entries.append(
                {"action_id": action.id, "column": column, "index": indices,
                 "value": raw}
            )
            action.reversible = True
        else:
            action.reversible = False  # honest: undo would exceed the cap
    return {"entries": entries, "column_dtypes": dtypes}


def _record_action(
    report: CleanReport, plan: RepairPlan, action: PlannedAction, *, applied: bool
) -> None:
    column = str(action.column) if action.column is not None else None
    metadata = {
        "action_id": action.id,
        "issue_type": action.kind,
        "raw_value": _json_safe(action.params.get("raw_value")),
        "proposed_value": _json_safe(action.params.get("proposed_value")),
        "expert": action.params.get("expert"),
        "policy_rule_id": action.policy_rule_id,
    }
    if applied:
        report.add(
            "apply_plan",
            f"applied {action.id}: {action.params.get('raw_value')!r} -> "
            f"{action.params.get('proposed_value')!r}",
            column=column,
            count=action.n_affected,
            rationale=action.rationale,
            risk=action.risk,
            confidence=action.confidence,
            model_id=f"plan:{action.kind}",
            status="approved",
            reversible=action.reversible,
            metadata=metadata,
        )
        return
    if action.approval == "rejected":
        why = plan.rejection_reasons.get(action.id, "rejected by reviewer")
        status = "skipped"
    elif action.decision == "blocked":
        why = str(action.params.get("gate_reason", "blocked by policy"))
        status = "skipped"
    elif action.approval == "pending":
        why, status = "not approved", "suggested"
    else:
        why, status = str(action.params.get("gate_reason", "skipped")), "skipped"
    report.add(
        "apply_plan",
        f"did not apply {action.id} ({why})",
        column=column,
        count=0,
        rationale=action.rationale,
        risk=action.risk,
        confidence=action.confidence,
        model_id=f"plan:{action.kind}",
        status=status,
        human_review=status == "suggested",
        metadata=metadata,
    )


def execute_plan(
    df: pd.DataFrame,
    plan: RepairPlan,
    *,
    keep_undo: bool = False,
    allow_drift: bool = False,
    undo_cell_limit: int = 100_000,
) -> tuple[pd.DataFrame, CleanReport]:
    """Execute exactly the approved actions of *plan* against *df*.

    See :func:`freshdata.apply_plan` for the public contract. The input frame
    is never mutated; on any failure (including a protected-column violation)
    the caller's frame is untouched.
    """
    from .guard import protected_column_set, snapshot_protected, verify_protected  # noqa: PLC0415
    from .semantic.apply import _apply_column  # noqa: PLC0415 — same mapping semantics as clean()

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"apply_plan expects a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(plan, RepairPlan):
        raise TypeError(f"plan must be a RepairPlan, got {type(plan).__name__}")
    if not allow_drift:
        _check_drift(df, plan)

    report = CleanReport(
        rows_before=len(df),
        cols_before=df.shape[1],
        missing_before=int(df.isna().sum().sum()),
    )
    out = df.copy(deep=False)
    guard_snapshot = snapshot_protected(
        out, protected_column_set(plan.config, out.columns, include_legacy=True)
    )

    mappings, executed = _collect_mappings(out, plan, report)
    undo_log = _capture_undo(out, executed, undo_cell_limit) if keep_undo else None
    for column, mapping in mappings.items():
        out[column] = _apply_column(out[column], mapping)
    for action in plan.actions:
        _record_action(report, plan, action, applied=action in executed)

    verify_protected(out, guard_snapshot, report)

    report.rows_after = len(out)
    report.cols_after = out.shape[1]
    report.missing_after = int(out.isna().sum().sum())
    report.decisions_hash = compute_decisions_hash(plan)
    report.undo_log = undo_log
    return out, report
