"""Profile replay: drift gating and config folding for ``profile=``.

A learned profile replays in two layers, mirroring how ``memory=`` works:

1. **Config folding** (here) — learned config deltas (extra sentinels,
   dayfirst) and per-column semantic hints (email/phone/allowed values) are
   folded into the run's options *only where the user did not set them*.
   Explicit user options, ``context=``/``policy=``, and protected columns
   always win: policy lowering runs later in the pipeline and overrides
   per-column hints, and the byte-identity guard is physical.
2. **Proposal backend** — value maps, embedded memory, and (optionally)
   example retrieval flow through
   :class:`~freshdata.semantic.backends.profile.ProfileBackend` and the
   standard policy/confidence gates.

Drift gating happens before either layer: severe schema drift blocks the
whole replay with a report warning; mild drift replays only the columns
that still exist with compatible dtypes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .profile import LearningProfile, load_profile

__all__ = [
    "ProfileReplayGate",
    "annotate_profile_report",
    "check_profile_drift",
    "fold_profile_options",
    "resolve_profile",
]

#: Below this column overlap the profile is considered severely drifted and
#: replay is blocked entirely (same spirit as CleaningMemory's 0.7 gate, but
#: profiles degrade gracefully in between).
_SEVERE_OVERLAP = 0.4
_MILD_OVERLAP = 0.999


@dataclass
class ProfileReplayGate:
    """Outcome of the drift check for one frame/profile pair."""

    ok: bool
    severity: str  # "none" | "mild" | "severe"
    overlap: float
    reasons: list[str] = field(default_factory=list)
    compatible_columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "severity": self.severity,
            "overlap": round(self.overlap, 4),
            "reasons": list(self.reasons),
            "compatible_columns": list(self.compatible_columns),
        }


def resolve_profile(profile: object) -> LearningProfile:
    """Accept a LearningProfile or a path to a ``.fdprofile``."""
    if isinstance(profile, LearningProfile):
        return profile
    if isinstance(profile, (str, Path)):
        return load_profile(profile)
    raise TypeError(
        "profile= must be a LearningProfile or a path to a .fdprofile "
        f"(got {type(profile).__name__})"
    )


def _profile_schema(profile: LearningProfile) -> dict[str, str]:
    """Source schema the profile was learned from (audit first, memory next)."""
    if profile.audit_info is not None:
        schema = profile.audit_info.alignment.get("source_schema")
        if isinstance(schema, Mapping) and schema:
            return {str(c): str(t) for c, t in schema.items()}
    if profile.memory is not None and isinstance(profile.memory.signature, Mapping):
        columns = profile.memory.signature.get("columns")
        if isinstance(columns, Mapping) and columns:
            return {str(c): str(t) for c, t in columns.items()}
    return {}


def _referenced_columns(profile: LearningProfile) -> set[str]:
    columns = {r.column for r in profile.rules if r.column}
    columns.update(profile.value_maps)
    return columns


def check_profile_drift(df: pd.DataFrame, profile: LearningProfile) -> ProfileReplayGate:
    """Judge how far ``df`` has drifted from the profile's training schema."""
    schema = _profile_schema(profile)
    referenced = _referenced_columns(profile)
    baseline = set(schema) or referenced
    if not baseline:
        return ProfileReplayGate(
            ok=True,
            severity="none",
            overlap=1.0,
            compatible_columns=tuple(str(c) for c in df.columns),
        )

    present = {str(c) for c in df.columns}
    shared = baseline & present
    overlap = len(shared) / len(baseline)
    reasons: list[str] = []
    missing = sorted(baseline - present)
    if missing:
        reasons.append(
            f"{len(missing)} learned column(s) missing from the frame: "
            + ", ".join(missing[:5])
            + ("…" if len(missing) > 5 else "")
        )

    dtype_incompatible: list[str] = []
    for column in sorted(shared):
        learned = schema.get(column)
        if learned is None:
            continue
        actual = str(df[column].dtype)
        if learned != actual and (learned == "object") != (actual == "object"):
            dtype_incompatible.append(f"{column} ({learned} -> {actual})")
    if dtype_incompatible:
        reasons.append("dtype changed for: " + ", ".join(dtype_incompatible[:5]))

    compatible = tuple(
        c for c in sorted(shared) if not any(x.startswith(f"{c} ") for x in dtype_incompatible)
    )

    if overlap < _SEVERE_OVERLAP:
        reasons.insert(
            0,
            f"severe schema drift: only {overlap:.0%} of learned columns present; "
            "profile replay blocked",
        )
        return ProfileReplayGate(
            ok=False,
            severity="severe",
            overlap=overlap,
            reasons=reasons,
            compatible_columns=(),
        )
    severity = "none" if overlap >= _MILD_OVERLAP and not dtype_incompatible else "mild"
    if severity == "mild":
        reasons.append("mild drift: replaying only columns still present and compatible")
    return ProfileReplayGate(
        ok=True,
        severity=severity,
        overlap=overlap,
        reasons=reasons,
        compatible_columns=compatible,
    )


# ---------------------------------------------------------------------------
# Option folding (user options always win)
# ---------------------------------------------------------------------------


def _learned_deltas(
    profile: LearningProfile, columns: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """(config deltas, per-column semantic hints) limited to ``columns``."""
    allowed = set(columns)
    config_deltas: dict[str, Any] = {}
    hints: dict[str, dict[str, Any]] = {}
    for rule in profile.rules:
        if rule.enforcement == "advisory":
            continue
        params = rule.params
        if rule.rule == "config_delta":
            option = str(params.get("option", ""))
            if option:
                config_deltas[option] = params.get("value")
            continue
        if rule.column is None or rule.column not in allowed:
            continue
        hint = hints.setdefault(rule.column, {})
        if rule.rule == "valid_format" and params.get("format") == "email":
            hint.setdefault("semantic_type", "email")
        elif rule.rule == "locale_format" and params.get("format") == "phone":
            hint.setdefault("semantic_type", "phone")
            if params.get("region"):
                hint.setdefault("region", str(params["region"]))
        elif rule.rule == "valid_format" and params.get("format") == "date":
            if params.get("dayfirst"):
                hint.setdefault("dayfirst", True)
        elif rule.rule == "allowed_values" and params.get("values"):
            hint.setdefault("allowed_values", [str(v) for v in params["values"]])
    return config_deltas, {c: h for c, h in hints.items() if h}


def fold_profile_options(
    profile: LearningProfile,
    options: dict[str, Any],
    gate: ProfileReplayGate,
) -> dict[str, Any]:
    """Fold learned config deltas and hints into user options (gaps only).

    Never overrides a key the caller provided; ``extra_sentinels`` is merged
    additively.  Returns the (new) options dict.
    """
    if not gate.ok:
        return options
    config_deltas, hints = _learned_deltas(profile, gate.compatible_columns)

    out = dict(options)
    sentinels = config_deltas.pop("extra_sentinels", None)
    if sentinels:
        user = out.get("extra_sentinels") or ()
        merged = tuple(dict.fromkeys((*tuple(user), *tuple(sentinels))))
        out["extra_sentinels"] = merged
    for option, value in config_deltas.items():
        out.setdefault(option, value)

    if hints:
        semantic_context = dict(out.get("semantic_context") or {})
        columns = dict(semantic_context.get("columns") or {})
        for column, hint in hints.items():
            existing = dict(columns.get(column) or {})
            for key, value in hint.items():
                existing.setdefault(key, value)
            columns[column] = existing
        semantic_context["columns"] = columns
        out["semantic_context"] = semantic_context
    return out


def annotate_profile_report(
    report: Any,
    profile: LearningProfile,
    gate: ProfileReplayGate,
) -> None:
    """Record the replay decision on the report (warnings + metadata).

    Idempotent per profile: ``fd.clean`` and ``Cleaner.clean`` may both see
    the same profile on one run, so a second call for the same profile_id is
    a no-op instead of duplicating warnings.
    """
    if report is None:
        return
    existing = getattr(report, "profile_replay", None)
    if isinstance(existing, Mapping) and existing.get("profile_id") == profile.profile_id:
        return
    if not gate.ok:
        report.add_warning(
            f"Learned profile {profile.profile_id} not replayed: {gate.reasons[0]}"
            if gate.reasons
            else f"Learned profile {profile.profile_id} not replayed (severe drift)"
        )
    elif gate.severity == "mild":
        report.add_warning(
            f"Learned profile {profile.profile_id} partially replayed "
            f"(mild drift, {gate.overlap:.0%} column overlap)"
        )
    if profile.manifest.contains_raw_values:
        report.add_warning(
            f"Learned profile {profile.profile_id} contains raw sensitive values "
            "(learned with privacy='none', include_sensitive=True)"
        )
    report.profile_replay = {
        "profile_id": profile.profile_id,
        "privacy_mode": profile.manifest.privacy_mode,
        **gate.to_dict(),
    }
