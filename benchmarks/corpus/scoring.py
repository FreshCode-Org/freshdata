"""Run the trap corpus against FreshData and score the result.

This is the measurement half of the corpus.  It answers the question the whole
programme turns on: *how often does the library make a semantic decision a
knowledgeable human would call wrong, and how often does it do so silently?*

Scoring rules
-------------
The observed disposition is derived from what actually happened to the cell,
never from what the report claims:

``PRESERVE``   the value is unchanged (compared after a NaN-aware equality).
``REPAIR``     the value changed.  Whether it changed *correctly* is a separate
               axis -- ``repaired_correctly`` -- so a wrong repair is visible as
               a corruption rather than hidden inside a repair count.
``QUARANTINE`` / ``REJECT`` / ``REVIEW``
               taken from ``validate_fields`` row actions, mapped through
               :func:`~.dispositions.from_field_action`.
``FLAG``       unchanged, but the cell drew an issue or a report warning.

A case whose ``expected`` is ``None`` (a recorded specification gap) is
measured and reported but never scored pass/fail -- asserting an outcome there
would be inventing the contract the gap exists to record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .dispositions import Disposition, satisfies
from .frames import frame_for, trap_position
from .traps import TRAPS, TrapCase

__all__ = ["Observation", "observe", "score", "Metrics", "run_corpus"]

D = Disposition


def _equal(a: Any, b: Any) -> bool:
    """NaN-aware scalar equality; two missing values count as equal."""
    a_missing = a is None or (isinstance(a, float) and pd.isna(a)) or a is pd.NaT
    b_missing = b is None or (isinstance(b, float) and pd.isna(b)) or b is pd.NaT
    try:
        a_missing = a_missing or bool(pd.isna(a))
        b_missing = b_missing or bool(pd.isna(b))
    except (TypeError, ValueError):
        pass
    if a_missing or b_missing:
        return a_missing and b_missing
    if isinstance(a, str) != isinstance(b, str):
        # '7' and 7 are different outcomes for a cleaner; do not conflate them.
        return False
    try:
        return bool(a == b)
    except Exception:
        return False


@dataclass
class Observation:
    """What the library actually did to one trap cell."""

    case: TrapCase
    observed: Disposition | None
    final_value: Any = None
    changed: bool = False
    repaired_correctly: bool | None = None
    field_action: str | None = None
    audited: bool = False
    error: str | None = None

    @property
    def measured(self) -> bool:
        return self.error is None and self.observed is not None


def observe(case: TrapCase, *, semantic: bool = True) -> Observation:
    """Run one case through ``fd.clean`` and ``fd.validate_fields``."""
    import freshdata as fd

    frame = frame_for(case)
    if frame is None:
        return Observation(case, None, error="no filler for role")

    pos = trap_position()
    original = frame[case.role].iloc[pos]
    kwargs: dict[str, Any] = {"verbose": False, "return_report": True}
    if semantic:
        kwargs["semantic_mode"] = "auto"

    try:
        cleaned, report = fd.clean(frame, **kwargs)
    except Exception as exc:  # a crash is itself a finding
        return Observation(case, None, error=f"{type(exc).__name__}: {exc}"[:160])

    # The anchor column keeps the row alive, so position is stable.
    final = cleaned[case.role].iloc[pos] if pos < len(cleaned) else None
    changed = not _equal(original, final)

    audited = any(a.column == case.role for a in report.actions) or bool(report.warnings)

    # Ask the validation surface what it would do with the row.
    action: str | None = None
    try:
        spec: dict[str, Any] = {}
        if case.semantic_type:
            spec[case.role] = case.semantic_type
        vreport = fd.validate_fields(frame, spec or None)
        actions = vreport.row_actions()
        action = actions.get(pos) if isinstance(actions, dict) else None
    except Exception:
        action = None

    from .dispositions import from_field_action

    observed: Disposition
    if action in {"quarantine", "reject", "manual_review"}:
        observed = from_field_action(action)
    elif changed:
        observed = D.REPAIR
    elif audited:
        observed = D.FLAG
    else:
        observed = D.PRESERVE

    repaired_correctly = None
    if case.expected is D.REPAIR:
        repaired_correctly = _equal(final, case.repaired)

    return Observation(
        case=case,
        observed=observed,
        final_value=final,
        changed=changed,
        repaired_correctly=repaired_correctly,
        field_action=action,
        audited=audited,
    )


@dataclass
class Metrics:
    """The Phase 23 metric set, as rates."""

    total: int = 0
    measured: int = 0
    spec_gaps: int = 0
    errors: int = 0
    correct: int = 0
    #: expected PRESERVE and the value survived
    preserved: int = 0
    preserve_cases: int = 0
    #: expected PRESERVE but the value changed -- false-positive corruption
    corrupted: int = 0
    #: expected REPAIR and the gold value was produced
    repaired_correctly: int = 0
    repair_cases: int = 0
    #: expected REPAIR, the value changed, but not to the gold value
    misrepaired: int = 0
    #: expected REPAIR/REVIEW/... but nothing happened at all
    escaped: int = 0
    routed_to_review: int = 0
    review_cases: int = 0
    audited: int = 0
    changes: int = 0
    failures: list[str] = field(default_factory=list)

    def _rate(self, num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "measured": self.measured,
            "spec_gaps": self.spec_gaps,
            "errors": self.errors,
            "accuracy": self._rate(self.correct, self.measured),
            "preservation_rate": self._rate(self.preserved, self.preserve_cases),
            "corruption_rate": self._rate(self.corrupted, self.preserve_cases),
            "repair_precision": self._rate(self.repaired_correctly, self.changes),
            "repair_recall": self._rate(self.repaired_correctly, self.repair_cases),
            "false_positive_rate": self._rate(self.corrupted, self.preserve_cases),
            "false_negative_rate": self._rate(self.escaped, self.measured),
            "review_rate": self._rate(self.routed_to_review, self.review_cases),
            "escape_rate": self._rate(self.escaped, self.measured),
            "audit_completeness": self._rate(self.audited, self.changes),
        }


def score(observations: list[Observation]) -> Metrics:
    """Aggregate observations into the named metrics."""
    m = Metrics(total=len(observations))
    for o in observations:
        if o.error:
            m.errors += 1
            m.failures.append(f"{o.case.family}/{o.case.role}: {o.error}")
            continue
        if o.case.expected is None:
            m.spec_gaps += 1
            continue
        m.measured += 1
        expected = o.case.expected

        if o.changed:
            m.changes += 1
            if o.audited:
                m.audited += 1

        if expected is D.PRESERVE:
            m.preserve_cases += 1
            if o.changed:
                m.corrupted += 1
                m.failures.append(
                    f"CORRUPTION {o.case.family}/{o.case.role}: "
                    f"{o.case.token!r} -> {o.final_value!r}"
                )
            else:
                m.preserved += 1
                m.correct += 1
            continue

        if expected is D.REPAIR:
            m.repair_cases += 1
            if o.repaired_correctly:
                m.repaired_correctly += 1
                m.correct += 1
            elif o.changed:
                m.misrepaired += 1
                m.failures.append(
                    f"MISREPAIR {o.case.family}/{o.case.role}: {o.case.token!r} -> "
                    f"{o.final_value!r}, expected {o.case.repaired!r}"
                )
            else:
                m.escaped += 1
                m.failures.append(
                    f"ESCAPE {o.case.family}/{o.case.role}: {o.case.token!r} unchanged, "
                    f"expected repair to {o.case.repaired!r}"
                )
            continue

        # REVIEW / QUARANTINE / REJECT / FLAG
        m.review_cases += 1
        if satisfies(expected, o.observed):
            m.routed_to_review += 1
            m.correct += 1
        else:
            m.escaped += 1
            m.failures.append(
                f"ESCAPE {o.case.family}/{o.case.role}: {o.case.token!r} got "
                f"{o.observed.value}, expected {expected.value}"
            )
    return m


def run_corpus(cases=TRAPS, *, semantic: bool = True) -> tuple[list[Observation], Metrics]:
    observations = [observe(c, semantic=semantic) for c in cases]
    return observations, score(observations)
