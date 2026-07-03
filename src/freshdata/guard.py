"""Physical protected-column enforcement (the Phase-2 hard safety gate).

Policy-level protection (the semantic gate, engine role gates, ``lower()``
folding ``protected`` constraints into ``preserve_columns``) decides what
*should* happen. This module verifies what *did* happen: before a cleaned
frame is returned, every hard-protected column must be **byte-identical** to
its snapshot — same dtype, same values on every surviving row. A mismatch is
a bug somewhere in the executor, and it raises :class:`ProtectedColumnError`
instead of silently returning corrupted data. The guard is deliberately
independent of any policy logic so it also catches bugs introduced by future
backends or steps.

Hard-protected columns are the *context-protected* ones:

- columns under a ``protected`` constraint in a compiled
  :class:`~freshdata.ContextPolicy`;
- columns marked ``mutable=False`` in ``semantic_context["columns"]``.

Legacy ``preserve_columns`` keeps its long-standing meaning ("never dropped
by the engine" — representation repair still applies) unless a column is
*also* context-protected. :func:`protected_column_set` can additionally fold
in ``preserve_columns`` / ``target_column`` / ``id_columns`` for callers such
as :func:`freshdata.apply_plan`, where nothing is ever allowed to write to
them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from .config import CleanConfig


class ProtectedColumnError(Exception):
    """A hard-protected column was modified (or dropped) by the executor.

    Carries ``column`` (the offending column) and ``detail`` (what differed)
    so reports and callers can explain exactly which guarantee was violated.
    """

    def __init__(self, column: str, detail: str) -> None:
        super().__init__(
            f"protected column {column!r} was modified during cleaning: {detail}. "
            "This is a freshdata executor bug — the original input frame is "
            "unchanged; please report this."
        )
        self.column = column
        self.detail = detail


def _snake_ref(name: str) -> str:
    # Local import: keeps ``freshdata.guard`` importable without the context
    # package in pathological partial installs, and avoids import cycles.
    from .context.normalize import snake_ref  # noqa: PLC0415

    return snake_ref(name)


def _declared_hard_protected(config: CleanConfig) -> list[str]:
    """Column names declared immutable by context/policy, as written."""
    names: list[str] = []
    semantic = config.semantic_context
    if isinstance(semantic, Mapping):
        columns = semantic.get("columns")
        if isinstance(columns, Mapping):
            for name, meta in columns.items():
                if isinstance(meta, Mapping) and meta.get("mutable") is False:
                    names.append(str(name))
    policy = config.policy
    if policy is not None:
        for name in getattr(policy, "protected_columns", ()):
            if name not in names:
                names.append(str(name))
    return names


def _match_columns(declared: Iterable[str], actual: Iterable[object]) -> tuple[str, ...]:
    """Resolve declared names against actual frame columns (exact or snake)."""
    actual_names = [str(c) for c in actual]
    by_snake: dict[str, str] = {}
    for name in actual_names:
        by_snake.setdefault(_snake_ref(name), name)
    matched: list[str] = []
    for name in declared:
        resolved = name if name in actual_names else by_snake.get(_snake_ref(name), name)
        if resolved not in matched:
            matched.append(resolved)
    return tuple(matched)


def hard_protected_columns(
    config: CleanConfig, columns: Iterable[object]
) -> tuple[str, ...]:
    """Context-protected column names resolved against *columns*.

    Cheap no-op (returns ``()``) when no context/policy protection exists, so
    every pipeline step can call it unconditionally.
    """
    declared = _declared_hard_protected(config)
    if not declared:
        return ()
    return _match_columns(declared, columns)


def protected_column_set(
    config: CleanConfig,
    columns: Iterable[object],
    *,
    include_legacy: bool = False,
) -> tuple[str, ...]:
    """All columns the guard should verify for *columns*.

    With ``include_legacy=True`` (used by :func:`freshdata.apply_plan`, where
    no step may ever write to them), ``preserve_columns``, ``target_column``,
    and ``id_columns`` are verified too, on top of the context-protected set.
    """
    declared = _declared_hard_protected(config)
    if include_legacy:
        declared.extend(str(c) for c in config.preserve_columns)
        if config.target_column is not None:
            declared.append(str(config.target_column))
        declared.extend(str(c) for c in config.id_columns)
    if not declared:
        return ()
    # De-duplicate, preserving order.
    seen: list[str] = []
    for name in declared:
        if name not in seen:
            seen.append(name)
    return _match_columns(seen, columns)


def snapshot_protected(
    df: pd.DataFrame, protected: Iterable[str]
) -> dict[str, pd.Series]:
    """Deep-copy each protected column present in *df* for later verification."""
    snapshot: dict[str, pd.Series] = {}
    for name in protected:
        if name in df.columns:
            snapshot[name] = df[name].copy(deep=True)
    return snapshot


def _series_identical(before: pd.Series, after: pd.Series) -> str | None:
    """Return a human explanation of the first difference, or ``None`` if none."""
    if str(before.dtype) != str(after.dtype):
        return f"dtype changed from {before.dtype} to {after.dtype}"
    if len(after) > len(before):
        return f"gained rows ({len(before)} -> {len(after)})"
    if len(after) < len(before):
        # Row-level steps (dedupe, empty-row drops) legitimately remove rows;
        # surviving rows must still hold their original values, aligned by
        # index label.
        try:
            before = before.loc[after.index]
        except KeyError:
            return "surviving row index labels do not match the original frame"
    if before.equals(after):
        return None
    try:
        differing = before.compare(after)
        n = len(differing)
        first = differing.index[0] if n else "?"
        return f"{n} cell(s) differ (first at row {first!r})"
    except (ValueError, TypeError):
        return "cell values differ"


def verify_protected(
    df: pd.DataFrame,
    snapshot: Mapping[str, pd.Series],
    report: object | None = None,
) -> None:
    """Assert every snapshotted column survived byte-identically in *df*.

    Raises :class:`ProtectedColumnError` on the first violation. With a
    *report*, records one auditable ``guard`` action stating the guarantee
    that was checked (and, on violation, which column broke it before the
    raise).
    """
    for name, before in snapshot.items():
        if name not in df.columns:
            _record_violation(report, name, "column was dropped")
            raise ProtectedColumnError(name, "column was dropped")
        detail = _series_identical(before, df[name])
        if detail is not None:
            _record_violation(report, name, detail)
            raise ProtectedColumnError(name, detail)
    if snapshot and report is not None and hasattr(report, "add"):
        report.add(
            "guard",
            f"verified {len(snapshot)} protected column(s) byte-identical",
            count=0,
            rationale="hard protected-column guarantee (context policy / mutable=False)",
            metadata={"protected_columns": sorted(snapshot)},
        )


def _record_violation(report: object | None, column: str, detail: str) -> None:
    if report is not None and hasattr(report, "add"):
        report.add(
            "guard",
            f"protected column {column!r} violation: {detail}",
            column=column,
            count=0,
            risk="high",
            rationale="hard protected-column guarantee failed",
            status="skipped",
            human_review=True,
            metadata={"protected_column_violation": {"column": column, "detail": detail}},
        )
