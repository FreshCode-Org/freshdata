"""Development-only instrumentation for semantic context repetition.

The probe wraps the pure value-shape helpers used by
``freshdata.semantic.context``.  It deliberately keeps all observations in
memory and creates a fresh collector for each context build, making the
result useful for estimating the benefit of a per-build memoization layer
without changing production behavior.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import patch

import pandas as pd

from freshdata.config import CleanConfig
from freshdata.semantic import context as semantic_context
from freshdata.semantic.types import SemanticContext

# ``column_name_is_identifier`` is intentionally not included: the repetition
# candidates are the six value parsers plus the email-value regex check.
_OPERATIONS = (
    "is_plain_number",
    "parse_number_words",
    "parse_boolean",
    "parse_currency",
    "parse_unit",
    "email_value",
    "looks_like_date_value",
)
_CALLABLE_OPERATIONS = (
    "is_plain_number",
    "parse_number_words",
    "parse_boolean",
    "parse_currency",
    "parse_unit",
    "looks_like_date_value",
)

# Exact-type gating keeps key construction safe and mirrors the values each
# helper is designed to inspect.  In particular, bool is observable by
# ``is_plain_number`` even though the helper itself excludes it semantically.
_ALLOWED_TYPES: Mapping[str, tuple[type[object], ...]] = {
    "is_plain_number": (int, float, bool, str),
    "parse_number_words": (str,),
    "parse_boolean": (str,),
    "parse_currency": (str,),
    "parse_unit": (str,),
    "looks_like_date_value": (str,),
    "email_value": (str,),
}


@dataclass(frozen=True)
class OperationProbe:
    """Immutable observations for one semantic helper during one build."""

    total_calls: int
    eligible_calls: int
    bypassed_calls: int
    unique_keys: int
    theoretical_hits: int
    eligible_values: tuple[object, ...]

    @property
    def hit_rate(self) -> float:
        return self.theoretical_hits / self.eligible_calls if self.eligible_calls else 0.0


@dataclass(frozen=True)
class SemanticProbeBuild:
    """Per-operation probe results for one ``build_semantic_context`` call."""

    by_operation: Mapping[str, OperationProbe]

    @property
    def total_theoretical_hits(self) -> int:
        return sum(item.theoretical_hits for item in self.by_operation.values())

    @property
    def total_calls(self) -> int:
        return sum(item.total_calls for item in self.by_operation.values())

    @property
    def total_eligible_calls(self) -> int:
        return sum(item.eligible_calls for item in self.by_operation.values())

    @property
    def total_bypassed_calls(self) -> int:
        return sum(item.bypassed_calls for item in self.by_operation.values())


@dataclass
class _MutableOperationProbe:
    total_calls: int = 0
    eligible_calls: int = 0
    bypassed_calls: int = 0
    theoretical_hits: int = 0

    def __post_init__(self) -> None:
        self.seen_keys: set[tuple[str, type[object], object]] = set()
        self.eligible_values: list[object] = []


def _eligible(operation: str, value: object) -> bool:
    """Return whether *value* may safely participate in a repetition key."""

    return type(value) in _ALLOWED_TYPES[operation]


class _ProbeCollector:
    """Mutable collector used only while one context build is running."""

    def __init__(self) -> None:
        self._operations = {name: _MutableOperationProbe() for name in _OPERATIONS}

    def record(self, operation: str, value: object) -> None:
        state = self._operations[operation]
        state.total_calls += 1
        if not _eligible(operation, value):
            state.bypassed_calls += 1
            return

        state.eligible_calls += 1
        state.eligible_values.append(value)
        # Every allowed exact type is hash-safe.  Unsafe values are returned
        # above before this key expression is evaluated.
        key = (operation, type(value), value)
        if key in state.seen_keys:
            state.theoretical_hits += 1
        else:
            state.seen_keys.add(key)

    def finish_build(self) -> SemanticProbeBuild:
        return SemanticProbeBuild(
            by_operation={
                name: OperationProbe(
                    total_calls=state.total_calls,
                    eligible_calls=state.eligible_calls,
                    bypassed_calls=state.bypassed_calls,
                    unique_keys=len(state.seen_keys),
                    theoretical_hits=state.theoretical_hits,
                    eligible_values=tuple(state.eligible_values),
                )
                for name, state in self._operations.items()
            }
        )


class _EmailValueProxy:
    """Regex proxy that records the already-stripped email candidate."""

    def __init__(self, delegate: Any, probe: _ProbeCollector) -> None:
        self._delegate = delegate
        self._probe = probe

    def match(self, value: object, *args: object, **kwargs: object) -> Any:
        self._probe.record("email_value", value)
        return self._delegate.match(value, *args, **kwargs)


@contextmanager
def _patched_context_operations(
    probe: _ProbeCollector | None = None,
) -> Iterator[_ProbeCollector]:
    """Temporarily wrap context helper references with *probe* recorders."""

    if probe is None:
        probe = _ProbeCollector()
    with ExitStack() as stack:
        for operation in _CALLABLE_OPERATIONS:
            original = getattr(semantic_context, operation)

            def wrapped(
                value: object,
                *args: object,
                _operation: str = operation,
                _original: Callable[..., Any] = original,
                **kwargs: object,
            ) -> Any:
                probe.record(_operation, value)
                return _original(value, *args, **kwargs)

            stack.enter_context(patch.object(semantic_context, operation, wrapped))

        stack.enter_context(
            patch.object(
                semantic_context,
                "_EMAIL_VALUE",
                _EmailValueProxy(semantic_context._EMAIL_VALUE, probe),
            )
        )
        yield probe


def probe_context_build(
    df: pd.DataFrame,
    config: CleanConfig,
    *,
    stats: dict[object, tuple[int, int, int | None]] | None = None,
) -> tuple[SemanticContext, SemanticProbeBuild]:
    """Build semantic context and collect helper repetition observations."""

    with _patched_context_operations() as probe:
        context = semantic_context.build_semantic_context(df, config, stats=stats)
    return context, probe.finish_build()


__all__ = [
    "OperationProbe",
    "SemanticProbeBuild",
    "probe_context_build",
]
