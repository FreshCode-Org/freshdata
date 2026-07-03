"""Semantic backend protocol and shared budget accounting.

A backend is a *proposal generator*: it inspects the frame through the
distinct-values discipline and emits :class:`SemanticProposal`s. Backends never
apply repairs and cannot bypass the policy gate or the protected-column guard —
every proposal still flows through :func:`freshdata.semantic.policy.decide`
and the executor's byte-identity check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from ...report import CleanReport
    from ..types import SemanticContext, SemanticProposal


class BackendUnavailable(Exception):
    """Raised by ``warm_up`` when a backend cannot run; carries the reason."""


@dataclass
class Budget:
    """Resource ceilings for model-assisted backends, with usage counters.

    Deterministic and memory backends are never metered — the budget only
    bounds optional model inference. ``exhausted_reason`` becomes non-empty the
    first time any ceiling is hit; backends must then stop cleanly (already
    complete proposals stand, nothing is partially applied).
    """

    max_columns: int | None = None
    max_distinct_values: int | None = None
    max_model_calls: int | None = None
    max_seconds: float | None = None

    columns_used: int = 0
    distinct_values_used: int = 0
    model_calls_used: int = 0
    started_at: float = field(default_factory=time.monotonic)
    exhausted_reason: str = ""

    @classmethod
    def from_config(cls, raw: object) -> Budget:
        """Build a budget from the ``semantic_budget`` config dict (or None)."""
        if not isinstance(raw, dict):
            return cls()
        def _num(key: str) -> float | None:
            value = raw.get(key)
            return float(value) if isinstance(value, (int, float)) and value >= 0 else None

        max_cols = _num("max_columns")
        max_distinct = _num("max_distinct_values")
        max_calls = _num("max_model_calls")
        return cls(
            max_columns=int(max_cols) if max_cols is not None else None,
            max_distinct_values=int(max_distinct) if max_distinct is not None else None,
            max_model_calls=int(max_calls) if max_calls is not None else None,
            max_seconds=_num("max_seconds"),
        )

    def _check_time(self) -> None:
        if (
            not self.exhausted_reason
            and self.max_seconds is not None
            and time.monotonic() - self.started_at > self.max_seconds
        ):
            self.exhausted_reason = f"max_seconds={self.max_seconds} exceeded"

    @property
    def exhausted(self) -> bool:
        self._check_time()
        return bool(self.exhausted_reason)

    def try_column(self, n_distinct: int) -> bool:
        """Reserve budget for one column of ``n_distinct`` values, or refuse."""
        if self.exhausted:
            return False
        if self.max_columns is not None and self.columns_used + 1 > self.max_columns:
            self.exhausted_reason = f"max_columns={self.max_columns} exceeded"
            return False
        if (
            self.max_distinct_values is not None
            and self.distinct_values_used + n_distinct > self.max_distinct_values
        ):
            self.exhausted_reason = f"max_distinct_values={self.max_distinct_values} exceeded"
            return False
        self.columns_used += 1
        self.distinct_values_used += n_distinct
        return True

    def try_model_call(self) -> bool:
        """Reserve budget for one model invocation, or refuse."""
        if self.exhausted:
            return False
        if self.max_model_calls is not None and self.model_calls_used + 1 > self.max_model_calls:
            self.exhausted_reason = f"max_model_calls={self.max_model_calls} exceeded"
            return False
        self.model_calls_used += 1
        return True


@runtime_checkable
class SemanticBackend(Protocol):
    """Contract every named entry of ``semantic_backends`` satisfies."""

    name: str

    def warm_up(self) -> None:
        """Load lazy resources; raise :class:`BackendUnavailable` to self-disable."""
        ...

    def propose(
        self, df: pd.DataFrame, ctx: SemanticContext, budget: Budget
    ) -> list[SemanticProposal]:
        """Generate proposals from distinct values only; never mutate ``df``."""
        ...


def record_backend_skip(report: CleanReport | None, backend: str, reason: str) -> None:
    """Note a self-disabled backend in the report (fallback event + warning)."""
    if report is None:
        return
    report.record_fallback(backend=backend, step="semantic", reason=reason)
    report.add_warning(f"Semantic backend '{backend}' skipped: {reason}")
