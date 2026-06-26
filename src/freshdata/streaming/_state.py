"""Cross-batch state: bounded per-column running statistics + stream metadata.

:class:`StreamingState` is the memory of a :class:`~freshdata.StreamingCleaner`. It holds
one :class:`ColumnState` per column (each backed by the bounded accumulators in
:mod:`._stats`) plus stream-level bookkeeping: batch count, rows seen, the schema
baseline locked from the first batch, the trust-score history, and the drift log.

Crucially, **nothing here grows with the number of rows**: numeric columns keep a
fixed-size reservoir + a Welford triple + min/max; categorical columns keep a bounded
Space-Saving counter; the trust history is a capped deque. That is what makes 100M-row
streaming run in constant memory.
"""

from __future__ import annotations

import contextlib
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from ._stats import BoundedCounter, ReservoirSampler, Welford

#: Cap on the retained trust-score history (keeps the state bounded on long streams).
_TRUST_HISTORY_CAP = 10_000


@dataclass
class NumericSnapshot:
    """A read-only view of a numeric column's global running statistics."""

    count: int
    mean: float
    std: float
    median: float | None
    q1: float | None
    q3: float | None
    minimum: float
    maximum: float

    @property
    def skew_approx(self) -> float | None:
        """Pearson's second skewness coefficient: ``3·(mean − median)/std``.

        A bounded-memory stand-in for the engine's sample skewness, used to pick
        mean vs. median imputation the same way the in-memory pipeline does.
        """
        if self.median is None or self.std == 0:
            return None
        return 3.0 * (self.mean - self.median) / self.std

    @property
    def has_outliers(self) -> bool:
        """Whether the tracked min/max fall outside 1.5×IQR Tukey fences."""
        if self.q1 is None or self.q3 is None:
            return False
        iqr = self.q3 - self.q1
        if iqr <= 0:
            return False
        return self.minimum < self.q1 - 1.5 * iqr or self.maximum > self.q3 + 1.5 * iqr


class ColumnState:
    """Running, bounded statistics for one column across all batches seen."""

    def __init__(self, name: str, role: str, baseline_dtype: str, first_seen_batch: int,
                 *, reservoir_size: int, max_categories: int, seed: int) -> None:
        self.name = name
        self.role = role
        self.baseline_dtype = baseline_dtype
        self.first_seen_batch = first_seen_batch
        self.non_null = 0
        self.missing = 0
        self.dtype_counts: Counter[str] = Counter()
        # Per-batch distinct-count tracker → cardinality-explosion drift signal.
        self.nunique_stat = Welford()
        # Role-appropriate accumulators (created lazily as data is observed).
        self._welford = Welford()
        self._reservoir = ReservoirSampler(reservoir_size, seed=seed)
        self._minimum: float = np.inf
        self._maximum: float = -np.inf
        self._counter = BoundedCounter(max_categories)
        self._dt_min: pd.Timestamp | None = None
        self._dt_max: pd.Timestamp | None = None
        self._dt_prev_max: pd.Timestamp | None = None
        self.datetime_ordered = True  # until a batch proves otherwise

    @property
    def seen(self) -> int:
        return self.non_null + self.missing

    @property
    def missing_ratio(self) -> float:
        return (self.missing / self.seen) if self.seen else 0.0

    def update(self, s: pd.Series) -> None:
        """Fold one batch's column into the running statistics."""
        n = len(s)
        n_missing = int(s.isna().sum())
        self.missing += n_missing
        self.non_null += n - n_missing
        self.dtype_counts[str(s.dtype)] += 1
        with contextlib.suppress(TypeError):
            self.nunique_stat.update(np.array([float(s.nunique(dropna=True))]))

        if is_numeric_dtype(s) and not is_bool_dtype(s):
            values = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
            values = values[~np.isnan(values)]
            if values.size:
                self._welford.update(values)
                self._reservoir.update(values)
                self._minimum = min(self._minimum, float(values.min()))
                self._maximum = max(self._maximum, float(values.max()))
        elif is_datetime64_any_dtype(s):
            self._update_datetime(s)
        else:
            self._counter.update_from_series(s)

    def _update_datetime(self, s: pd.Series) -> None:
        nonnull = s.dropna()
        if nonnull.empty:
            return
        batch_min, batch_max = nonnull.min(), nonnull.max()
        self._dt_min = batch_min if self._dt_min is None else min(self._dt_min, batch_min)
        self._dt_max = batch_max if self._dt_max is None else max(self._dt_max, batch_max)
        # Globally ordered iff each batch starts no earlier than the previous ended
        # and is itself internally non-decreasing.
        if self._dt_prev_max is not None and batch_min < self._dt_prev_max:
            self.datetime_ordered = False
        if not nonnull.is_monotonic_increasing:
            self.datetime_ordered = False
        self._dt_prev_max = batch_max

    # -- decision-time snapshots ------------------------------------------------

    def numeric_snapshot(self) -> NumericSnapshot:
        return NumericSnapshot(
            count=self._welford.count,
            mean=self._welford.mean,
            std=self._welford.std,
            median=self._reservoir.median(),
            q1=self._reservoir.quantile(0.25),
            q3=self._reservoir.quantile(0.75),
            minimum=self._minimum,
            maximum=self._maximum,
        )

    def mode(self) -> Any | None:
        return self._counter.mode()

    def mode_ratio(self) -> float | None:
        return self._counter.mode_ratio()

    @property
    def batch_nunique_mean(self) -> float:
        return self.nunique_stat.mean

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "baseline_dtype": self.baseline_dtype,
            "dtypes_seen": dict(self.dtype_counts),
            "non_null": self.non_null,
            "missing": self.missing,
            "missing_ratio": round(self.missing_ratio, 6),
            "first_seen_batch": self.first_seen_batch,
        }
        if self._welford.count:
            snap = self.numeric_snapshot()
            payload["numeric"] = {
                "count": snap.count,
                "mean": snap.mean,
                "std": snap.std,
                "median_approx": snap.median,
                "min": snap.minimum,
                "max": snap.maximum,
            }
        if self._counter.total:
            payload["categorical"] = self._counter.to_dict()
        if self._dt_min is not None:
            payload["datetime"] = {
                "min": str(self._dt_min),
                "max": str(self._dt_max),
                "ordered": self.datetime_ordered,
            }
        return payload


@dataclass
class StreamingState:
    """The full memory of a streaming clean: per-column stats + stream metadata."""

    reservoir_size: int = 20_000
    max_categories: int = 256
    rolling_trust_window: int = 20
    seed: int = 0

    batch_count: int = 0
    rows_seen: int = 0
    schema_baseline: list[str] = field(default_factory=list)
    baseline_dtypes: dict[str, str] = field(default_factory=dict)
    columns: dict[str, ColumnState] = field(default_factory=dict)
    drift_log: list[dict[str, Any]] = field(default_factory=list)

    _trust_history: deque = field(default_factory=lambda: deque(maxlen=_TRUST_HISTORY_CAP))
    _rolling: deque = field(default_factory=deque)
    _trust_weighted_sum: float = 0.0
    _trust_weight: int = 0

    def observe_batch(self, df: pd.DataFrame, *, roles: dict[str, str]) -> None:
        """Update rows/columns/state from one already-repaired batch."""
        self.batch_count += 1
        self.rows_seen += len(df)
        if not self.schema_baseline:
            self.schema_baseline = [str(c) for c in df.columns]
            self.baseline_dtypes = {str(c): str(df[c].dtype) for c in df.columns}
        for col in df.columns:
            name = str(col)
            state = self.columns.get(name)
            if state is None:
                state = ColumnState(
                    name, roles.get(name, "categorical"),
                    str(df[col].dtype), self.batch_count,
                    reservoir_size=self.reservoir_size,
                    max_categories=self.max_categories, seed=self.seed,
                )
                self.columns[name] = state
            state.update(df[col])

    def record_trust(self, overall: float, rows: int) -> tuple[float, float]:
        """Record a batch trust score; return ``(rolling, cumulative)``."""
        self._trust_history.append(overall)
        self._rolling.append(overall)
        while len(self._rolling) > self.rolling_trust_window:
            self._rolling.popleft()
        self._trust_weighted_sum += overall * rows
        self._trust_weight += rows
        rolling = float(np.mean(self._rolling)) if self._rolling else overall
        cumulative = (
            self._trust_weighted_sum / self._trust_weight if self._trust_weight else overall
        )
        return rolling, cumulative

    @property
    def rolling_trust_score(self) -> float | None:
        return float(np.mean(self._rolling)) if self._rolling else None

    @property
    def cumulative_trust_score(self) -> float | None:
        if not self._trust_weight:
            return None
        return self._trust_weighted_sum / self._trust_weight

    @property
    def trust_history(self) -> list[float]:
        return list(self._trust_history)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_count": self.batch_count,
            "rows_seen": self.rows_seen,
            "schema_baseline": list(self.schema_baseline),
            "baseline_dtypes": dict(self.baseline_dtypes),
            "rolling_trust_score": self.rolling_trust_score,
            "cumulative_trust_score": self.cumulative_trust_score,
            "n_drift_events": len(self.drift_log),
            "columns": {name: state.to_dict() for name, state in self.columns.items()},
        }
