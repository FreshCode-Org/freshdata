"""Bounded, memory-safe running-statistic primitives for streaming cleaning.

Every accumulator here uses **O(1)** or **O(capacity)** memory — never O(rows) — so
feeding 100M rows through them costs the same memory as feeding 100k. They are the
building blocks the :class:`~freshdata.streaming.StreamingState` keeps per column:

- :class:`Welford` — running mean/variance (Chan's parallel batch update).
- :class:`ReservoirSampler` — bounded uniform sample for approximate quantiles/median.
- :class:`BoundedCounter` — Space-Saving top-k for approximate mode and frequent values.

All updates are vectorized over a NumPy array / pandas Series so they stay fast on
large batches; none of them retain the batch after the update returns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Welford:
    """Running count/mean/variance via Chan's parallel (batched) algorithm.

    Numerically stable and exact for the mean; variance is the sample variance.
    Memory is three floats regardless of how many values are seen.
    """

    __slots__ = ("count", "mean", "_m2")

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0  # sum of squared deviations from the mean

    def update(self, values: np.ndarray) -> None:
        """Fold a batch of numeric values (NaNs already removed) into the stats."""
        n_b = values.size
        if n_b == 0:
            return
        mean_b = float(values.mean())
        # M2 of the batch = sum((x - mean_b)^2); var(ddof=0) * n is exact and fast.
        m2_b = float(values.var()) * n_b
        if self.count == 0:
            self.count, self.mean, self._m2 = n_b, mean_b, m2_b
            return
        n_a = self.count
        delta = mean_b - self.mean
        total = n_a + n_b
        self.mean += delta * n_b / total
        self._m2 += m2_b + delta * delta * n_a * n_b / total
        self.count = total

    @property
    def variance(self) -> float:
        """Sample variance, or 0.0 with fewer than two observations."""
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def to_dict(self) -> dict[str, float]:
        return {"count": self.count, "mean": self.mean, "std": self.std}


class ReservoirSampler:
    """Vitter Algorithm R reservoir: a bounded uniform sample of the stream.

    Holds at most ``capacity`` values; each value seen so far has an equal
    probability of being in the reservoir, so ``quantile`` gives an unbiased
    (approximate) estimate of the stream's quantiles without storing every row.
    The batch update is fully vectorized — no per-row Python loop.
    """

    __slots__ = ("capacity", "_buf", "_size", "n_seen", "_rng")

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError(f"reservoir capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._buf = np.empty(capacity, dtype="float64")
        self._size = 0
        self.n_seen = 0
        self._rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        """Fold a batch of numeric values (NaNs already removed) into the sample."""
        m = values.size
        if m == 0:
            return
        filled = 0
        if self._size < self.capacity:
            take = min(self.capacity - self._size, m)
            self._buf[self._size : self._size + take] = values[:take]
            self._size += take
            filled = take
        rest = values[filled:]
        if rest.size:
            # Global 1-indexed position of each remaining item.
            t = self.n_seen + filled + 1 + np.arange(rest.size)
            j = np.floor(self._rng.random(rest.size) * t).astype(np.int64)
            keep = j < self.capacity
            # Fancy assignment keeps the last write per slot, matching the
            # sequential semantics of Algorithm R (later items overwrite earlier).
            self._buf[j[keep]] = rest[keep]
        self.n_seen += m

    @property
    def size(self) -> int:
        return self._size

    def quantile(self, q: float) -> float | None:
        """Approximate quantile *q* in [0, 1], or ``None`` if nothing sampled."""
        if self._size == 0:
            return None
        return float(np.quantile(self._buf[: self._size], q))

    def median(self) -> float | None:
        return self.quantile(0.5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self._size,
            "n_seen": self.n_seen,
            "median_approx": self.median(),
        }


class BoundedCounter:
    """Space-Saving summary: approximate top-k frequencies in bounded memory.

    Tracks at most ``capacity`` keys. When a new key arrives and the summary is
    full, it evicts the current least-frequent key and inherits its count (the
    Space-Saving guarantee), so the heaviest hitters — what we need for mode
    imputation — are tracked accurately while memory stays bounded. Counts for
    evicted-then-readmitted keys may be slight overestimates; this is recorded
    via :attr:`saturated`.
    """

    __slots__ = ("capacity", "counts", "total", "saturated")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"counter capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.counts: dict[Any, int] = {}
        self.total: int = 0
        self.saturated: bool = False

    def update_from_series(self, s: pd.Series) -> None:
        """Fold a column's non-null value counts into the summary."""
        try:
            vc = s.value_counts(dropna=True)
        except TypeError:  # unhashable values — nothing countable
            return
        for key, cnt in vc.items():
            self._add(key, int(cnt))

    def _add(self, key: Any, cnt: int) -> None:
        self.total += cnt
        if key in self.counts:
            self.counts[key] += cnt
        elif len(self.counts) < self.capacity:
            self.counts[key] = cnt
        else:
            self.saturated = True
            min_key = min(self.counts, key=self.counts.__getitem__)
            min_val = self.counts.pop(min_key)
            self.counts[key] = min_val + cnt

    def top(self, k: int = 1) -> list[tuple[Any, int]]:
        """The *k* most frequent tracked keys, highest first."""
        return sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def mode(self) -> Any | None:
        """The single most frequent tracked value, or ``None`` if empty."""
        top = self.top(1)
        return top[0][0] if top else None

    def mode_ratio(self) -> float | None:
        """Share of counted values held by the mode (approximate)."""
        if self.total == 0:
            return None
        top = self.top(1)
        return (top[0][1] / self.total) if top else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracked_keys": len(self.counts),
            "total": self.total,
            "saturated": self.saturated,
            "mode": self.mode(),
            "mode_ratio": self.mode_ratio(),
            "top": [[k, v] for k, v in self.top(5)],
        }
