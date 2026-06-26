"""Configuration for streaming / micro-batch cleaning.

:class:`StreamingCleanConfig` carries the *streaming-execution* knobs (window size,
warmup length, bounded-state caps, trust gate, drift thresholds). It is kept separate
from :class:`~freshdata.CleanConfig`, which still owns every *cleaning decision*
(strategy, thresholds, role gates) — exactly the same split the out-of-core
:class:`~freshdata.EngineConfig` uses. A :class:`StreamingCleaner` holds one of each.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamingCleanConfig:
    """How a :class:`~freshdata.StreamingCleaner` executes across batches.

    Parameters
    ----------
    window_size:
        Size of the recent-window used for rolling statistics and the rolling
        trust score. The *caller* controls how big each batch is; this only
        bounds how much recent history influences "recent-window" reporting.
    warmup_batches:
        Number of leading batches during which the cleaner only repairs
        representation and *collects* statistics — it defers statistical
        imputation until enough global state exists (every deferral is audited).
    max_categories:
        Capacity of the per-column bounded top-k counter (Space-Saving). Caps
        categorical state memory; high-cardinality columns saturate gracefully.
    quantile_reservoir_size:
        Capacity of the per-column reservoir used for approximate medians /
        quantiles. Larger is more accurate but uses more (still bounded) memory.
    rolling_trust_window:
        Number of recent batches averaged into the rolling trust score.
    fail_under_trust:
        If set, a batch whose trust score is below this value is marked failed
        (``StreamingBatchResult.passed_gate is False``); the CLI exits non-zero.
    global_duplicates:
        When False (default), duplicate removal is scoped **within each batch**
        only — cross-batch duplicate detection would require unbounded state.
    drift_missing_jump:
        Absolute jump in a column's missing ratio (batch vs. running) that flags
        missing-rate drift.
    drift_cardinality_factor:
        A batch's distinct-count exceeding ``factor ×`` the running mean distinct
        count flags a cardinality explosion.
    drift_zscore:
        |batch mean − running mean| / running std above this flags a numeric
        distribution shift.
    seed:
        Seed for reservoir sampling, for reproducible approximate quantiles.
    """

    window_size: int = 100_000
    warmup_batches: int = 3
    max_categories: int = 256
    quantile_reservoir_size: int = 20_000
    rolling_trust_window: int = 20
    fail_under_trust: float | None = None
    global_duplicates: bool = False
    drift_missing_jump: float = 0.25
    drift_cardinality_factor: float = 5.0
    drift_zscore: float = 4.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("window_size", "max_categories", "quantile_reservoir_size",
                     "rolling_trust_window"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.warmup_batches < 0:
            raise ValueError(f"warmup_batches must be >= 0, got {self.warmup_batches!r}")
        if self.fail_under_trust is not None and not 0.0 <= self.fail_under_trust <= 100.0:
            raise ValueError(
                f"fail_under_trust must be in [0, 100], got {self.fail_under_trust!r}"
            )
        for name in ("drift_missing_jump", "drift_cardinality_factor", "drift_zscore"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
