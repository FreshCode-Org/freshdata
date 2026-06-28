"""Time-series and streaming-aware cleaning modes layered on the streaming cleaner.

:class:`TimeSeriesCleanConfig` carries the *time-series* knobs (timestamp/entity
columns, short-gap interpolation, seasonal imputation, ordered dedupe, watermark-based
late-data handling, windowed anomaly detection). It is kept separate from
:class:`~freshdata.streaming.StreamingCleanConfig` (which owns streaming *execution*) and
:class:`~freshdata.CleanConfig` (which owns every per-column cleaning *decision*), the
same three-way split the rest of the package uses.

:class:`TimeSeriesProcessor` is the engine that applies those steps to one batch. It
holds the only *unbounded-by-design* piece of cross-batch state a time-series stream
needs — a single per-stream watermark timestamp — so a :class:`StreamingCleaner` can run
it batch-by-batch while still recognising events that arrive late relative to everything
seen so far. Every transformation is audited as a :class:`~freshdata.report.Action` with
one of the ``timeseries_interpolation`` / ``seasonal_imputation`` / ``ordered_dedupe`` /
``late_data`` / ``windowed_anomaly`` step names, so the trust contract is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import CleanConfig
from ..engine.context import infer_role
from ..report import CleanReport

#: Interpolation methods accepted by :class:`TimeSeriesCleanConfig`.
INTERPOLATION_METHODS = ("linear", "time", "ffill", "bfill")
#: Tie-break strategies for ordered dedupe.
DEDUPE_KEEP = ("first", "last", "latest_event_time", "highest_quality")
#: What to do with events that arrive past the watermark.
LATE_DATA_ACTIONS = ("quarantine", "keep_with_warning", "drop")
#: Windowed anomaly detectors.
ANOMALY_METHODS = ("rolling_zscore", "mad", "iqr", "ewma")
#: What to do with a flagged anomaly.
ANOMALY_ACTIONS = ("flag", "cap", "quarantine")

#: Named seasonal buckets → a function mapping a datetime index to a season key.
_SEASON_KEYS = {
    "hour": lambda idx: idx.hour,
    "day": lambda idx: idx.dayofweek,
    "dayofweek": lambda idx: idx.dayofweek,
    "week": lambda idx: idx.isocalendar().week.to_numpy(),
    "month": lambda idx: idx.month,
}


def to_timedelta(value: object) -> pd.Timedelta | None:
    """Coerce ``allowed_lateness`` (``"10m"``, seconds, ``Timedelta``) to a Timedelta."""
    if value is None:
        return None
    if isinstance(value, pd.Timedelta):
        return value
    if isinstance(value, (int, float)):
        return pd.to_timedelta(float(value), unit="s")
    return pd.to_timedelta(value)


@dataclass(frozen=True)
class TimeSeriesCleanConfig:
    """How a time-series / streaming-aware clean treats ordered, timestamped data.

    Parameters
    ----------
    timestamp_column:
        Column used to order rows within each entity (interpolation, anomaly windows).
    entity_id_columns:
        Columns identifying an independent series (e.g. ``("sensor_id",)``); all
        ordering, interpolation, seasonality and anomaly windows are computed *within*
        each entity group. Empty means one global series.
    frequency:
        Expected sampling frequency (pandas offset alias, e.g. ``"1min"``). Informational
        today; recorded on the report so downstream resampling can use it.
    max_interpolation_gap:
        Longest run of consecutive missing values that short-gap interpolation will fill.
        Longer gaps are left missing (and only touched by seasonal imputation if enabled).
    interpolation_method:
        One of :data:`INTERPOLATION_METHODS` — ``"time"`` (default) and ``"linear"``
        interpolate numerically; ``"ffill"`` / ``"bfill"`` carry the last/next value.
    seasonal_period:
        Seasonal bucket for seasonal imputation: ``"hour"``, ``"day"``/``"dayofweek"``,
        ``"week"``, or ``"month"``. ``None`` disables the seasonal bucket (rolling-median
        fallback still applies when ``seasonal_imputation_enabled``).
    seasonal_imputation_enabled:
        When True, fill the gaps short-gap interpolation left behind using the same
        season's median (falling back to a rolling/global median), each with a confidence.
    ordered_dedupe_keys:
        Identity keys for deduplication; duplicates are collapsed using event-time order.
        Empty disables ordered dedupe.
    ordered_dedupe_keep:
        One of :data:`DEDUPE_KEEP`: keep the ``first``/``last`` row, the
        ``latest_event_time`` row, or the ``highest_quality`` row (needs ``quality_column``).
    event_time_column:
        Column carrying the true event time for watermarking and dedupe ordering.
        Defaults to ``timestamp_column``. (``watermark_column`` is an accepted alias.)
    allowed_lateness:
        How far behind the watermark an event may still be accepted. Accepts a pandas
        duration string (``"10m"``), a number of seconds, or a ``Timedelta``.
    late_data_action:
        One of :data:`LATE_DATA_ACTIONS` for events older than
        ``watermark - allowed_lateness``.
    anomaly_window_size:
        Rolling window (in rows, per entity) for windowed anomaly detection. ``0`` disables.
    anomaly_method:
        One of :data:`ANOMALY_METHODS`.
    anomaly_threshold:
        Score cutoff: a |z|/robust-z above this (or, for ``iqr``, outside the fence) flags.
    anomaly_action:
        One of :data:`ANOMALY_ACTIONS`. ``"flag"`` (default) only adds a boolean column and
        never drops rows; ``"cap"`` clips to the window fence; ``"quarantine"`` removes
        flagged rows into the exceptions output.
    quality_column:
        Numeric column whose larger value wins ``highest_quality`` dedupe ties.
    protected_columns:
        Extra columns never interpolated, seasonally filled, or anomaly-scored.
    """

    timestamp_column: str
    entity_id_columns: tuple[str, ...] = ()
    frequency: str | None = None
    max_interpolation_gap: int = 1
    interpolation_method: str = "time"
    seasonal_period: str | None = None
    seasonal_imputation_enabled: bool = False
    ordered_dedupe_keys: tuple[str, ...] = ()
    ordered_dedupe_keep: str = "latest_event_time"
    event_time_column: str | None = None
    watermark_column: str | None = None
    allowed_lateness: object = None
    late_data_action: str = "quarantine"
    anomaly_window_size: int = 0
    anomaly_method: str = "rolling_zscore"
    anomaly_threshold: float = 3.0
    anomaly_action: str = "flag"
    quality_column: str | None = None
    protected_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_column, str) or not self.timestamp_column:
            raise ValueError("timestamp_column must be a non-empty string")
        _check("interpolation_method", self.interpolation_method, INTERPOLATION_METHODS)
        _check("ordered_dedupe_keep", self.ordered_dedupe_keep, DEDUPE_KEEP)
        _check("late_data_action", self.late_data_action, LATE_DATA_ACTIONS)
        _check("anomaly_method", self.anomaly_method, ANOMALY_METHODS)
        _check("anomaly_action", self.anomaly_action, ANOMALY_ACTIONS)
        if self.max_interpolation_gap < 0:
            raise ValueError("max_interpolation_gap must be >= 0")
        if self.anomaly_window_size < 0:
            raise ValueError("anomaly_window_size must be >= 0")
        if self.anomaly_threshold <= 0:
            raise ValueError("anomaly_threshold must be > 0")
        if (self.ordered_dedupe_keep == "highest_quality"
                and self.ordered_dedupe_keys and self.quality_column is None):
            raise ValueError("ordered_dedupe_keep='highest_quality' needs quality_column")
        # Normalise allowed_lateness eagerly so a bad value fails fast.
        to_timedelta(self.allowed_lateness)

    @property
    def resolved_event_time_column(self) -> str:
        """The column used for watermarking / event-time ordering."""
        return (self.event_time_column or self.watermark_column
                or self.timestamp_column)

    @property
    def late_data_enabled(self) -> bool:
        """Late-data handling only runs when explicitly requested — i.e. an event-time
        or watermark column is named, or ``allowed_lateness`` is set. Otherwise the
        timestamp fallback would spuriously flag interleaved entities as "late"."""
        return bool(self.event_time_column or self.watermark_column
                    or self.allowed_lateness)


def _check(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


@dataclass
class TimeSeriesProcessor:
    """Applies the time-series steps to each batch, keeping the per-stream watermark.

    Construct once and call :meth:`process` per batch; the watermark advances across
    batches so late data is judged against everything seen so far. Designed to be driven
    by :class:`~freshdata.streaming.StreamingCleaner`, but usable standalone.
    """

    config: TimeSeriesCleanConfig
    clean_config: CleanConfig = field(default_factory=CleanConfig)

    def __post_init__(self) -> None:
        self.watermark: pd.Timestamp | None = None  # global max, for reporting
        # Per-entity watermarks (keyed by the entity-id tuple, or () for a single
        # stream); each advances monotonically across batches.
        self._entity_watermarks: dict[tuple, pd.Timestamp] = {}
        self.late_quarantined_total = 0
        self.late_dropped_total = 0
        self.anomalies_flagged_total = 0
        self.anomalies_quarantined_total = 0
        self.last_summary: dict[str, object] = {}
        self.last_numeric_cols: list[str] = []

    # -- public ----------------------------------------------------------------

    def process(self, df: pd.DataFrame, report: CleanReport, *,
                roles: dict[str, str] | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run every configured time-series step on *df*.

        Returns ``(cleaned, exceptions)`` where *exceptions* holds the rows pulled out by
        late-data or anomaly quarantine (empty frame when nothing was quarantined). The
        cleaned frame is returned sorted by entity then timestamp — the natural order for
        a time series — and *report* is appended to in place.
        """
        cfg = self.config
        df = df.copy()
        cleaned_exc: list[pd.DataFrame] = []
        summary: dict[str, object] = {}

        ts_col = cfg.timestamp_column
        if ts_col not in df.columns:
            # Nothing time-series-shaped about this batch; leave it untouched.
            self.last_summary = {}
            return df, _empty_like(df)
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

        # 1. Late data is judged in arrival order, before any sort reorders the batch.
        df, late_exc, late_meta = self._handle_late_data(df, report)
        if late_exc is not None and len(late_exc):
            cleaned_exc.append(late_exc)
        summary.update(late_meta)

        # 2. Ordered dedupe (deterministic: keys + event-time / quality ordering).
        df = self._ordered_dedupe(df, report)

        # Establish the canonical (entity, timestamp) order for the remaining steps.
        sort_cols = [*cfg.entity_id_columns, ts_col]
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        roles = roles or self._infer_roles(df)
        numeric_cols = self.numeric_targets(df, roles)
        self.last_numeric_cols = list(numeric_cols)

        # 3. Short-gap interpolation, then 4. seasonal imputation of what's left.
        df = self._interpolate(df, numeric_cols, report)
        if cfg.seasonal_imputation_enabled:
            df = self._seasonal_impute(df, numeric_cols, report)

        # 5. Windowed anomaly detection (flag columns by default; cap/quarantine opt-in).
        df, anom_exc, anom_meta = self._windowed_anomaly(df, numeric_cols, report)
        if anom_exc is not None and len(anom_exc):
            cleaned_exc.append(anom_exc)
        summary.update(anom_meta)

        exceptions = (pd.concat(cleaned_exc, ignore_index=True)
                      if cleaned_exc else _empty_like(df))
        if cfg.frequency:
            summary["frequency"] = cfg.frequency
        self.last_summary = summary
        return df, exceptions

    # -- step 5 (numbered by the spec): watermark-aware late data ---------------

    def _handle_late_data(self, df: pd.DataFrame, report: CleanReport
                          ) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, object]]:
        cfg = self.config
        if not cfg.late_data_enabled:
            return df, None, {}
        lateness = to_timedelta(cfg.allowed_lateness) or pd.Timedelta(0)
        event_col = cfg.resolved_event_time_column
        if event_col not in df.columns:
            return df, None, {}
        event_time = pd.to_datetime(df[event_col], errors="coerce")

        late_mask = self._late_mask(df, event_time, lateness)
        n_late = int(late_mask.sum())
        if not n_late:
            return df, None, {"watermark": _iso(self.watermark)}

        late_rows = df.loc[late_mask]
        meta: dict[str, object] = {"watermark": _iso(self.watermark)}
        if cfg.late_data_action == "drop":
            report.add("late_data", f"dropped {n_late} late event(s) past the watermark",
                       count=n_late, risk="medium",
                       rationale=f"event_time < watermark - {lateness} (allowed_lateness)")
            self.late_dropped_total += n_late
            meta["late_dropped"] = n_late
            return df.loc[~late_mask].reset_index(drop=True), None, meta
        if cfg.late_data_action == "keep_with_warning":
            report.add("late_data", f"kept {n_late} late event(s) with a warning",
                       count=n_late, risk="medium",
                       rationale=f"event_time < watermark - {lateness}; kept per config")
            report.add_warning(f"{n_late} late event(s) kept past the watermark")
            meta["late_kept_with_warning"] = n_late
            return df.reset_index(drop=True), None, meta
        # quarantine (default)
        report.add("late_data", f"quarantined {n_late} late event(s) past the watermark",
                   count=n_late, risk="medium",
                   rationale=f"event_time < watermark - {lateness} (allowed_lateness)")
        self.late_quarantined_total += n_late
        meta["late_quarantined"] = n_late
        exc = late_rows.copy()
        exc["_quarantine_reason"] = "late_data"
        return df.loc[~late_mask].reset_index(drop=True), exc, meta

    def _late_mask(self, df: pd.DataFrame, event_time: pd.Series,
                   lateness: pd.Timedelta) -> pd.Series:
        """Per-entity progressive watermark: each entity's watermark advances row-by-row
        in arrival order (as if each row were its own micro-batch), seeded by the
        carry-over from earlier batches. A row is late iff it predates
        ``watermark_so_far - allowed_lateness``. Independent series never make each other
        look late. Updates ``self._entity_watermarks`` and the global ``self.watermark``."""
        keys = [k for k in self.config.entity_id_columns if k in df.columns]
        late_mask = pd.Series(False, index=df.index)
        # Group by a scalar (not a 1-element list) for a single entity key and iterate
        # the groups directly — ``groupby([col]).groups`` raises a Pandas4Warning.
        if keys:
            by: Any = keys[0] if len(keys) == 1 else keys
            grouped: list[tuple[Any, pd.Index]] = [
                (k, g.index) for k, g in df.groupby(by, sort=False, dropna=False)]
        else:
            grouped = [((), df.index)]
        for key, idx in grouped:
            ekey = () if not keys else (key,) if len(keys) == 1 else tuple(key)
            et = event_time.loc[idx]
            start_wm = self._entity_watermarks.get(ekey)
            prior_wm = et.cummax().shift(1)
            if start_wm is not None:
                prior_wm = prior_wm.fillna(start_wm).clip(lower=start_wm)
            late_mask.loc[idx] = (
                prior_wm.notna() & (et < prior_wm - lateness)).fillna(False)
            batch_max = et.max()
            if pd.notna(batch_max):
                self._entity_watermarks[ekey] = (
                    batch_max if start_wm is None else max(start_wm, batch_max))
        if self._entity_watermarks:
            self.watermark = max(self._entity_watermarks.values())
        return late_mask

    # -- step 4: ordered dedupe -------------------------------------------------

    def _ordered_dedupe(self, df: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
        cfg = self.config
        keys = [k for k in cfg.ordered_dedupe_keys if k in df.columns]
        if not keys:
            return df
        before = len(df)
        event_col = cfg.resolved_event_time_column
        keep = cfg.ordered_dedupe_keep
        # Sort so the row to keep lands last within each key group, then keep='last'.
        # mergesort is stable → deterministic tie-breaking by original order.
        qcol = cfg.quality_column
        if keep == "highest_quality" and qcol is not None and qcol in df.columns:
            order, by = [*keys, qcol], "highest quality score"
            ordered = df.sort_values(order, kind="mergesort")
            deduped = ordered.drop_duplicates(subset=keys, keep="last")
        elif keep == "first":
            ordered, by = df, "first occurrence"
            deduped = ordered.drop_duplicates(subset=keys, keep="first")
        elif keep == "last":
            ordered, by = df, "last occurrence"
            deduped = ordered.drop_duplicates(subset=keys, keep="last")
        else:  # latest_event_time (default)
            order = [*keys, event_col] if event_col in df.columns else keys
            by = "latest event time"
            ordered = df.sort_values(order, kind="mergesort")
            deduped = ordered.drop_duplicates(subset=keys, keep="last")
        removed = before - len(deduped)
        # Restore the input row order among the survivors for a deterministic result.
        deduped = deduped.sort_index().reset_index(drop=True)
        if removed:
            report.add("ordered_dedupe",
                       f"collapsed {removed} duplicate row(s) on {keys}, keeping {by}",
                       count=removed, risk="low",
                       rationale=f"ordered dedupe keep={keep}")
            report.duplicates_removed += removed
        return deduped

    # -- step 2: short-gap interpolation ---------------------------------------

    def _interpolate(self, df: pd.DataFrame, numeric_cols: list[str],
                     report: CleanReport) -> pd.DataFrame:
        cfg = self.config
        if not numeric_cols or cfg.max_interpolation_gap <= 0:
            return df
        method = cfg.interpolation_method
        ts_col = cfg.timestamp_column
        for col in numeric_cols:
            filled_total = 0

            def fill_group(g: pd.DataFrame, col: str = col) -> pd.DataFrame:
                nonlocal filled_total
                s = g[col]
                isna = s.isna()
                if not isna.any():
                    return g
                # Length of each consecutive run; only short NaN runs are fillable.
                run = (isna != isna.shift()).cumsum()
                run_len = isna.groupby(run).transform("sum")
                fillable = isna & (run_len <= cfg.max_interpolation_gap)
                if not fillable.any():
                    return g
                interp = self._interp_series(s, g[ts_col], method)
                newly = fillable & interp.notna() & s.isna()
                g.loc[newly, col] = interp[newly]
                filled_total += int(newly.sum())
                return g

            df = self._apply_per_entity(df, fill_group)
            if filled_total:
                report.add("timeseries_interpolation",
                           f"interpolated {filled_total} short-gap value(s) "
                           f"(<= {cfg.max_interpolation_gap} step gap, method={method})",
                           column=col, count=filled_total, risk="low",
                           confidence=0.8 if method in ("time", "linear") else 0.7,
                           rationale="short consecutive gap in an ordered series",
                           model_id=f"interp_{method}")
                report.columns_imputed.append(col)
        return df

    @staticmethod
    def _interp_series(s: pd.Series, ts: pd.Series, method: str) -> pd.Series:
        if method == "ffill":
            return s.ffill()
        if method == "bfill":
            return s.bfill()
        if method == "time":
            tmp = pd.Series(s.to_numpy(dtype="float64"),
                            index=pd.DatetimeIndex(ts))
            try:
                out = tmp.interpolate(method="time", limit_area="inside")
            except ValueError:  # non-monotonic / non-datetime index → linear fallback
                out = pd.Series(s.to_numpy(dtype="float64")).interpolate(
                    method="linear", limit_area="inside")
            return pd.Series(out.to_numpy(), index=s.index)
        # linear
        return pd.Series(s.to_numpy(dtype="float64"),
                         index=s.index).interpolate(method="linear", limit_area="inside")

    # -- step 3: seasonal imputation -------------------------------------------

    def _seasonal_impute(self, df: pd.DataFrame, numeric_cols: list[str],
                         report: CleanReport) -> pd.DataFrame:
        cfg = self.config
        if not numeric_cols:
            return df
        ts_col = cfg.timestamp_column
        season_fn = _SEASON_KEYS.get(cfg.seasonal_period or "")
        for col in numeric_cols:
            if not df[col].isna().any():
                continue
            seasonal_filled = fallback_filled = 0

            def fill_group(g: pd.DataFrame, col: str = col) -> pd.DataFrame:
                nonlocal seasonal_filled, fallback_filled
                s = g[col]
                missing = s.isna()
                if not missing.any():
                    return g
                global_med = s.median()
                if season_fn is not None:
                    keys = pd.Series(season_fn(pd.DatetimeIndex(g[ts_col])), index=g.index)
                    season_med = s.groupby(keys).transform("median")
                    season_count = s.groupby(keys).transform("count")
                    use_season = missing & season_med.notna() & (season_count >= 3)
                    g.loc[use_season, col] = season_med[use_season]
                    seasonal_filled += int(use_season.sum())
                    missing = g[col].isna()
                # Fallback: rolling median (then global) for whatever's left.
                if missing.any() and pd.notna(global_med):
                    rolling = s.rolling(window=5, min_periods=1, center=True).median()
                    fill_val = rolling.where(rolling.notna(), global_med)
                    g.loc[missing, col] = fill_val[missing]
                    fallback_filled += int(missing.sum())
                return g

            df = self._apply_per_entity(df, fill_group)
            total = seasonal_filled + fallback_filled
            if total:
                conf = 0.7 if seasonal_filled >= fallback_filled else 0.5
                rationale = (f"{seasonal_filled} from matching {cfg.seasonal_period} season, "
                             f"{fallback_filled} from rolling/global median fallback")
                report.add("seasonal_imputation",
                           f"seasonally imputed {total} value(s)", column=col, count=total,
                           risk="medium", confidence=conf, rationale=rationale,
                           model_id="seasonal_median")
                report.columns_imputed.append(col)
        return df

    # -- step 6: windowed anomaly detection ------------------------------------

    def _windowed_anomaly(self, df: pd.DataFrame, numeric_cols: list[str],
                          report: CleanReport
                          ) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, object]]:
        cfg = self.config
        if not numeric_cols or cfg.anomaly_window_size <= 0:
            return df, None, {}
        win, thr, method = cfg.anomaly_window_size, cfg.anomaly_threshold, cfg.anomaly_method
        any_flag = pd.Series(False, index=df.index)
        meta: dict[str, object] = {}
        for col in numeric_cols:
            flag_col = f"{col}_anomaly"
            flags = pd.Series(False, index=df.index)
            lower_all = pd.Series(np.nan, index=df.index)
            upper_all = pd.Series(np.nan, index=df.index)

            def score_group(g: pd.DataFrame, col: str = col, flags: pd.Series = flags,
                            lower_all: pd.Series = lower_all,
                            upper_all: pd.Series = upper_all) -> pd.DataFrame:
                s = g[col].astype("float64")
                f, lo, hi = _anomaly_scores(s, win, thr, method)
                flags.loc[g.index] = f.to_numpy()
                lower_all.loc[g.index] = lo.to_numpy()
                upper_all.loc[g.index] = hi.to_numpy()
                return g

            self._apply_per_entity(df, score_group)  # populates flags/bounds by index
            n_flag = int(flags.sum())
            if not n_flag:
                df[flag_col] = False
                continue
            df[flag_col] = flags.to_numpy()
            any_flag = any_flag | flags
            self.anomalies_flagged_total += n_flag
            action_note = "flagged"
            if cfg.anomaly_action == "cap":
                capped = df[col].astype("float64").clip(lower=lower_all, upper=upper_all)
                df.loc[flags, col] = capped[flags]
                report.outliers_handled += n_flag
                action_note = "flagged and capped"
            report.add("windowed_anomaly",
                       f"{action_note} {n_flag} windowed anomaly(ies) (method={method}, "
                       f"window={win})", column=col, count=n_flag, risk="medium",
                       confidence=0.7,
                       rationale=f"rolling {method} score beyond {thr}",
                       model_id=f"anomaly_{method}")
            meta[f"{col}_anomalies"] = n_flag

        if cfg.anomaly_action == "quarantine" and any_flag.any():
            n = int(any_flag.sum())
            exc = df.loc[any_flag].copy()
            exc["_quarantine_reason"] = "windowed_anomaly"
            self.anomalies_quarantined_total += n
            meta["anomalies_quarantined"] = n
            report.add("windowed_anomaly", f"quarantined {n} anomalous row(s)",
                       count=n, risk="medium", rationale="anomaly_action=quarantine")
            kept = df.loc[~any_flag].reset_index(drop=True)
            return kept, exc, meta
        return df, None, meta

    # -- helpers ----------------------------------------------------------------

    def _apply_per_entity(self, df: pd.DataFrame,
                          fn) -> pd.DataFrame:
        """Apply *fn* to each entity group (or the whole frame), preserving row order."""
        keys = [k for k in self.config.entity_id_columns if k in df.columns]
        if not keys:
            return fn(df)
        # Iterate groups explicitly rather than ``groupby.apply`` — the latter both
        # strips grouping columns (pandas 2.2 FutureWarning) and can drop rows when the
        # callback returns a same-shaped frame. Each group keeps its original (unique)
        # index, so a final ``reindex`` restores the input order without losing rows.
        pieces = [fn(g.copy()) for _, g in df.groupby(keys, sort=False, dropna=False)]
        return pd.concat(pieces).reindex(df.index)

    def _infer_roles(self, df: pd.DataFrame) -> dict[str, str]:
        return {str(c): infer_role(str(c), df[c], self.clean_config) for c in df.columns}

    def numeric_targets(self, df: pd.DataFrame,
                        roles: dict[str, str] | None = None) -> list[str]:
        """Numeric columns this processor manages (interpolation/seasonal/anomaly).

        These are the columns a host streaming cleaner should *not* statistically
        impute, so the time-series policy (short-gap interpolation, long-gap
        preservation) owns their missing values.
        """
        roles = roles or self._infer_roles(df)
        cfg = self.config
        protected = {
            cfg.timestamp_column, cfg.resolved_event_time_column,
            cfg.quality_column, *cfg.entity_id_columns, *cfg.ordered_dedupe_keys,
            *cfg.protected_columns,
        }
        cols = []
        for c in df.columns:
            name = str(c)
            if name in protected or name.endswith("_anomaly"):
                continue
            if roles.get(name) != "numeric":
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            cols.append(name)
        return cols


def _anomaly_scores(s: pd.Series, win: int, thr: float, method: str
                    ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(flags, lower_fence, upper_fence)`` for one series and method."""
    min_p = max(2, win // 2)
    if method == "iqr":
        q1 = s.rolling(win, min_periods=min_p).quantile(0.25)
        q3 = s.rolling(win, min_periods=min_p).quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags = ((s < lower) | (s > upper)).fillna(False)
        return flags, lower, upper
    if method == "mad":
        med = s.rolling(win, min_periods=min_p).median()
        mad = (s - med).abs().rolling(win, min_periods=min_p).median()
        robust_z = 0.6745 * (s - med) / mad.replace(0, np.nan)
        lower, upper = med - thr * mad / 0.6745, med + thr * mad / 0.6745
        flags = _flag(robust_z, thr, s, med, mad)
        return flags, lower, upper
    if method == "ewma":
        # Judge each point against the EWMA *forecast* from prior points (shifted by one).
        # Including the current value would pull the mean toward a spike and inflate the
        # std, masking the very anomaly we are looking for.
        mean = s.ewm(span=win, min_periods=min_p).mean().shift(1)
        std = s.ewm(span=win, min_periods=min_p).std().shift(1)
        z = (s - mean) / std.replace(0, np.nan)
        lower, upper = mean - thr * std, mean + thr * std
        flags = _flag(z, thr, s, mean, std)
        return flags, lower, upper
    # rolling_zscore (default)
    mean = s.rolling(win, min_periods=min_p).mean()
    std = s.rolling(win, min_periods=min_p).std()
    z = (s - mean) / std.replace(0, np.nan)
    lower, upper = mean - thr * std, mean + thr * std
    flags = _flag(z, thr, s, mean, std)
    return flags, lower, upper


def _flag(z: pd.Series, thr: float, s: pd.Series, center: pd.Series,
          scale: pd.Series) -> pd.Series:
    """Flag points beyond *thr* standard scores, plus the degenerate case where the
    rolling scale is exactly zero (a flat window) yet the point still deviates from the
    centre — there the z-score is NaN, so it would otherwise slip through unflagged."""
    beyond = (z.abs() > thr).fillna(False)
    flat_spike = (scale.fillna(np.nan) == 0) & ((s - center).abs() > 0)
    return (beyond | flat_spike.fillna(False))


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    out = df.iloc[0:0].copy()
    if "_quarantine_reason" not in out.columns:
        out["_quarantine_reason"] = pd.Series(dtype="object")
    return out


def _iso(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None or pd.isna(ts) else ts.isoformat()
