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
one of the ``timeseries_timestamp_parse`` / ``timeseries_interpolation`` /
``seasonal_imputation`` / ``ordered_dedupe`` / ``late_data`` / ``windowed_anomaly`` step
names, so the trust contract is preserved.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._util import mask_sensitive_value, safe_median
from ..config import CleanConfig
from ..engine.context import infer_role
from ..report import CleanReport
from ..steps.dtypes import COERCED_CELLS_CAP

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
#: Epoch units accepted for numeric timestamp columns.
TIMESTAMP_UNITS = ("s", "ms", "us", "ns")
#: Upper bounds on the median |epoch value| for each inferred unit: seconds up to
#: 1e11 (year ~5100), milliseconds up to 1e14, microseconds up to 1e17, else ns.
_EPOCH_UNIT_BOUNDS = ((1e11, "s"), (1e14, "ms"), (1e17, "us"))
#: ``pd.to_datetime`` keyword sets :func:`coerce_datetimes` tries, in order.
#: ``format="ISO8601"`` exists (and is needed) only on pandas >= 2.
_DATETIME_PARSE_ATTEMPTS: tuple[dict[str, Any], ...] = (
    ({}, {"utc": True}, {"format": "ISO8601"}, {"format": "ISO8601", "utc": True})
    if int(pd.__version__.split(".")[0]) >= 2 else ({}, {"utc": True}))

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


def _n_lost(values: pd.Series, parsed: pd.Series) -> int:
    """How many present values the parse turned into ``NaT``."""
    return int((values.notna() & parsed.isna()).sum())


def coerce_datetimes(values: pd.Series) -> pd.Series:
    """Parse *values* with ``errors="coerce"``, surviving mixed UTC offsets.

    Datetime columns pass through unchanged. Values with different UTC offsets
    (for example either side of a DST change), or a mix of naive and
    offset-aware values, have no single naive or fixed-offset representation.
    Depending on the pandas version, a plain parse then raises, returns an
    ``object`` column, or turns the minority into ``NaT``. So when the plain
    parse is unusable or loses values, the batch is re-parsed with ``utc=True``
    (reading naive values as UTC) and, on pandas >= 2, which infers one format
    from the first value, also with ``format="ISO8601"``. The attempt that
    loses the fewest values wins, the earliest on a tie, so clean batches pay
    for one parse. Numeric input keeps pandas' default (nanoseconds);
    :func:`parse_timestamps` infers epoch units.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    best: pd.Series | None = None
    best_lost = 0
    for kwargs in _DATETIME_PARSE_ATTEMPTS:
        try:
            with warnings.catch_warnings():
                # pandas 2 warns before returning an object column for mixed
                # offsets; a later utc=True attempt handles exactly that case.
                warnings.filterwarnings("ignore", message=".*mixed time zones.*")
                parsed = pd.to_datetime(values, errors="coerce", **kwargs)
        except (ValueError, TypeError):
            continue
        if not pd.api.types.is_datetime64_any_dtype(parsed):
            continue  # object column of mixed-offset datetimes
        lost = _n_lost(values, parsed)
        if best is None or lost < best_lost:
            best, best_lost = parsed, lost
        if not lost:
            break
    if best is None:  # every attempt raised: surface the plain parse's error
        return pd.to_datetime(values, errors="coerce")
    return best


def infer_epoch_unit(values: pd.Series) -> str:
    """Guess the epoch unit of numeric timestamps from their median magnitude."""
    present = values.dropna()
    if not len(present):
        return "ns"
    magnitude = float(present.abs().median())
    for bound, unit in _EPOCH_UNIT_BOUNDS:
        if magnitude < bound:
            return unit
    return "ns"


def parse_timestamps(values: pd.Series, unit: str | None = None
                     ) -> tuple[pd.Series, str | None]:
    """Parse a timestamp column; return ``(parsed, epoch_unit)``.

    Numeric (non-bool) columns are epoch values in *unit*, or in the unit
    :func:`infer_epoch_unit` infers when *unit* is ``None``; ``epoch_unit``
    reports the unit used. Other columns go through :func:`coerce_datetimes`,
    and ``epoch_unit`` is ``None``. Values that do not parse (or are out of
    range) become ``NaT``.
    """
    dtype = values.dtype
    if not (pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)):
        return coerce_datetimes(values), None
    unit = unit or infer_epoch_unit(values)
    present = values.notna().to_numpy()
    kept = values[present]
    # Signed ints stay exact (ns epochs exceed float precision); everything else
    # goes through float64 so nullable dtypes never hit pd.NA.
    raw = (kept.to_numpy(dtype="int64") if pd.api.types.is_signed_integer_dtype(dtype)
           else kept.to_numpy(dtype="float64"))
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if len(raw):
        parsed[present] = pd.to_datetime(raw, unit=unit, errors="coerce").to_numpy()
    return parsed, unit


def _as_float(s: pd.Series) -> pd.Series:
    """Score a column as float64; non-numeric cells (e.g. a stray string) become NaN."""
    numeric = pd.to_numeric(s, errors="coerce")
    return pd.Series(numeric.to_numpy(dtype="float64", na_value=np.nan), index=s.index)


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
    timestamp_unit:
        Epoch unit (one of :data:`TIMESTAMP_UNITS`) for *numeric* timestamp and
        event-time columns. ``None`` (default) infers it from the magnitude of the
        values (seconds, milliseconds, microseconds or nanoseconds). Ignored for
        string and datetime columns.
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
    timestamp_unit: str | None = None

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
        if self.anomaly_window_size < 0 or self.anomaly_window_size == 1:
            # A one-row window has no spread to score against (and the rolling
            # min_periods of 2 would exceed it), so 0 (disabled) or >= 2 only.
            raise ValueError("anomaly_window_size must be 0 or >= 2")
        if self.timestamp_unit is not None:
            _check("timestamp_unit", self.timestamp_unit, TIMESTAMP_UNITS)
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
        df[ts_col] = self._parse_timestamp_column(df[ts_col], report)

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
        self.last_numeric_cols = [str(c) for c in numeric_cols]
        anomaly_cols = self.anomaly_targets(df, roles)

        # 3. Short-gap interpolation, then 4. seasonal imputation of what's left.
        df = self._interpolate(df, numeric_cols, report)
        if cfg.seasonal_imputation_enabled:
            df = self._seasonal_impute(df, numeric_cols, report)

        # 5. Windowed anomaly detection (flag columns by default; cap/quarantine opt-in).
        df, anom_exc, anom_meta = self._windowed_anomaly(df, anomaly_cols, report)
        if anom_exc is not None and len(anom_exc):
            cleaned_exc.append(anom_exc)
        summary.update(anom_meta)

        exceptions = (pd.concat(cleaned_exc, ignore_index=True)
                      if cleaned_exc else _empty_like(df))
        if cfg.frequency:
            summary["frequency"] = cfg.frequency
        self.last_summary = summary
        return df, exceptions

    def _parse_timestamp_column(self, raw: pd.Series, report: CleanReport) -> pd.Series:
        """Parse the timestamp column, auditing the epoch unit and every lost value.

        Rows whose timestamp does not parse are kept (with ``NaT``); their
        original values are preserved in ``report.coerced_cells``.
        """
        cfg = self.config
        name = str(cfg.timestamp_column)
        parsed, epoch_unit = parse_timestamps(raw, cfg.timestamp_unit)
        if epoch_unit is not None:
            source = ("timestamp_unit" if cfg.timestamp_unit
                      else "inferred from the magnitude of the values")
            report.add("timeseries_timestamp_parse",
                       f"read numeric timestamps as epoch unit '{epoch_unit}'",
                       column=name, count=int(parsed.notna().sum()), risk="low",
                       rationale=f"epoch unit {source}")
        lost = raw.notna() & parsed.isna()
        n_lost = int(lost.sum())
        if not n_lost:
            return parsed
        originals = raw[lost]
        sensitive = name in self.clean_config.sensitive_columns
        cells = report.coerced_cells.setdefault(name, {})
        for row, value in originals.items():
            if len(cells) >= COERCED_CELLS_CAP:
                break
            cells[row] = mask_sensitive_value(value) if sensitive else value
        report.coerced_rows[name] = tuple(
            dict.fromkeys([*report.coerced_rows.get(name, ()), *originals.index]))
        examples = ", ".join(
            f"{mask_sensitive_value(v) if sensitive else repr(v)} (row {i})"
            for i, v in list(originals.head(3).items()))
        report.add("timeseries_timestamp_parse",
                   f"{n_lost} timestamp value(s) could not be parsed and were set "
                   "to missing; rows kept", column=name, count=n_lost, risk="medium",
                   rationale="unparseable or out-of-range timestamp; originals "
                             "preserved in report.coerced_cells")
        report.add_warning(
            f"column '{name}': {n_lost} timestamp value(s) could not be parsed "
            f"and were set to missing, e.g. {examples}. Originals are preserved "
            "in report.coerced_cells.")
        return parsed

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
        event_time, _ = parse_timestamps(df[event_col], cfg.timestamp_unit)

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
                           column=str(col), count=filled_total, risk="low",
                           confidence=0.8 if method in ("time", "linear") else 0.7,
                           rationale="short consecutive gap in an ordered series",
                           model_id=f"interp_{method}")
                report.columns_imputed.append(str(col))
        return df

    @staticmethod
    def _interp_series(s: pd.Series, ts: pd.Series, method: str) -> pd.Series:
        if method == "ffill":
            return s.ffill()
        if method == "bfill":
            return s.bfill()
        # na_value: nullable (masked) columns with missing cells refuse a plain
        # float64 conversion on pandas < 2.
        values = s.to_numpy(dtype="float64", na_value=np.nan)
        if method == "time":
            tmp = pd.Series(values, index=pd.DatetimeIndex(ts))
            try:
                out = tmp.interpolate(method="time", limit_area="inside")
            except ValueError:  # non-monotonic / non-datetime index → linear fallback
                out = pd.Series(values).interpolate(method="linear", limit_area="inside")
            return pd.Series(out.to_numpy(), index=s.index)
        # linear
        return pd.Series(values, index=s.index).interpolate(
            method="linear", limit_area="inside")

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
                global_med = safe_median(s)
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
                           f"seasonally imputed {total} value(s)", column=str(col), count=total,
                           risk="medium", confidence=conf, rationale=rationale,
                           model_id="seasonal_median")
                report.columns_imputed.append(str(col))
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
                s = _as_float(g[col])
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
                capped = _as_float(df[col]).clip(lower=lower_all, upper=upper_all)
                df.loc[flags, col] = capped[flags]
                report.outliers_handled += n_flag
                action_note = "flagged and capped"
            report.add("windowed_anomaly",
                       f"{action_note} {n_flag} windowed anomaly(ies) (method={method}, "
                       f"window={win})", column=str(col), count=n_flag, risk="medium",
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
        return [c for c in self._numeric_role_columns(df, roles)
                if pd.api.types.is_numeric_dtype(df[c])]

    def anomaly_targets(self, df: pd.DataFrame,
                        roles: dict[str, str] | None = None) -> list[str]:
        """Columns that get a ``<col>_anomaly`` flag column.

        Chosen by (locked) role rather than by this batch's dtype: a numeric
        column whose batch holds a stray string arrives as ``object``, and must
        still be scored (non-numeric cells score as missing) so every batch emits
        the same flag columns.
        """
        return [c for c in self._numeric_role_columns(df, roles)
                if pd.api.types.is_numeric_dtype(df[c])
                or pd.api.types.is_object_dtype(df[c])
                or pd.api.types.is_string_dtype(df[c])]

    def _numeric_role_columns(self, df: pd.DataFrame,
                              roles: dict[str, str] | None) -> list[Any]:
        """Unprotected columns whose role is ``numeric``, as the frame's own labels."""
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
            cols.append(c)  # the frame's own label (may be an int), not str(c)
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
        # A zero MAD only means more than half the window equals the median,
        # which is routine for sparse or two-valued series. The window is flat
        # only when every *other* point in it (the win - 1 before this one) is
        # identical; a deviating point there is a spike. Otherwise a zero MAD
        # gives no scale, so the point is not flagged and has no fence to cap to.
        others = s.shift(1).rolling(win - 1, min_periods=1)
        flat = (mad == 0) & (others.max() == others.min())
        fence = mad.where((mad > 0) | flat)
        lower, upper = med - thr * fence / 0.6745, med + thr * fence / 0.6745
        flags = _flag(robust_z, thr, s, med, flat)
        return flags, lower, upper
    if method == "ewma":
        # Judge each point against the EWMA *forecast* from prior points (shifted by one).
        # Including the current value would pull the mean toward a spike and inflate the
        # std, masking the very anomaly we are looking for.
        mean = s.ewm(span=win, min_periods=min_p).mean().shift(1)
        std = s.ewm(span=win, min_periods=min_p).std().shift(1)
        z = (s - mean) / std.replace(0, np.nan)
        lower, upper = mean - thr * std, mean + thr * std
        flags = _flag(z, thr, s, mean, std == 0)
        return flags, lower, upper
    # rolling_zscore (default)
    mean = s.rolling(win, min_periods=min_p).mean()
    std = s.rolling(win, min_periods=min_p).std()
    z = (s - mean) / std.replace(0, np.nan)
    lower, upper = mean - thr * std, mean + thr * std
    flags = _flag(z, thr, s, mean, std == 0)
    return flags, lower, upper


def _flag(z: pd.Series, thr: float, s: pd.Series, center: pd.Series,
          flat: pd.Series) -> pd.Series:
    """Flag points beyond *thr* standard scores, plus the degenerate case where the
    window is *flat* (the caller decides what that means for its scale) yet the point
    still deviates from the centre — there the z-score is NaN, so it would otherwise
    slip through unflagged."""
    beyond = (z.abs() > thr).fillna(False)
    flat_spike = flat.fillna(False) & ((s - center).abs() > 0)
    return (beyond | flat_spike.fillna(False))


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    out = df.iloc[0:0].copy()
    if "_quarantine_reason" not in out.columns:
        out["_quarantine_reason"] = pd.Series(dtype="object")
    return out


def _iso(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None or pd.isna(ts) else ts.isoformat()
