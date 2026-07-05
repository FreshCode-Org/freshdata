"""The stateful streaming / micro-batch cleaner.

:class:`StreamingCleaner` consumes batches, keeps **bounded** running statistics across
them (:class:`~freshdata.streaming.StreamingState`), and emits a cleaned batch plus a
per-batch :class:`~freshdata.CleanReport`. It reuses the in-memory pipeline for
representation repair and the enterprise trust score verbatim; only the
*statistical* imputation is reimplemented to draw on global running stats instead of
per-batch ones (so decisions are stable across the stream and memory never grows with
rows).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ..context import ContextPolicy

from ..cleaner import run_pipeline
from ..config import CleanConfig, merge_options
from ..engine.context import infer_role
from ..engine.missing import _band
from ..report import CleanReport
from ._config import StreamingCleanConfig
from ._connectors import coerce_to_pandas
from ._drift import detect_drift
from ._state import StreamingState
from ._timeseries import TimeSeriesCleanConfig, TimeSeriesProcessor

_STREAM_FIELDS = {f.name for f in dataclasses.fields(StreamingCleanConfig)}


class StreamingCleaner:
    """Clean an unbounded stream of DataFrame batches in constant memory.

    Examples
    --------
    >>> cleaner = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
    ...                               window_size=100_000, warmup_batches=3)
    >>> for cleaned_batch, report in cleaner.clean_batches(batch_iterable):  # doctest: +SKIP
    ...     write(cleaned_batch)
    ...     log(report.to_dict())
    >>> final = cleaner.finalize()  # doctest: +SKIP
    """

    def __init__(self, *, config: CleanConfig | None = None,
                 streaming_config: StreamingCleanConfig | None = None,
                 time_series_config: TimeSeriesCleanConfig | None = None,
                 **options: Any) -> None:
        stream_kwargs = {k: options.pop(k) for k in list(options) if k in _STREAM_FIELDS}
        self.scfg = streaming_config or StreamingCleanConfig(**stream_kwargs)
        self.config = merge_options(config, **options)
        # Context-policy state. A supplied ``context=``/``policy=`` is compiled
        # once, lazily, against the first batch's schema (see ``_ensure_policy``)
        # and lowered into stable protected columns / constraints, so every batch
        # is governed by the *same* decisions — no per-batch recompilation.
        self._policy: ContextPolicy | None = None
        self._protected: frozenset[str] = frozenset()
        self._policy_pending = (self.config.context is not None
                                or self.config.policy is not None)
        self._policy_reported = False
        # Representation-only config: engine off, no impute/outliers, schema-stable
        # across batches (never drop a column that is merely empty in one batch).
        self._rep_config = self._build_rep_config(self.config)
        self.state = StreamingState(
            reservoir_size=self.scfg.quantile_reservoir_size,
            max_categories=self.scfg.max_categories,
            rolling_trust_window=self.scfg.rolling_trust_window,
            seed=self.scfg.seed,
        )
        self._roles: dict[str, str] = {}
        self.report_: CleanReport | None = None
        self._seen_hashes: set[int] = set()  # recent-window cross-batch dedup
        self._n_imputed = 0
        self._n_deferred = 0
        self._gate_failures = 0
        # Time-series mode: one processor holds the per-stream watermark across batches.
        self.tcfg = time_series_config
        self._ts = (TimeSeriesProcessor(time_series_config, clean_config=self.config)
                    if time_series_config is not None else None)
        self.last_exceptions_: pd.DataFrame | None = None
        self._exception_batches: list[pd.DataFrame] = []

    # -- public API -------------------------------------------------------------

    def clean_batch(self, batch: Any) -> tuple[pd.DataFrame, CleanReport]:
        """Clean one micro-batch; return ``(cleaned_df, report)``."""
        from ..enterprise.metrics import compute_trust_score

        df = coerce_to_pandas(batch)
        self._ensure_policy(df)
        cleaned, report = run_pipeline(df, self._rep_config)
        if self._policy is not None and not self._policy_reported:
            self._policy_reported = True
            self._surface_policy(report)
        if self.scfg.global_duplicates:
            cleaned = self._dedup_window(cleaned, report)

        # Drift is judged before this batch updates the running state.
        findings = detect_drift(cleaned, self.state, self.scfg)
        for f in findings:
            report.add("drift", f.message, column=f.column, risk=f.risk,
                       rationale="schema/distribution drift vs. running baseline")
            report.add_warning(f"drift: {f.message}")
        self.state.drift_log.extend(
            {"batch": self.state.batch_count + 1, "kind": f.kind, "column": f.column,
             "message": f.message} for f in findings
        )

        self._lock_roles(cleaned)
        self.state.observe_batch(cleaned, roles=self._roles)

        # In time-series mode the TS processor owns missing-value policy for its
        # numeric columns (short-gap interpolation, long-gap preservation), so the
        # generic statistical imputer must not pre-fill them.
        ts_skip = (set(self._ts.numeric_targets(cleaned, self._roles))
                   if self._ts is not None else set())
        # Context-protected columns are never touched by the statistical imputer
        # either (the representation pass already byte-guards them); leave their
        # missing cells as-is so "never modify" holds across the whole stream.
        skip = ts_skip | set(self._protected)

        warmup = self.state.batch_count <= self.scfg.warmup_batches
        if warmup:
            self._defer_imputation(cleaned, report, skip=skip)
        else:
            cleaned = self._impute(cleaned, report, skip=skip)

        ts_active = self._ts is not None
        ts_summary: dict[str, Any] = {}
        if self._ts is not None:
            cleaned, exceptions = self._ts.process(cleaned, report, roles=self._roles)
            ts_summary = dict(self._ts.last_summary)
            self.last_exceptions_ = exceptions
            if len(exceptions):
                self._exception_batches.append(exceptions)

        trust = compute_trust_score(cleaned, config=self.config).overall
        rolling, cumulative = self.state.record_trust(trust, len(cleaned))
        gate_passed = (self.scfg.fail_under_trust is None
                       or trust >= self.scfg.fail_under_trust)
        if not gate_passed:
            self._gate_failures += 1

        report.streaming = {
            "batch_id": self.state.batch_count,
            "rows_in_batch": len(cleaned),
            "rows_seen_total": self.state.rows_seen,
            "batch_trust_score": round(trust, 2),
            "rolling_trust_score": round(rolling, 2),
            "cumulative_trust_score": round(cumulative, 2),
            "schema_drift_detected": bool(findings),
            "warmup_phase": warmup,
            "trust_gate_passed": gate_passed,
        }
        if ts_active:
            report.streaming["time_series"] = ts_summary
        if self._policy is not None:
            report.streaming["context_policy"] = self._policy_summary()
        self.report_ = report
        return cleaned, report

    def clean_batches(self, batches: Iterable[Any]) -> Iterator[tuple[pd.DataFrame, CleanReport]]:
        """Clean each batch from any iterable, yielding ``(cleaned_df, report)``."""
        for batch in batches:
            yield self.clean_batch(batch)

    def clean_arrow_batches(self, source: Any) -> Iterator[tuple[pd.DataFrame, CleanReport]]:
        """Clean a pyarrow ``RecordBatchReader`` / iterable of RecordBatch/Table."""
        return self.clean_batches(source)

    def clean_kafka(self, *, topic: str, bootstrap_servers: str,
                    value_format: str = "json", batch_size: int = 10_000,
                    max_batches: int | None = None, consumer_kwargs: dict[str, Any] | None = None
                    ) -> Iterator[tuple[pd.DataFrame, CleanReport]]:
        """Consume *topic* in micro-batches and clean each (optional dependency)."""
        from ._connectors import kafka_batches

        return self.clean_batches(kafka_batches(
            topic=topic, bootstrap_servers=bootstrap_servers, value_format=value_format,
            batch_size=batch_size, max_batches=max_batches,
            consumer_kwargs=consumer_kwargs,
        ))

    def clean_arrow_flight(self, location: str, *, descriptor: Any = None, batch_size: int = 10_000
                           ) -> Iterator[tuple[pd.DataFrame, CleanReport]]:
        """Stream an Arrow Flight endpoint in micro-batches (optional dependency)."""
        from ._connectors import flight_batches

        return self.clean_batches(flight_batches(
            location, descriptor=descriptor, batch_size=batch_size,
        ))

    def finalize(self) -> CleanReport:
        """Build the cumulative report summarizing every batch processed."""
        rep = CleanReport(rows_before=self.state.rows_seen, rows_after=self.state.rows_seen,
                          cols_before=len(self.state.schema_baseline),
                          cols_after=len(self.state.columns))
        rep.missing_before = sum(c.missing for c in self.state.columns.values())
        for col, cs in self.state.columns.items():
            if cs.missing:
                rep.add("missing", f"{cs.missing} missing cell(s) seen across stream",
                        column=col, count=cs.missing, rationale=f"role={cs.role}")
        for f in self.state.drift_log:
            rep.add_warning(f"drift [batch {f['batch']}]: {f['message']}")
        rep.streaming = {
            "batches": self.state.batch_count,
            "rows_seen_total": self.state.rows_seen,
            "cells_imputed": self._n_imputed,
            "decisions_deferred_warmup": self._n_deferred,
            "rolling_trust_score": _round(self.state.rolling_trust_score),
            "cumulative_trust_score": _round(self.state.cumulative_trust_score),
            "trust_gate_failures": self._gate_failures,
            "drift_events": len(self.state.drift_log),
        }
        if self._ts is not None:
            rep.streaming["time_series"] = {
                "watermark": self._ts.watermark.isoformat()
                if self._ts.watermark is not None else None,
                "late_quarantined_total": self._ts.late_quarantined_total,
                "late_dropped_total": self._ts.late_dropped_total,
                "anomalies_flagged_total": self._ts.anomalies_flagged_total,
                "anomalies_quarantined_total": self._ts.anomalies_quarantined_total,
                "exceptions_total": sum(len(b) for b in self._exception_batches),
            }
        if self._policy is not None:
            rep.streaming["context_policy"] = self._policy_summary()
        return rep

    @property
    def state_(self) -> dict[str, Any]:
        """JSON-friendly snapshot of the running state."""
        return self.state.to_dict()

    @property
    def exceptions_(self) -> pd.DataFrame | None:
        """All rows quarantined by time-series late-data/anomaly handling so far.

        ``None`` when not in time-series mode; an empty frame when nothing was
        quarantined. Each row carries a ``_quarantine_reason`` column.
        """
        if self._ts is None:
            return None
        if not self._exception_batches:
            return self.last_exceptions_
        return pd.concat(self._exception_batches, ignore_index=True)

    @property
    def policy_(self) -> ContextPolicy | None:
        """The compiled context policy governing the stream (``None`` if unset).

        Compiled once against the first batch's schema; the same protected
        columns and constraints apply to every batch. Inspect ``.constraints``,
        ``.protected_columns``, ``.unresolved`` and ``.to_dict()`` for an audit.
        """
        return self._policy

    @property
    def n_rows_seen(self) -> int:
        return self.state.rows_seen

    @property
    def n_batches_seen(self) -> int:
        return self.state.batch_count

    @property
    def is_warmed_up(self) -> bool:
        return self.state.batch_count > self.scfg.warmup_batches

    @property
    def rolling_trust_score(self) -> float | None:
        return self.state.rolling_trust_score

    @property
    def cumulative_trust_score(self) -> float | None:
        return self.state.cumulative_trust_score

    # -- internals --------------------------------------------------------------

    def _build_rep_config(self, cfg: CleanConfig) -> CleanConfig:
        """Derive the representation-only per-batch config from *cfg*.

        Engine off, no impute/outliers, schema-stable across batches. Any
        ``context``/``policy`` is cleared here because it is lowered exactly
        once in :meth:`_ensure_policy` (protected columns already live in
        ``preserve_columns`` / ``semantic_context[mutable=False]``, which the
        representation pass hard-guards) — recompiling it per batch would be
        wasteful and could drift with a batch that is missing a column.
        """
        return dataclasses.replace(
            cfg, strategy="conservative", impute=None, outliers=None,
            drop_empty_columns=False, drop_constant_columns=False,
            verbose=False, reset_index=True, context=None, policy=None,
        )

    def _ensure_policy(self, df: pd.DataFrame) -> None:
        """Compile the context policy once, against the first batch's schema.

        No-op (zero behaviour change) when no ``context``/``policy`` was given.
        Otherwise the text/object is compiled and lowered a single time into
        stable ``preserve_columns`` / ``id_columns`` / ``semantic_context``
        (protected columns marked ``mutable=False``), so every subsequent batch
        is governed by the same decisions rather than being recompiled.
        """
        if not self._policy_pending:
            return
        self._policy_pending = False
        from ..context import apply_policy_to_config  # noqa: PLC0415

        lowered = apply_policy_to_config(self.config, df=df)
        self.config = lowered
        self._policy = getattr(lowered, "policy", None)
        self._protected = (frozenset(self._policy.protected_columns)
                           if self._policy is not None else frozenset())
        self._rep_config = self._build_rep_config(lowered)

    def _policy_summary(self) -> dict[str, Any]:
        """Compact per-report snapshot of the governing context policy."""
        policy = self._policy
        assert policy is not None  # guarded by callers
        return {
            "constraints": len(policy.constraints),
            "protected_columns": list(policy.protected_columns),
            "unresolved": len(policy.unresolved),
            "issues": len(policy.issues),
        }

    def _surface_policy(self, report: CleanReport) -> None:
        """Record the compiled policy on the batch where it was first applied.

        Mirrors ``apply_policy_to_config``'s report block (which is bypassed in
        streaming so the policy is compiled only once): one summary entry, a
        warning per unresolved reference / issue, and a protected-column note.
        """
        policy = self._policy
        if policy is None:
            return
        report.add(
            step="context",
            description=f"compiled context policy: {len(policy.constraints)} "
                        f"constraint(s), {len(policy.unresolved)} unresolved, "
                        f"{len(policy.issues)} issue(s)",
            rationale="deterministic tier-0 context compiler (streaming: compiled "
                      "once against the first batch, applied to every batch)",
            metadata={"policy": policy.to_dict()},
        )
        for u in policy.unresolved:
            cands = ", ".join(f"{c} ({s:.2f})" for c, s in u.candidates[:3])
            report.add_warning(
                f"[context] unresolved column reference {u.ref!r}: {u.reason}"
                + (f" — candidates: {cands}" if cands else "")
            )
        for issue in policy.issues:
            report.add_warning(f"[context] {issue.kind}: {issue.message}")
        for col in policy.protected_columns:
            report.add(
                step="context",
                description=f"column {col!r} protected by policy "
                            "(never modified across the stream)",
                column=col,
                rationale="context policy: never_modify",
                metadata={"enforcement": "hard"},
            )

    def _lock_roles(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            name = str(col)
            if name not in self._roles:
                self._roles[name] = infer_role(name, df[col], self.config)

    def _dedup_window(self, df: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
        # ponytail: bounded recent-window dedup (cap = window_size), not true global.
        hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
        keep = [h not in self._seen_hashes for h in hashes]
        for h, k in zip(hashes, keep):
            if k and len(self._seen_hashes) < self.scfg.window_size:
                self._seen_hashes.add(int(h))
        removed = len(df) - sum(keep)
        if removed:
            report.add("duplicates", f"removed {removed} cross-batch duplicate row(s) "
                       "(recent-window)", count=removed)
            report.duplicates_removed += removed
            df = df.loc[keep].reset_index(drop=True)
        return df

    def _defer_imputation(self, df: pd.DataFrame, report: CleanReport,
                          *, skip: set[str] | None = None) -> None:
        skip = skip or set()
        k, n = self.state.batch_count, self.scfg.warmup_batches
        for col in df.columns:
            if str(col) in skip:
                continue
            miss = int(df[col].isna().sum())
            if miss:
                self._n_deferred += 1
                report.add("missing", f"deferred {miss} fill(s): collecting statistics "
                           f"(warmup batch {k}/{n})", column=str(col), count=0,
                           rationale="not enough global statistics yet to impute safely",
                           confidence=0.5)

    def _impute(self, df: pd.DataFrame, report: CleanReport,
                *, skip: set[str] | None = None) -> pd.DataFrame:
        skip = skip or set()
        for col in list(df.columns):
            name = str(col)
            # Skip TS-managed columns and any column without running state (e.g. a
            # flag column a later step may add); never KeyError on state lookup.
            if name in skip or name not in self.state.columns:
                continue
            miss = int(df[col].isna().sum())
            if miss == 0:
                continue
            df = self._impute_column(df, name, miss, report)
        return df

    def _impute_column(self, df: pd.DataFrame, col: str, miss: int,
                       report: CleanReport) -> pd.DataFrame:
        cs = self.state.columns[col]
        role, ratio = cs.role, cs.missing_ratio
        band = _band(ratio, self.config)
        pct = f"{100 * ratio:.1f}%"

        if role in ("target", "id", "text"):
            return self._preserve(df, col, miss, report,
                                  rationale=f"{role} column — never auto-filled (streaming "
                                            "safety gate)",
                                  risk="medium" if role != "text" else "low")
        if band in ("high", "extreme"):  # balanced-mode behavior: keep + warn
            report.add_warning(f"column '{col}' has {pct} missing (cumulative); preserved "
                               "in streaming balanced mode")
            return self._preserve(df, col, miss, report,
                                  rationale=f"{pct} missing — preserved instead of "
                                            "force-filling", risk="high", confidence=0.7)

        if role == "datetime":
            if not cs.datetime_ordered:
                return self._preserve(df, col, miss, report,
                                      rationale="datetime without a usable order; fill would "
                                                "invent timestamps", risk="medium")
            df[col] = df[col].ffill().bfill()
            return self._record(df, report, col, miss, "forward/backward fill within batch",
                                rationale="datetime column with monotonic order", confidence=0.8)

        if role == "numeric":
            snap = cs.numeric_snapshot()
            skewed = snap.skew_approx is not None and abs(snap.skew_approx) >= 0.5
            if skewed or snap.has_outliers:
                value, label = snap.median, "running median"
                rationale = "skewed/outlier-bearing distribution; running median is robust"
            else:
                value, label = snap.mean, "running mean"
                rationale = "approximately symmetric distribution; running mean"
            if value is None or pd.isna(value):
                return self._preserve(df, col, miss, report,
                                      rationale="no running statistic available yet",
                                      risk="medium", confidence=0.5)
            df[col] = (df[col].astype("float64") if df[col].dtype.kind in "iu"
                       else df[col]).fillna(value)
            return self._record(df, report, col, miss, f"{label} ({value:.6g})",
                                rationale=rationale, confidence=0.8 if band == "low" else 0.7)

        # categorical / boolean
        mode, mode_ratio = cs.mode(), cs.mode_ratio()
        threshold = 0.5 if band == "low" else 0.6
        if mode is not None and mode_ratio is not None and mode_ratio >= threshold:
            df[col] = df[col].fillna(mode)
            return self._record(df, report, col, miss, f"running mode ({mode!r})",
                                rationale=f"dominant category ({100 * mode_ratio:.0f}% of seen)",
                                confidence=0.8 if band == "low" else 0.7)
        sentinel = "Unknown" if band == "low" else "Missing"
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype) and sentinel not in s.cat.categories:
            s = s.cat.add_categories([sentinel])
        df[col] = s.fillna(sentinel)
        return self._record(df, report, col, miss, f'sentinel "{sentinel}"',
                            rationale="no dominant category; sentinel keeps the gap visible",
                            confidence=0.7, risk="low" if band == "low" else "medium")

    def _record(self, df: pd.DataFrame, report: CleanReport, col: str, miss: int, label: str, *,
                rationale: str, confidence: float, risk: str = "low") -> pd.DataFrame:
        self._n_imputed += miss
        report.add("missing", f"filled {miss} missing value(s) with {label}",
                   column=col, count=miss, rationale=rationale, risk=risk,
                   confidence=confidence, model_id=label.split(maxsplit=1)[0])
        report.columns_imputed.append(col)
        return df

    def _preserve(self, df: pd.DataFrame, col: str, miss: int, report: CleanReport, *,
                  rationale: str, risk: str = "low", confidence: float = 1.0) -> pd.DataFrame:
        report.add("missing", f"preserved {miss} missing value(s)", column=col, count=0,
                   rationale=rationale, risk=risk, confidence=confidence, model_id="preserve")
        report.columns_preserved.append(col)
        return df


def _round(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None
