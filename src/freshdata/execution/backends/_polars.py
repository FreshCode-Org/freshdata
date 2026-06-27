"""Polars backend — native out-of-core cleaning on LazyFrames.

Reproduces freshdata's deterministic representation-repair + structural-reduction
subset (column rename, whitespace/sentinel normalization, empty column/row drops,
full-row dedup) directly on a ``pl.LazyFrame`` with projection/predicate pushdown
and streaming collection. Steps outside that subset (the decision engine,
heuristic dtype repair, opt-in impute/outliers) transparently fall back to the
pandas pipeline, so output stays identical to ``fd.clean``.

Action records use the *same* ``step`` names and ``count`` semantics the pandas
``steps/`` modules emit, so ``CleanReport`` consumers are unaffected.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .._base import ExecutionEngine
from .._lazy import has_polars, require_polars
from .._metadata import MetadataScanner
from .._native_steps import (
    impute_defined_for,
    integer_safe_bounds,
    iqr_bounds,
    native_outlier_factor,
    native_outlier_method,
    outlier_label,
    resolve_impute_strategy,
    zscore_bounds,
)
from .._plan import NativePlan, PlanGenerator
from .._report import finalize_report, init_report
from ._pandas import materialize_to_pandas

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config import CleanConfig
    from ...report import CleanReport
    from .._config import EngineConfig

log = logging.getLogger("freshdata.execution.polars")


class PolarsEngine(ExecutionEngine):
    name = "polars"

    def supports_source(self, source: Any) -> bool:
        if isinstance(source, str):
            return True
        import pandas as pd

        if isinstance(source, pd.DataFrame):
            return True
        try:
            import pyarrow as pa

            if isinstance(source, (pa.Table, pa.RecordBatch)):
                return True
        except ImportError:
            pass
        if has_polars():
            import polars as pl

            return isinstance(source, (pl.DataFrame, pl.LazyFrame))
        return False

    # -- source ingestion ---------------------------------------------------

    def _to_lazy(self, source: Any, pl: Any) -> tuple[Any, int]:
        """Return ``(LazyFrame, memory_before_bytes)`` for *source*."""
        if isinstance(source, pl.LazyFrame):
            return source, 0
        if isinstance(source, pl.DataFrame):
            return source.lazy(), int(source.estimated_size())
        if isinstance(source, str):
            low = source.lower()
            if low.endswith((".parquet", ".pq")):
                return pl.scan_parquet(source), 0
            if low.endswith(".csv"):
                return pl.scan_csv(source), 0
            if low.endswith((".ipc", ".feather", ".arrow")):
                return pl.scan_ipc(source), 0
            raise ValueError(f"PolarsEngine: unsupported file type for path {source!r}")
        try:
            import pyarrow as pa

            if isinstance(source, pa.Table):
                return pl.from_arrow(source).lazy(), int(source.nbytes)
            if isinstance(source, pa.RecordBatch):
                table = pa.Table.from_batches([source])
                return pl.from_arrow(table).lazy(), int(table.nbytes)
        except ImportError:
            pass
        import pandas as pd

        if isinstance(source, pd.DataFrame):
            return pl.from_pandas(source).lazy(), int(source.memory_usage(deep=True).sum())
        raise TypeError(f"PolarsEngine: unsupported source type {type(source).__name__}")

    def _pandas_index_forces_fallback(self, source: Any) -> bool:
        """A non-default pandas index (e.g. DatetimeIndex) has index-aware
        dedup semantics that a polars frame cannot carry — fall back."""
        import pandas as pd

        return isinstance(source, pd.DataFrame) and not isinstance(source.index, pd.RangeIndex)

    # -- execution ----------------------------------------------------------

    def execute(
        self,
        source: Any,
        config: CleanConfig,
        engine_config: EngineConfig,
    ) -> tuple[Any, CleanReport]:
        pl = require_polars()
        self._configure_threads(engine_config)
        started = time.perf_counter()

        lf, memory_before = self._to_lazy(source, pl)
        names = list(lf.collect_schema().names())
        plan = PlanGenerator(config).plan(names)

        if plan.needs_fallback or self._pandas_index_forces_fallback(source):
            reason = plan.fallback_reason or "pandas index semantics"
            log.warning("freshdata PolarsEngine: falling back to pandas (%s)", reason)
            cleaned, report = self._fallback(source, config)
            report.backend = "pandas"
            report.record_fallback("polars", "pipeline", reason)
            return cleaned, report

        meta = MetadataScanner.from_polars_lazy(lf)
        report = init_report(meta, memory_before)
        report.backend = "polars"
        lf = self._apply_native(lf, plan, config, report, pl)
        cleaned = self._collect(lf, engine_config, pl)
        finalize_report(report, cleaned, started)
        return cleaned, report

    def _fallback(self, source: Any, config: CleanConfig) -> tuple[Any, CleanReport]:
        from ...cleaner import run_pipeline

        df = materialize_to_pandas(source)
        return run_pipeline(df, config)

    # -- native stages (mirror cleaner.run_pipeline order) ------------------

    def _apply_native(
        self, lf: Any, plan: NativePlan, config: CleanConfig, report: CleanReport, pl: Any
    ) -> Any:
        rows_before = report.rows_before
        for stage in plan.stages:
            if stage == "column_names":
                lf = self._stage_rename(lf, plan, report)
            elif stage == "clean_strings":
                lf = self._stage_clean_strings(lf, config, report, pl)
            elif stage == "drop_empty_columns":
                if rows_before > 0:
                    lf = self._stage_drop_empty_columns(lf, report, pl)
            elif stage == "drop_empty_rows":
                if rows_before > 0:
                    lf = self._stage_drop_empty_rows(lf, report, pl)
            elif stage == "drop_duplicates":
                lf = self._stage_drop_duplicates(lf, config, report, pl)
            elif stage == "impute":
                lf = self._stage_impute(lf, config, report, pl)
            elif stage == "outliers":
                lf = self._stage_outliers(lf, config, report, pl)
            elif stage == "reset_index":
                pass  # polars frames carry no index
        return lf

    def _integer_dtypes(self, pl: Any) -> tuple[Any, ...]:
        return (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)

    def _stage_impute(
        self, lf: Any, config: CleanConfig, report: CleanReport, pl: Any
    ) -> Any:
        strategy = config.impute
        if strategy is None:
            return lf
        report.record_backend_difference(
            "polars", "impute",
            "median/mode fill values are computed with polars aggregates and may "
            "differ from the pandas reference's interpolation/tie-breaking",
        )
        schema = lf.collect_schema()
        int_dtypes = self._integer_dtypes(pl)

        aggs: list[Any] = [pl.len().alias("__h__")]
        cols: list[tuple[str, bool, str]] = []
        for name in schema.names():
            dtype = schema[name]
            is_numeric = dtype.is_numeric() and dtype != pl.Boolean
            resolved = resolve_impute_strategy(strategy, is_numeric=is_numeric)
            aggs.append(pl.col(name).null_count().alias(f"__n__{name}"))
            if impute_defined_for(strategy, is_numeric=is_numeric):
                if resolved == "mean":
                    value_expr = pl.col(name).mean()
                elif resolved == "median":
                    value_expr = pl.col(name).median()
                else:
                    value_expr = pl.col(name).drop_nulls().mode().first()
                aggs.append(value_expr.alias(f"__v__{name}"))
            cols.append((name, is_numeric, resolved))

        row = lf.select(aggs).collect().row(0, named=True)
        height = int(row["__h__"])
        fills: list[Any] = []
        for name, is_numeric, resolved in cols:
            n_missing = int(row[f"__n__{name}"])
            if n_missing == 0 or height - n_missing == 0:
                continue
            if not impute_defined_for(strategy, is_numeric=is_numeric):
                report.add("impute", f"skipped ({strategy} is not defined for {name})",
                           column=name)
                continue
            value = row.get(f"__v__{name}")
            if value is None:
                report.add("impute", f"skipped (could not compute {resolved} for {name})",
                           column=name)
                continue
            expr = pl.col(name)
            if isinstance(value, float) and schema[name] in int_dtypes:
                expr = expr.cast(pl.Float64)
            fills.append(expr.fill_null(pl.lit(value)).alias(name))
            shown = f"{value:.6g}" if isinstance(value, float) else repr(value)
            report.add("impute",
                       f"filled {n_missing} missing value(s) with {resolved} ({shown})",
                       column=name, count=n_missing)
            report.columns_imputed.append(name)
        return lf.with_columns(fills) if fills else lf

    def _stage_outliers(
        self, lf: Any, config: CleanConfig, report: CleanReport, pl: Any
    ) -> Any:
        if config.outliers is None:
            return lf
        method = native_outlier_method(config)
        factor = native_outlier_factor(config, method)
        schema = lf.collect_schema()
        numeric = [n for n in schema.names()
                   if schema[n].is_numeric() and schema[n] != pl.Boolean]
        if not numeric:
            return lf
        report.record_backend_difference(
            "polars", "outliers",
            "outlier counts depend on polars quantile/stddev statistics and may "
            "differ from the pandas reference",
        )
        int_dtypes = self._integer_dtypes(pl)

        stat_aggs: list[Any] = []
        for n in numeric:
            if method == "iqr":
                stat_aggs.append(pl.col(n).quantile(0.25, "linear").alias(f"q1_{n}"))
                stat_aggs.append(pl.col(n).quantile(0.75, "linear").alias(f"q3_{n}"))
            else:
                stat_aggs.append(pl.col(n).mean().alias(f"m_{n}"))
                stat_aggs.append(pl.col(n).std().alias(f"s_{n}"))
        stats = lf.select(stat_aggs).collect().row(0, named=True)

        bounds: dict[str, tuple[float, float]] = {}
        for n in numeric:
            if method == "iqr":
                q1, q3 = stats[f"q1_{n}"], stats[f"q3_{n}"]
                raw = None if q1 is None or q3 is None else iqr_bounds(q1, q3, factor)
            else:
                m, s = stats[f"m_{n}"], stats[f"s_{n}"]
                raw = None if m is None or s is None else zscore_bounds(m, s, factor)
            if raw is None:
                continue
            bounds[n] = integer_safe_bounds(*raw, is_integer=schema[n] in int_dtypes)

        if not bounds:
            return lf
        count_aggs = [
            ((pl.col(n) < lo) | (pl.col(n) > hi)).sum().alias(f"c_{n}")
            for n, (lo, hi) in bounds.items()
        ]
        counts = lf.select(count_aggs).collect().row(0, named=True)

        transforms: list[Any] = []
        for n, (lo, hi) in bounds.items():
            n_out = int(counts.get(f"c_{n}", 0) or 0)
            if n_out == 0:
                continue
            label = outlier_label(method, factor)
            if config.outliers == "clip":
                transforms.append(pl.col(n).clip(lo, hi).alias(n))
                report.add("outliers",
                           f"clipped {n_out} outlier(s) to [{lo:g}, {hi:g}] ({label})",
                           column=n, count=n_out)
            else:
                flag = self._unique_flag(schema.names(), f"{n}_outlier")
                mask = ((pl.col(n) < lo) | (pl.col(n) > hi)).fill_null(False)
                transforms.append(mask.alias(flag))
                report.add("outliers",
                           f"flagged {n_out} outlier(s) in new column {flag!r} ({label})",
                           column=n, count=n_out)
            report.outliers_handled += n_out
        return lf.with_columns(transforms) if transforms else lf

    @staticmethod
    def _unique_flag(existing: Any, base: str) -> str:
        names = set(existing)
        name, k = base, 1
        while name in names:
            k += 1
            name = f"{base}_{k}"
        return name

    def _stage_rename(self, lf: Any, plan: NativePlan, report: CleanReport) -> Any:
        if not plan.rename_map:
            return lf
        changes = list(plan.rename_map.items())
        preview = ", ".join(f"{o!r}->{n!r}" for o, n in changes[:4])
        if len(changes) > 4:
            preview += f", … (+{len(changes) - 4} more)"
        report.add("column_names", f"renamed {len(changes)} column(s): {preview}",
                   count=len(changes))
        return lf.rename(plan.rename_map)

    def _stage_clean_strings(
        self, lf: Any, config: CleanConfig, report: CleanReport, pl: Any
    ) -> Any:
        from ...steps.strings import active_sentinels

        sentinels = list(active_sentinels(config))
        schema = lf.collect_schema()
        string_cols = [n for n in schema.names() if schema[n] == pl.Utf8]
        if not string_cols:
            return lf

        # One aggregate pass for all strip / sentinel counts.
        count_exprs: list[Any] = []
        for c in string_cols:
            col = pl.col(c)
            stripped = col.str.strip_chars()
            base = stripped if config.strip_whitespace else col
            if config.strip_whitespace:
                count_exprs.append(
                    ((stripped != col) & col.is_not_null()).sum().alias(f"__strip__{c}")
                )
            if config.normalize_sentinels:
                count_exprs.append(
                    (base.str.to_lowercase().is_in(sentinels) & base.is_not_null())
                    .sum()
                    .alias(f"__sent__{c}")
                )
        counts = lf.select(count_exprs).collect().row(0, named=True) if count_exprs else {}

        transforms: list[Any] = []
        for c in string_cols:
            col = pl.col(c)
            stripped = col.str.strip_chars()
            base = stripped if config.strip_whitespace else col
            n_strip = int(counts.get(f"__strip__{c}", 0) or 0)
            n_sent = int(counts.get(f"__sent__{c}", 0) or 0)
            if config.strip_whitespace and n_strip:
                report.add("strip_whitespace", "trimmed surrounding whitespace",
                           column=c, count=n_strip)
            if config.normalize_sentinels and n_sent:
                report.add("normalize_sentinels",
                           'replaced sentinel strings ("N/A", "-", "", …) with missing',
                           column=c, count=n_sent)
            if config.normalize_sentinels:
                expr = (
                    pl.when(base.str.to_lowercase().is_in(sentinels))
                    .then(None)
                    .otherwise(base)
                    .alias(c)
                )
            else:
                expr = base.alias(c)
            transforms.append(expr)
        return lf.with_columns(transforms)

    def _stage_drop_empty_columns(self, lf: Any, report: CleanReport, pl: Any) -> Any:
        schema = lf.collect_schema()
        names = list(schema.names())
        stats = lf.select(
            [pl.len().alias("__h__")] + [pl.col(c).null_count().alias(c) for c in names]
        ).collect().row(0, named=True)
        height = int(stats["__h__"])
        dropped = [c for c in names if int(stats[c]) == height]
        if dropped:
            report.columns_dropped.extend(dropped)
            report.add(
                "drop_empty_columns",
                f"dropped {len(dropped)} all-missing column(s): {', '.join(dropped[:6])}"
                + (" …" if len(dropped) > 6 else ""),
                count=len(dropped),
            )
            lf = lf.drop(dropped)
        return lf

    def _stage_drop_empty_rows(self, lf: Any, report: CleanReport, pl: Any) -> Any:
        if not lf.collect_schema().names():
            return lf
        all_null = pl.all_horizontal(pl.all().is_null())
        n = int(lf.select(all_null.sum().alias("__n__")).collect().item())
        if n:
            report.add("drop_empty_rows", f"dropped {n} all-missing row(s)", count=n)
            lf = lf.filter(~all_null)
        return lf

    def _stage_drop_duplicates(
        self, lf: Any, config: CleanConfig, report: CleanReport, pl: Any
    ) -> Any:
        n_before = int(lf.select(pl.len()).collect().item())
        if n_before < 1:
            return lf
        deduped = lf.unique(keep=config.duplicate_keep, maintain_order=True)
        n_after = int(deduped.select(pl.len()).collect().item())
        n_dup = n_before - n_after
        if n_dup <= 0:
            return lf
        pct = 100.0 * n_dup / n_before
        risk = "medium" if (n_dup / n_before) > config.duplicate_threshold else "low"
        report.add(
            "drop_duplicates",
            f"dropped {n_dup} duplicate row(s) "
            f"({pct:.1f}% of rows, keep={config.duplicate_keep!r})",
            count=n_dup,
            risk=risk,
        )
        report.duplicates_removed += n_dup
        if risk == "medium":
            report.add_warning(
                f"{pct:.1f}% of rows were duplicates "
                f"(> {100 * config.duplicate_threshold:.0f}%); confirm they are not legitimate"
            )
        return deduped

    # -- collection ---------------------------------------------------------

    def _collect(self, lf: Any, engine_config: EngineConfig, pl: Any) -> Any:
        if not engine_config.streaming:
            return lf.collect()
        # Polars renamed the streaming switch across versions; try the modern
        # keyword first, then the legacy one, then a plain collect.
        try:
            return lf.collect(engine="streaming")
        except TypeError:
            pass
        try:
            return lf.collect(streaming=True)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("freshdata: streaming collect failed (%s); retrying eagerly", exc)
            return lf.collect()

    def _configure_threads(self, engine_config: EngineConfig) -> None:
        if engine_config.polars_n_threads is not None:
            import os

            # Only effective if set before polars is first imported; harmless otherwise.
            os.environ.setdefault("POLARS_MAX_THREADS", str(engine_config.polars_n_threads))
