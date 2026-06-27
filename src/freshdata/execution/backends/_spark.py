"""Spark backend — native distributed cleaning on PySpark DataFrames.

Reproduces freshdata's deterministic representation-repair + structural-reduction
subset (column rename, whitespace/sentinel normalization, empty column/row drops,
full-row dedup) plus the opt-in ``impute``/``outliers`` overrides directly on a
``pyspark.sql.DataFrame`` so a clean can run on a Spark cluster. Steps outside
that subset (the decision engine, heuristic dtype repair, subset dedup)
transparently fall back to the pandas pipeline.

Action records use the *same* ``step`` names and ``count`` semantics the pandas
``steps/`` modules emit, so ``CleanReport`` consumers are unaffected. pyspark is
an optional dependency, imported lazily; importing freshdata never requires it.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .._base import ExecutionEngine
from .._lazy import has_polars, has_pyspark, require_pyspark
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
from ._pandas import materialize_to_pandas

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config import CleanConfig
    from ...report import CleanReport
    from .._config import EngineConfig

log = logging.getLogger("freshdata.execution.spark")


def _is_spark_frame(source: Any) -> bool:
    if not has_pyspark():
        return False
    from pyspark.sql import DataFrame as SparkDataFrame

    return isinstance(source, SparkDataFrame)


class SparkEngine(ExecutionEngine):
    name = "spark"

    def supports_source(self, source: Any) -> bool:
        if _is_spark_frame(source):
            return True
        if isinstance(source, str):
            return True
        import pandas as pd

        if isinstance(source, pd.DataFrame):
            return True
        if has_polars():
            import polars as pl

            return isinstance(source, (pl.DataFrame, pl.LazyFrame))
        return False

    # -- session / ingestion ------------------------------------------------

    def _session(self, engine_config: EngineConfig, source: Any) -> Any:
        sql = require_pyspark()
        session = engine_config.spark_session
        if session is None and _is_spark_frame(source):
            session = source.sparkSession
        if session is None:
            session = sql.SparkSession.builder.appName("freshdata").getOrCreate()
        if engine_config.spark_shuffle_partitions is not None:
            session.conf.set(
                "spark.sql.shuffle.partitions", str(engine_config.spark_shuffle_partitions)
            )
        return session

    def _to_spark(self, source: Any, session: Any) -> Any:
        """Return a Spark DataFrame for *source*."""
        if _is_spark_frame(source):
            return source
        if isinstance(source, str):
            low = source.lower()
            if low.endswith((".parquet", ".pq")):
                return session.read.parquet(source)
            if low.endswith(".csv"):
                return session.read.option("header", True).option(
                    "inferSchema", True
                ).csv(source)
            raise ValueError(f"SparkEngine: unsupported file type for path {source!r}")
        return session.createDataFrame(materialize_to_pandas(source))

    def _pandas_index_forces_fallback(self, source: Any) -> bool:
        import pandas as pd

        return isinstance(source, pd.DataFrame) and not isinstance(source.index, pd.RangeIndex)

    # -- execution ----------------------------------------------------------

    def execute(
        self,
        source: Any,
        config: CleanConfig,
        engine_config: EngineConfig,
    ) -> tuple[Any, CleanReport]:
        from ...report import CleanReport

        session = self._session(engine_config, source)
        sdf = self._to_spark(source, session)
        names = list(sdf.columns)
        plan = PlanGenerator(config).plan(names)

        if plan.needs_fallback or self._pandas_index_forces_fallback(source):
            reason = plan.fallback_reason or "pandas index semantics"
            log.warning("freshdata SparkEngine: falling back to pandas (%s)", reason)
            cleaned, report = self._fallback(source, config)
            report.backend = "pandas"
            report.record_fallback("spark", "pipeline", reason)
            return cleaned, report

        started = time.perf_counter()
        report = CleanReport(backend="spark")
        self._init_before(sdf, report)
        sdf = self._apply_native(sdf, plan, config, report)
        self._finalize(sdf, report, started)
        return sdf, report

    def _fallback(self, source: Any, config: CleanConfig) -> tuple[Any, CleanReport]:
        from ...cleaner import run_pipeline

        return run_pipeline(materialize_to_pandas(source), config)

    # -- report bookkeeping -------------------------------------------------

    def _null_count_total(self, sdf: Any) -> int:
        from pyspark.sql import functions as F

        if not sdf.columns:
            return 0
        exprs = [
            F.sum(F.col(c).isNull().cast("long")).alias(c) for c in sdf.columns
        ]
        row = sdf.select(exprs).first()
        return int(sum(int(row[c] or 0) for c in sdf.columns)) if row else 0

    def _init_before(self, sdf: Any, report: CleanReport) -> None:
        report.rows_before = sdf.count()
        report.cols_before = len(sdf.columns)
        report.missing_before = self._null_count_total(sdf)
        report.memory_before = 0

    def _finalize(self, sdf: Any, report: CleanReport, started: float) -> None:
        report.rows_after = sdf.count()
        report.cols_after = len(sdf.columns)
        report.missing_after = self._null_count_total(sdf)
        report.memory_after = 0
        report.duration_seconds = time.perf_counter() - started

    # -- native stages (mirror cleaner.run_pipeline order) ------------------

    def _string_columns(self, sdf: Any) -> list[str]:
        return [name for name, dtype in sdf.dtypes if dtype == "string"]

    def _numeric_columns(self, sdf: Any) -> list[str]:
        numeric = ("tinyint", "smallint", "int", "bigint", "float", "double", "decimal")
        return [
            name for name, dtype in sdf.dtypes
            if dtype.startswith(numeric) and dtype != "boolean"
        ]

    def _is_integer_column(self, sdf: Any, name: str) -> bool:
        dtype = dict(sdf.dtypes)[name]
        return dtype.startswith(("tinyint", "smallint", "int", "bigint"))

    def _apply_native(
        self, sdf: Any, plan: NativePlan, config: CleanConfig, report: CleanReport
    ) -> Any:
        rows_before = report.rows_before
        for stage in plan.stages:
            if stage == "column_names":
                sdf = self._stage_rename(sdf, plan, report)
            elif stage == "clean_strings":
                sdf = self._stage_clean_strings(sdf, config, report)
            elif stage == "drop_empty_columns" and rows_before > 0:
                sdf = self._stage_drop_empty_columns(sdf, report)
            elif stage == "drop_empty_rows" and rows_before > 0:
                sdf = self._stage_drop_empty_rows(sdf, report)
            elif stage == "drop_duplicates":
                sdf = self._stage_drop_duplicates(sdf, config, report)
            elif stage == "impute":
                sdf = self._stage_impute(sdf, config, report)
            elif stage == "outliers":
                sdf = self._stage_outliers(sdf, config, report)
            elif stage == "reset_index":
                pass  # spark frames carry no index
        return sdf

    def _stage_rename(self, sdf: Any, plan: NativePlan, report: CleanReport) -> Any:
        if not plan.rename_map:
            return sdf
        changes = list(plan.rename_map.items())
        preview = ", ".join(f"{o!r}->{n!r}" for o, n in changes[:4])
        if len(changes) > 4:
            preview += f", … (+{len(changes) - 4} more)"
        report.add("column_names", f"renamed {len(changes)} column(s): {preview}",
                   count=len(changes))
        for old, new in changes:
            sdf = sdf.withColumnRenamed(old, new)
        return sdf

    def _stage_clean_strings(
        self, sdf: Any, config: CleanConfig, report: CleanReport
    ) -> Any:
        from pyspark.sql import functions as F

        from ...steps.strings import active_sentinels

        sentinels = list(active_sentinels(config))
        string_cols = self._string_columns(sdf)
        if not string_cols:
            return sdf

        count_exprs = []
        for c in string_cols:
            col = F.col(c)
            stripped = F.regexp_replace(col, r"^\s+|\s+$", "")
            base = stripped if config.strip_whitespace else col
            if config.strip_whitespace:
                count_exprs.append(
                    F.sum(((stripped != col) & col.isNotNull()).cast("long")).alias(f"s_{c}")
                )
            if config.normalize_sentinels:
                count_exprs.append(
                    F.sum(
                        (F.lower(base).isin(sentinels) & base.isNotNull()).cast("long")
                    ).alias(f"z_{c}")
                )
        counts = sdf.select(count_exprs).first() if count_exprs else None

        for c in string_cols:
            col = F.col(c)
            stripped = F.regexp_replace(col, r"^\s+|\s+$", "")
            base = stripped if config.strip_whitespace else col
            if config.strip_whitespace and counts is not None:
                n_strip = int(counts[f"s_{c}"] or 0)
                if n_strip:
                    report.add("strip_whitespace", "trimmed surrounding whitespace",
                               column=c, count=n_strip)
            if config.normalize_sentinels and counts is not None:
                n_sent = int(counts[f"z_{c}"] or 0)
                if n_sent:
                    report.add("normalize_sentinels",
                               'replaced sentinel strings ("N/A", "-", "", …) with missing',
                               column=c, count=n_sent)
            if config.normalize_sentinels:
                new = F.when(F.lower(base).isin(sentinels), None).otherwise(base)
            else:
                new = base
            sdf = sdf.withColumn(c, new)
        return sdf

    def _stage_drop_empty_columns(self, sdf: Any, report: CleanReport) -> Any:
        from pyspark.sql import functions as F

        names = list(sdf.columns)
        exprs = [F.count(F.col(c)).alias(c) for c in names]
        row = sdf.select(exprs).first()
        dropped = [c for c in names if row is not None and int(row[c]) == 0]
        if dropped:
            report.columns_dropped.extend(dropped)
            report.add(
                "drop_empty_columns",
                f"dropped {len(dropped)} all-missing column(s): {', '.join(dropped[:6])}"
                + (" …" if len(dropped) > 6 else ""),
                count=len(dropped),
            )
            sdf = sdf.drop(*dropped)
        return sdf

    def _stage_drop_empty_rows(self, sdf: Any, report: CleanReport) -> Any:
        from functools import reduce

        from pyspark.sql import functions as F

        if not sdf.columns:
            return sdf
        all_null = reduce(lambda a, b: a & b, (F.col(c).isNull() for c in sdf.columns))
        n = sdf.filter(all_null).count()
        if n:
            report.add("drop_empty_rows", f"dropped {n} all-missing row(s)", count=n)
            sdf = sdf.filter(~all_null)
        return sdf

    def _stage_drop_duplicates(
        self, sdf: Any, config: CleanConfig, report: CleanReport
    ) -> Any:
        n_before = sdf.count()
        if n_before < 1:
            return sdf
        deduped = sdf.dropDuplicates()
        n_after = deduped.count()
        n_dup = n_before - n_after
        if n_dup <= 0:
            return sdf
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

    def _stage_impute(self, sdf: Any, config: CleanConfig, report: CleanReport) -> Any:
        from pyspark.sql import functions as F

        strategy = config.impute
        if strategy is None:
            return sdf
        report.record_backend_difference(
            "spark", "impute",
            "fill values use Spark aggregates; median/mode may differ from the "
            "pandas reference's interpolation/tie-breaking",
        )
        numeric_cols = set(self._numeric_columns(sdf))
        for name in list(sdf.columns):
            is_numeric = name in numeric_cols
            resolved = resolve_impute_strategy(strategy, is_numeric=is_numeric)
            n_missing = sdf.filter(F.col(name).isNull()).count()
            non_null = sdf.filter(F.col(name).isNotNull()).count()
            if n_missing == 0 or non_null == 0:
                continue
            if not impute_defined_for(strategy, is_numeric=is_numeric):
                report.add("impute", f"skipped ({strategy} is not defined for {name})",
                           column=name)
                continue
            value = self._impute_value(sdf, name, resolved, is_numeric)
            if value is None:
                report.add("impute", f"skipped (could not compute {resolved} for {name})",
                           column=name)
                continue
            sdf = sdf.fillna({name: value}) if not is_numeric else sdf.withColumn(
                name, F.when(F.col(name).isNull(), F.lit(value)).otherwise(F.col(name))
            )
            shown = f"{value:.6g}" if isinstance(value, float) else repr(value)
            report.add("impute",
                       f"filled {n_missing} missing value(s) with {resolved} ({shown})",
                       column=name, count=n_missing)
            report.columns_imputed.append(name)
        return sdf

    def _impute_value(self, sdf: Any, name: str, strategy: str, is_numeric: bool) -> Any:
        from pyspark.sql import functions as F

        if strategy == "mean":
            row = sdf.select(F.mean(F.col(name)).alias("v")).first()
            return None if row is None or row["v"] is None else float(row["v"])
        if strategy == "median":
            vals = sdf.approxQuantile(name, [0.5], 0.0)
            return float(vals[0]) if vals else None
        # mode: most frequent non-null value
        row = (
            sdf.filter(F.col(name).isNotNull())
            .groupBy(name).count().orderBy(F.desc("count"), F.col(name))
            .first()
        )
        return None if row is None else row[name]

    def _stage_outliers(self, sdf: Any, config: CleanConfig, report: CleanReport) -> Any:
        from pyspark.sql import functions as F

        if config.outliers is None:
            return sdf
        method = native_outlier_method(config)
        factor = native_outlier_factor(config, method)
        report.record_backend_difference(
            "spark", "outliers",
            "Spark approxQuantile/aggregate statistics may flag a different count "
            "than the pandas reference's interpolated quantiles",
        )
        for name in self._numeric_columns(sdf):
            bounds = self._outlier_bounds(sdf, name, method, factor)
            if bounds is None:
                continue
            lo, hi = integer_safe_bounds(*bounds, is_integer=self._is_integer_column(sdf, name))
            mask = (F.col(name) < F.lit(lo)) | (F.col(name) > F.lit(hi))
            n = sdf.filter(mask).count()
            if n == 0:
                continue
            label = outlier_label(method, factor)
            if config.outliers == "clip":
                clipped = F.when(F.col(name) < F.lit(lo), F.lit(lo)).when(
                    F.col(name) > F.lit(hi), F.lit(hi)
                ).otherwise(F.col(name))
                sdf = sdf.withColumn(name, clipped)
                report.add("outliers", f"clipped {n} outlier(s) to [{lo:g}, {hi:g}] ({label})",
                           column=name, count=n)
            else:
                flag = self._unique_flag(sdf, f"{name}_outlier")
                sdf = sdf.withColumn(flag, F.coalesce(mask, F.lit(False)))
                report.add("outliers",
                           f"flagged {n} outlier(s) in new column {flag!r} ({label})",
                           column=name, count=n)
            report.outliers_handled += n
        return sdf

    def _outlier_bounds(
        self, sdf: Any, name: str, method: str, factor: float
    ) -> tuple[float, float] | None:
        from pyspark.sql import functions as F

        if method == "iqr":
            q = sdf.approxQuantile(name, [0.25, 0.75], 0.0)
            if len(q) < 2:
                return None
            return iqr_bounds(float(q[0]), float(q[1]), factor)
        row = sdf.select(
            F.mean(F.col(name)).alias("m"), F.stddev(F.col(name)).alias("s")
        ).first()
        if row is None or row["m"] is None or row["s"] is None:
            return None
        return zscore_bounds(float(row["m"]), float(row["s"]), factor)

    @staticmethod
    def _unique_flag(sdf: Any, base: str) -> str:
        name, k = base, 1
        existing = set(sdf.columns)
        while name in existing:
            k += 1
            name = f"{base}_{k}"
        return name
