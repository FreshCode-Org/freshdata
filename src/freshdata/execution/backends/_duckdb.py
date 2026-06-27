"""DuckDB backend — native out-of-core cleaning via SQL with spill-to-disk.

Registers the source (Parquet path read in-place, or an in-memory frame via
Arrow) and applies the deterministic representation-repair + structural-reduction
subset as a staged SQL pipeline, letting DuckDB stream and spill to
``temp_directory`` under ``memory_limit``. Steps outside that subset fall back to
the pandas pipeline. The cleaned result is fetched once as a pandas DataFrame.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from .._base import ExecutionEngine
from .._lazy import has_duckdb, has_polars, require_duckdb
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
from .._plan import PlanGenerator
from .._report import finalize_report, init_report
from ._pandas import materialize_to_pandas

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from ...config import CleanConfig
    from ...report import CleanReport
    from .._config import EngineConfig

log = logging.getLogger("freshdata.execution.duckdb")

_TABLE = "freshdata_source"


def _q(name: str) -> str:
    """Quote a SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def _lit(value: str) -> str:
    """Quote a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def _strip_sql(col_sql: str) -> str:
    """Trim leading/trailing whitespace (matches Python ``str.strip`` semantics)."""
    inner = f"regexp_replace(CAST({col_sql} AS VARCHAR), '^\\s+', '', 'g')"
    return f"regexp_replace({inner}, '\\s+$', '', 'g')"


class DuckDBEngine(ExecutionEngine):
    name = "duckdb"

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

            if isinstance(source, (pl.DataFrame, pl.LazyFrame)):
                return True
        if has_duckdb():
            import duckdb

            if isinstance(source, duckdb.DuckDBPyRelation):
                return True
        return False

    # -- execution ----------------------------------------------------------

    def execute(
        self,
        source: Any,
        config: CleanConfig,
        engine_config: EngineConfig,
    ) -> tuple[pd.DataFrame, CleanReport]:
        duckdb = require_duckdb()

        plan_cols = self._peek_columns(source)
        plan = PlanGenerator(config).plan(plan_cols)
        if plan.needs_fallback or self._pandas_index_forces_fallback(source):
            reason = plan.fallback_reason or "pandas index semantics"
            log.warning("freshdata DuckDBEngine: falling back to pandas (%s)", reason)
            cleaned, report = self._fallback(source, config)
            report.backend = "pandas"
            report.record_fallback("duckdb", "pipeline", reason)
            return cleaned, report

        started = time.perf_counter()
        os.makedirs(engine_config.temp_directory, exist_ok=True)
        conn_config: dict[str, Any] = {
            "memory_limit": f"{engine_config.memory_limit_gb}GB",
            "temp_directory": engine_config.temp_directory,
        }
        if engine_config.duckdb_threads is not None:
            conn_config["threads"] = engine_config.duckdb_threads

        conn = duckdb.connect(config=conn_config)
        try:
            self._register_source(conn, source)
            meta = MetadataScanner.from_duckdb(conn, _TABLE)
            report = init_report(meta, self._memory_before(source))
            report.backend = "duckdb"
            cleaned = self._run_sql_pipeline(conn, meta, plan, config, report)
        finally:
            conn.close()

        finalize_report(report, cleaned, started)
        return cleaned, report

    def _fallback(self, source: Any, config: CleanConfig) -> tuple[pd.DataFrame, CleanReport]:
        from ...cleaner import run_pipeline

        df = materialize_to_pandas(source)
        return run_pipeline(df, config)

    # -- source handling ----------------------------------------------------

    def _peek_columns(self, source: Any) -> list[object]:
        import pandas as pd

        if isinstance(source, pd.DataFrame):
            return list(source.columns)
        arrow = self._as_arrow(source)
        if arrow is not None:
            return list(arrow.schema.names)
        if has_polars():
            import polars as pl

            if isinstance(source, pl.LazyFrame):
                return list(source.collect_schema().names())
            if isinstance(source, pl.DataFrame):
                return list(source.columns)
        if isinstance(source, str):
            duckdb = require_duckdb()
            conn = duckdb.connect()
            try:
                self._register_source(conn, source)
                return [r[0] for r in conn.execute(f"DESCRIBE {_TABLE}").fetchall()]
            finally:
                conn.close()
        if has_duckdb():
            import duckdb

            if isinstance(source, duckdb.DuckDBPyRelation):
                return list(source.columns)
        raise TypeError(f"DuckDBEngine: unsupported source type {type(source).__name__}")

    def _pandas_index_forces_fallback(self, source: Any) -> bool:
        import pandas as pd

        return isinstance(source, pd.DataFrame) and not isinstance(source.index, pd.RangeIndex)

    @staticmethod
    def _as_arrow(source: Any) -> Any:
        """Return *source* as a pyarrow Table (zero-copy) or ``None``."""
        try:
            import pyarrow as pa
        except ImportError:
            return None
        if isinstance(source, pa.Table):
            return source
        if isinstance(source, pa.RecordBatch):
            return pa.Table.from_batches([source])
        return None

    def _memory_before(self, source: Any) -> int:
        import pandas as pd

        if isinstance(source, pd.DataFrame):
            return int(source.memory_usage(deep=True).sum())
        if has_polars():
            import polars as pl

            if isinstance(source, pl.DataFrame):
                return int(source.estimated_size())
        return 0

    def _register_source(self, conn: Any, source: Any) -> None:
        import pandas as pd

        if isinstance(source, str):
            low = source.lower()
            if low.endswith((".parquet", ".pq")):
                conn.execute(
                    f"CREATE OR REPLACE VIEW {_TABLE} AS "
                    f"SELECT * FROM read_parquet({_lit(source)})"
                )
            elif low.endswith(".csv"):
                conn.execute(
                    f"CREATE OR REPLACE VIEW {_TABLE} AS "
                    f"SELECT * FROM read_csv_auto({_lit(source)})"
                )
            else:
                raise ValueError(f"DuckDBEngine: unsupported file type for path {source!r}")
            return
        if isinstance(source, pd.DataFrame):
            conn.register(_TABLE, source)
            return
        arrow = self._as_arrow(source)
        if arrow is not None:
            # DuckDB scans Arrow tables natively (zero-copy, no pandas).
            conn.register(_TABLE, arrow)
            return
        if has_polars():
            import polars as pl

            if isinstance(source, pl.LazyFrame):
                source = source.collect()
            if isinstance(source, pl.DataFrame):
                conn.register(_TABLE, source.to_arrow())
                return
        if has_duckdb():
            import duckdb

            if isinstance(source, duckdb.DuckDBPyRelation):
                source.create_view(_TABLE, replace=True)
                return
        raise TypeError(f"DuckDBEngine: cannot register source of type {type(source).__name__}")

    # -- staged SQL pipeline ------------------------------------------------

    def _run_sql_pipeline(
        self, conn: Any, meta: list, plan: Any, config: CleanConfig, report: CleanReport
    ) -> pd.DataFrame:
        from ...steps.strings import active_sentinels

        rename = plan.rename_map
        string_cols = {m.name for m in meta if m.is_string}
        cur = f"SELECT * FROM {_TABLE}"

        if "column_names" in plan.stages and rename:
            self._record_rename(rename, report)

        # rename + strip + sentinel in one projection
        if "column_names" in plan.stages or "clean_strings" in plan.stages:
            if "clean_strings" in plan.stages and string_cols:
                self._record_string_counts(conn, meta, rename, config,
                                           active_sentinels(config), report)
            cur = self._project_clean(meta, rename, config, plan, active_sentinels(config))

        # current column names after rename
        cols = [str(rename.get(m.name, m.name)) for m in meta]

        rows_before = report.rows_before
        if "drop_empty_columns" in plan.stages and rows_before > 0:
            cur, cols = self._drop_empty_columns(conn, cur, cols, report)
        if "drop_empty_rows" in plan.stages and rows_before > 0 and cols:
            cur = self._drop_empty_rows(conn, cur, cols, report)
        if "drop_duplicates" in plan.stages:
            cur = self._drop_duplicates(conn, cur, config, report)

        numeric_map = {str(rename.get(m.name, m.name)): m for m in meta}
        if "impute" in plan.stages and cols:
            cur = self._impute(conn, cur, cols, numeric_map, config, report)
        if "outliers" in plan.stages and cols:
            cur = self._outliers(conn, cur, cols, numeric_map, config, report)

        return conn.execute(cur).fetchdf()

    def _record_rename(self, rename: dict, report: CleanReport) -> None:
        changes = list(rename.items())
        preview = ", ".join(f"{o!r}->{n!r}" for o, n in changes[:4])
        if len(changes) > 4:
            preview += f", … (+{len(changes) - 4} more)"
        report.add("column_names", f"renamed {len(changes)} column(s): {preview}",
                   count=len(changes))

    def _project_clean(
        self, meta: list, rename: dict, config: CleanConfig, plan: Any, sentinels: frozenset
    ) -> str:
        do_strings = "clean_strings" in plan.stages
        sent_list = ", ".join(_lit(s) for s in sentinels)
        pieces = []
        for m in meta:
            target = str(rename.get(m.name, m.name))
            src = _q(m.name)
            if do_strings and m.is_string:
                base = _strip_sql(src) if config.strip_whitespace else src
                if config.normalize_sentinels and sent_list:
                    expr = f"CASE WHEN LOWER({base}) IN ({sent_list}) THEN NULL ELSE {base} END"
                else:
                    expr = base
            else:
                expr = src
            pieces.append(f"{expr} AS {_q(target)}")
        return f"SELECT {', '.join(pieces)} FROM {_TABLE}"

    def _record_string_counts(
        self, conn: Any, meta: list, rename: dict, config: CleanConfig,
        sentinels: frozenset, report: CleanReport
    ) -> None:
        sent_list = ", ".join(_lit(s) for s in sentinels)
        aggs = []
        string_meta = [m for m in meta if m.is_string]
        for i, m in enumerate(string_meta):
            src = _q(m.name)
            if config.strip_whitespace:
                aggs.append(
                    f"SUM(CASE WHEN {_strip_sql(src)} <> {src} AND {src} IS NOT NULL "
                    f"THEN 1 ELSE 0 END) AS s{i}"
                )
            base = _strip_sql(src) if config.strip_whitespace else src
            if config.normalize_sentinels and sent_list:
                aggs.append(
                    f"SUM(CASE WHEN LOWER({base}) IN ({sent_list}) AND {base} IS NOT NULL "
                    f"THEN 1 ELSE 0 END) AS z{i}"
                )
        if not aggs:
            return
        row = conn.execute(f"SELECT {', '.join(aggs)} FROM {_TABLE}").fetchone()
        # clean_strings runs after rename, so actions record the post-rename name.
        k = 0
        for m in string_meta:
            name = str(rename.get(m.name, m.name))
            if config.strip_whitespace:
                n_strip = int(row[k] or 0)
                k += 1
                if n_strip:
                    report.add("strip_whitespace", "trimmed surrounding whitespace",
                               column=name, count=n_strip)
            if config.normalize_sentinels and sent_list:
                n_sent = int(row[k] or 0)
                k += 1
                if n_sent:
                    report.add("normalize_sentinels",
                               'replaced sentinel strings ("N/A", "-", "", …) with missing',
                               column=name, count=n_sent)

    def _drop_empty_columns(
        self, conn: Any, cur: str, cols: list[str], report: CleanReport
    ) -> tuple[str, list[str]]:
        counts = ", ".join(f"COUNT({_q(c)}) AS c{i}" for i, c in enumerate(cols))
        row = conn.execute(f"SELECT {counts} FROM ({cur}) AS _s").fetchone()
        dropped = [c for i, c in enumerate(cols) if int(row[i]) == 0]
        if not dropped:
            return cur, cols
        kept = [c for c in cols if c not in dropped]
        report.columns_dropped.extend(dropped)
        report.add(
            "drop_empty_columns",
            f"dropped {len(dropped)} all-missing column(s): {', '.join(dropped[:6])}"
            + (" …" if len(dropped) > 6 else ""),
            count=len(dropped),
        )
        select_list = ", ".join(_q(c) for c in kept) if kept else "*"
        return f"SELECT {select_list} FROM ({cur}) AS _s", kept

    def _drop_empty_rows(self, conn: Any, cur: str, cols: list[str], report: CleanReport) -> str:
        all_null = " AND ".join(f"{_q(c)} IS NULL" for c in cols)
        (n,) = conn.execute(
            f"SELECT COUNT(*) FROM ({cur}) AS _s WHERE {all_null}"
        ).fetchone()
        n = int(n)
        if n:
            report.add("drop_empty_rows", f"dropped {n} all-missing row(s)", count=n)
            return f"SELECT * FROM ({cur}) AS _s WHERE NOT ({all_null})"
        return cur

    def _impute(
        self, conn: Any, cur: str, cols: list[str], numeric_map: dict,
        config: CleanConfig, report: CleanReport
    ) -> str:
        strategy = config.impute
        if strategy is None:
            return cur
        report.record_backend_difference(
            "duckdb", "impute",
            "median/mode fill values use duckdb aggregates and may differ from the "
            "pandas reference's interpolation/tie-breaking",
        )
        stat_parts: list[str] = []
        index: dict[tuple[int, str], int] = {}

        def add_stat(expr: str, i: int, kind: str) -> None:
            index[(i, kind)] = len(stat_parts)
            stat_parts.append(f"{expr} AS x{len(stat_parts)}")

        cplans: list[tuple[int, str, bool, str]] = []
        for i, c in enumerate(cols):
            m = numeric_map.get(c)
            is_numeric = bool(m and m.is_numeric)
            resolved = resolve_impute_strategy(strategy, is_numeric=is_numeric)
            add_stat(f"COUNT(*) - COUNT({_q(c)})", i, "nm")
            add_stat(f"COUNT({_q(c)})", i, "nn")
            defined = impute_defined_for(strategy, is_numeric=is_numeric)
            if defined and resolved in ("mean", "median"):
                agg = f"AVG({_q(c)})" if resolved == "mean" else f"quantile_cont({_q(c)}, 0.5)"
                add_stat(agg, i, "av")
            cplans.append((i, c, is_numeric, resolved))

        row = conn.execute(f"SELECT {', '.join(stat_parts)} FROM ({cur}) AS _s").fetchone()
        pieces: list[str] = []
        for i, c, is_numeric, resolved in cplans:
            n_missing = int(row[index[(i, "nm")]] or 0)
            non_null = int(row[index[(i, "nn")]] or 0)
            if n_missing == 0 or non_null == 0:
                pieces.append(_q(c))
                continue
            if not impute_defined_for(strategy, is_numeric=is_numeric):
                report.add("impute", f"skipped ({strategy} is not defined for {c})", column=c)
                pieces.append(_q(c))
                continue
            if resolved in ("mean", "median"):
                value = row[index[(i, "av")]]
            else:
                value = self._mode_value(conn, cur, c)
            if value is None:
                report.add("impute", f"skipped (could not compute {resolved} for {c})", column=c)
                pieces.append(_q(c))
                continue
            lit = f"{float(value)}" if is_numeric else _lit(str(value))
            pieces.append(f"COALESCE({_q(c)}, {lit}) AS {_q(c)}")
            shown = f"{value:.6g}" if isinstance(value, float) else repr(value)
            report.add("impute",
                       f"filled {n_missing} missing value(s) with {resolved} ({shown})",
                       column=c, count=n_missing)
            report.columns_imputed.append(c)
        return f"SELECT {', '.join(pieces)} FROM ({cur}) AS _imp"

    def _mode_value(self, conn: Any, cur: str, col: str) -> Any:
        sql = (
            f"SELECT {_q(col)} FROM ({cur}) AS _md WHERE {_q(col)} IS NOT NULL "
            f"GROUP BY {_q(col)} ORDER BY COUNT(*) DESC, {_q(col)} LIMIT 1"
        )
        row = conn.execute(sql).fetchone()
        return row[0] if row else None

    def _outlier_bounds(
        self, conn: Any, cur: str, numeric: list[str], numeric_map: dict,
        method: str, factor: float
    ) -> dict[str, tuple[float, float]]:
        """Resolve integer-safe (lower, upper) fences per numeric column."""
        stat_parts: list[str] = []
        index: dict[tuple[str, str], int] = {}

        def add_stat(expr: str, c: str, kind: str) -> None:
            index[(c, kind)] = len(stat_parts)
            stat_parts.append(f"{expr} AS x{len(stat_parts)}")

        for c in numeric:
            if method == "iqr":
                add_stat(f"quantile_cont({_q(c)}, 0.25)", c, "q1")
                add_stat(f"quantile_cont({_q(c)}, 0.75)", c, "q3")
            else:
                add_stat(f"AVG({_q(c)})", c, "m")
                add_stat(f"stddev_samp({_q(c)})", c, "s")
        srow = conn.execute(f"SELECT {', '.join(stat_parts)} FROM ({cur}) AS _s").fetchone()

        bounds: dict[str, tuple[float, float]] = {}
        for c in numeric:
            if method == "iqr":
                q1, q3 = srow[index[(c, "q1")]], srow[index[(c, "q3")]]
                raw = (None if q1 is None or q3 is None
                       else iqr_bounds(float(q1), float(q3), factor))
            else:
                m, s = srow[index[(c, "m")]], srow[index[(c, "s")]]
                raw = (None if m is None or s is None
                       else zscore_bounds(float(m), float(s), factor))
            if raw is None:
                continue
            bounds[c] = integer_safe_bounds(
                *raw, is_integer=numeric_map[c].dtype_str == "int64"
            )
        return bounds

    def _outlier_counts(
        self, conn: Any, cur: str, bounds: dict[str, tuple[float, float]]
    ) -> dict[str, int]:
        cnt_parts: list[str] = []
        cnt_index: dict[str, int] = {}
        for c, (lo, hi) in bounds.items():
            cnt_index[c] = len(cnt_parts)
            cnt_parts.append(
                f"SUM(CASE WHEN {_q(c)} < {lo} OR {_q(c)} > {hi} THEN 1 ELSE 0 END)"
            )
        crow = conn.execute(f"SELECT {', '.join(cnt_parts)} FROM ({cur}) AS _c").fetchone()
        return {c: int(crow[cnt_index[c]] or 0) for c in bounds}

    def _outliers(
        self, conn: Any, cur: str, cols: list[str], numeric_map: dict,
        config: CleanConfig, report: CleanReport
    ) -> str:
        method = native_outlier_method(config)
        factor = native_outlier_factor(config, method)
        numeric = [c for c in cols if numeric_map.get(c) and numeric_map[c].is_numeric]
        if not numeric:
            return cur
        report.record_backend_difference(
            "duckdb", "outliers",
            "outlier counts depend on duckdb quantile/stddev statistics and may "
            "differ from the pandas reference",
        )
        bounds = self._outlier_bounds(conn, cur, numeric, numeric_map, method, factor)
        if not bounds:
            return cur
        counts = self._outlier_counts(conn, cur, bounds)
        clip = config.outliers == "clip"
        existing = list(cols)
        pieces: list[str] = []
        flags: list[tuple[str, str]] = []
        for c in cols:
            n_out = counts.get(c, 0)
            if clip and c in bounds and n_out:
                lo, hi = bounds[c]
                pieces.append(f"LEAST(GREATEST({_q(c)}, {lo}), {hi}) AS {_q(c)}")
                report.add("outliers",
                           f"clipped {n_out} outlier(s) to [{lo:g}, {hi:g}] "
                           f"({outlier_label(method, factor)})", column=c, count=n_out)
                report.outliers_handled += n_out
            else:
                pieces.append(_q(c))
                if not clip and c in bounds and n_out:
                    flag = self._unique_flag(existing, f"{c}_outlier")
                    existing.append(flag)
                    flags.append((c, flag))
        for c, flag in flags:
            lo, hi = bounds[c]
            n_out = counts[c]
            pieces.append(
                f"COALESCE({_q(c)} < {lo} OR {_q(c)} > {hi}, FALSE) AS {_q(flag)}"
            )
            report.add("outliers",
                       f"flagged {n_out} outlier(s) in new column {flag!r} "
                       f"({outlier_label(method, factor)})", column=c, count=n_out)
            report.outliers_handled += n_out
        return f"SELECT {', '.join(pieces)} FROM ({cur}) AS _out"

    @staticmethod
    def _unique_flag(existing: list[str], base: str) -> str:
        names = set(existing)
        name, k = base, 1
        while name in names:
            k += 1
            name = f"{base}_{k}"
        return name

    def _drop_duplicates(
        self, conn: Any, cur: str, config: CleanConfig, report: CleanReport
    ) -> str:
        (n_before,) = conn.execute(f"SELECT COUNT(*) FROM ({cur}) AS _s").fetchone()
        n_before = int(n_before)
        if n_before < 1:
            return cur
        deduped = f"SELECT DISTINCT * FROM ({cur}) AS _s"
        (n_after,) = conn.execute(f"SELECT COUNT(*) FROM ({deduped}) AS _d").fetchone()
        n_dup = n_before - int(n_after)
        if n_dup <= 0:
            return cur
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
