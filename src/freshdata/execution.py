# src/freshdata/execution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from freshdata.config import CleanConfig
from freshdata.imputation import (
    _knn_fill,
    detect_identifier_column,
    is_hostile_object_column,
)
from freshdata.report import CleanAction, CleanReport
from freshdata.schema_envelope import ROW_ID_SENTINEL_COL, SchemaEnvelope


class StreamingUnsupportedError(RuntimeError):
    """Raised when strict_streaming=True and a requested stage cannot execute in Polars streaming."""


@dataclass
class EngineConfig:
    engine: str = "pandas"  # "pandas" | "polars" | "duckdb"
    streaming: bool = False
    strict_streaming: bool = False
    streaming_dedup_unordered: bool = False


def _canonicalize_pandas_string_col(
    s: pd.Series,
    *,
    strip_whitespace: bool,
    normalize_sentinels: bool,
    sentinels: set[str],
) -> pd.Series:
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats = s.cat.categories
        if pd.api.types.is_string_dtype(cats.dtype) or cats.dtype == object:
            new_cats = [c.strip() if isinstance(c, str) and strip_whitespace else c for c in cats]
            s = s.cat.rename_categories(new_cats)
            if normalize_sentinels:
                to_remove = [c for c in s.cat.categories if isinstance(c, str) and c in sentinels]
                if to_remove:
                    s = s.cat.remove_categories(to_remove)
        return s

    if is_hostile_object_column(s):
        if not strip_whitespace and not normalize_sentinels:
            return s

        def _clean_elem(val: Any) -> Any:
            if isinstance(val, str):
                v = val.strip() if strip_whitespace else val
                if normalize_sentinels and v in sentinels:
                    return np.nan
                return v
            return val

        return s.map(_clean_elem)

    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        orig_dtype = s.dtype
        out = s.copy()
        if strip_whitespace:
            out = out.str.strip()
        if normalize_sentinels:
            mask = out.isin(sentinels)
            na_val = pd.NA if isinstance(orig_dtype, pd.StringDtype) else np.nan
            out = out.mask(mask, na_val)
        return out.astype(orig_dtype)
    return s


def _try_promote_dtypes(
    df: pd.DataFrame,
    config: CleanConfig,
    report: CleanReport,
) -> pd.DataFrame:
    if not config.fix_dtypes:
        return df

    out = df.copy()
    bool_map = {"true": True, "false": False, "yes": True, "no": False}

    for col in out.columns:
        if col == ROW_ID_SENTINEL_COL:
            continue
        s = out[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            continue
        if not (pd.api.types.is_string_dtype(s.dtype) or s.dtype == object):
            continue
        if is_hostile_object_column(s):
            report.skipped_columns.append({
                "column": str(col),
                "reason": "skipped_dtype_promotion:hostile_object_elements",
            })
            continue

        non_null = s.dropna()
        if len(non_null) == 0:
            continue

        lowered = non_null.astype(str).str.strip().str.lower()
        if set(lowered.unique()).issubset(set(bool_map.keys())):
            old_dt = str(s.dtype)
            mapped = s.astype("string").str.strip().str.lower().map(bool_map)
            out[col] = mapped.astype("boolean")
            report.dtype_changes[str(col)] = (old_dt, "boolean")
            report.actions.append(CleanAction("fix_dtypes", str(col), f"Promoted {old_dt} -> boolean", len(non_null)))
            continue

        coerced = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
        if coerced.notna().all():
            old_dt = str(s.dtype)
            full_num = pd.to_numeric(s.astype("string").str.strip(), errors="coerce")
            vals = coerced.to_numpy(dtype=np.float64)
            if np.all(np.mod(vals, 1.0) == 0.0):
                out[col] = full_num.round().astype("Int64")
                report.dtype_changes[str(col)] = (old_dt, "Int64")
                report.actions.append(CleanAction("fix_dtypes", str(col), f"Promoted {old_dt} -> Int64", len(non_null)))
            else:
                out[col] = full_num.astype("Float64")
                report.dtype_changes[str(col)] = (old_dt, "Float64")
                report.actions.append(CleanAction("fix_dtypes", str(col), f"Promoted {old_dt} -> Float64", len(non_null)))
    return out


def _run_imputation_and_outliers(
    df: pd.DataFrame,
    config: CleanConfig,
    report: CleanReport,
) -> pd.DataFrame:
    out = df.copy()
    data_cols = [c for c in out.columns if c != ROW_ID_SENTINEL_COL]

    if config.handle_outliers and len(out) >= 4:
        for col in data_cols:
            if col == config.target_column:
                continue
            s = out[col]
            if pd.api.types.is_bool_dtype(s.dtype) or isinstance(s.dtype, pd.BooleanDtype):
                continue
            if not pd.api.types.is_numeric_dtype(s.dtype):
                continue
            is_id, _ = detect_identifier_column(s, col)
            if is_id:
                continue
            vals = s.dropna().to_numpy(dtype=np.float64, na_value=np.nan)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 4:
                continue
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            if iqr <= 0:
                continue
            low = q25 - config.outlier_iqr_multiplier * iqr
            high = q75 + config.outlier_iqr_multiplier * iqr
            arr = s.to_numpy(dtype=np.float64, na_value=np.nan)
            outlier_mask = np.isfinite(arr) & ((arr < low) | (arr > high))
            n_out = int(np.sum(outlier_mask))
            if n_out > 0:
                clipped = np.clip(arr, low, high)
                if pd.api.types.is_integer_dtype(s.dtype):
                    clipped_s = pd.Series(np.round(clipped), index=s.index).where(s.notna(), pd.NA).astype(s.dtype)
                else:
                    clipped_s = pd.Series(clipped, index=s.index).where(s.notna(), np.nan).astype(s.dtype)
                out[col] = clipped_s
                report.outliers_clipped += n_out
                report.actions.append(CleanAction("handle_outliers", str(col), f"Clipped {n_out} outliers", n_out))

    if config.impute_missing and len(out) > 0:
        for col in data_cols:
            if col == config.target_column:
                report.skipped_columns.append({"column": str(col), "reason": "skipped_target_column"})
                continue
            s = out[col]
            n_missing = int(s.isna().sum())
            if n_missing == 0:
                continue

            is_id, reason = detect_identifier_column(s, col)
            if is_id and reason is not None:
                report.skipped_columns.append({"column": str(col), "reason": reason})
                report.actions.append(CleanAction("skip_identifier", str(col), reason, 0))
                continue

            if is_hostile_object_column(s):
                report.skipped_columns.append({"column": str(col), "reason": "skipped_hostile_object_imputation"})
                continue

            non_null = s.dropna()
            if len(non_null) == 0:
                report.skipped_columns.append({"column": str(col), "reason": "skipped_all_null_column"})
                continue

            if pd.api.types.is_numeric_dtype(s.dtype) and not (
                pd.api.types.is_bool_dtype(s.dtype) or isinstance(s.dtype, pd.BooleanDtype)
            ):
                if config.imputer == "knn":
                    filled_s, count = _knn_fill(out[data_cols], col)
                    out[col] = filled_s.astype(s.dtype)
                else:
                    med = float(np.median(non_null.to_numpy(dtype=np.float64)))
                    fill_val: Any = int(round(med)) if pd.api.types.is_integer_dtype(s.dtype) else med
                    out[col] = s.fillna(fill_val).astype(s.dtype)
                    count = n_missing
                report.values_imputed += count
                report.actions.append(CleanAction("impute_missing", str(col), f"Imputed {count} numeric nulls", count))
            else:
                mode_vals = non_null.mode(dropna=True)
                if len(mode_vals) > 0:
                    # Deterministic tie-break by sorting string representation
                    sorted_modes = sorted(mode_vals.tolist(), key=lambda x: str(x))
                    fill_val = sorted_modes[0]
                    out[col] = s.fillna(fill_val).astype(s.dtype)
                    report.values_imputed += n_missing
                    report.actions.append(CleanAction("impute_missing", str(col), f"Imputed {n_missing} categorical nulls", n_missing))

    return out


class PandasEngine:
    @staticmethod
    def execute(df: pd.DataFrame, config: CleanConfig, engine_config: EngineConfig) -> tuple[pd.DataFrame, CleanReport]:
        report = CleanReport(
            engine_used="pandas",
            rows_before=len(df),
            streaming_requested=engine_config.streaming,
            streaming_executed_fully=False,
            row_order_deterministic=True,
        )
        envelope = SchemaEnvelope.capture(df)
        if df.shape[1] == 0 or df.shape[0] == 0:
            restored = envelope.restore(df, fix_dtypes=config.fix_dtypes, dtype_changes=report.dtype_changes)
            report.rows_after = len(restored)
            return restored, report

        work = df.copy(deep=True)
        sentinels = set(config.null_sentinels)

        for col in work.columns:
            work[col] = _canonicalize_pandas_string_col(
                work[col],
                strip_whitespace=config.strip_whitespace,
                normalize_sentinels=config.normalize_sentinels,
                sentinels=sentinels,
            )

        work = _try_promote_dtypes(work, config, report)

        if config.drop_duplicates:
            before_dedup = len(work)
            keep_arg: Any = "first" if config.duplicate_keep == "any" else config.duplicate_keep
            try:
                work = work.drop_duplicates(subset=config.duplicate_subset, keep=keep_arg)
                removed = before_dedup - len(work)
                report.duplicates_removed = removed
                if removed > 0:
                    report.actions.append(CleanAction("drop_duplicates", None, f"Removed {removed} duplicates", removed))
            except TypeError:
                pass
        else:
            try:
                dup_count = int(work.duplicated(subset=config.duplicate_subset).sum())
                if dup_count > 0:
                    report.warnings.append(f"Detected {dup_count} duplicate rows (drop_duplicates=False).")
            except TypeError:
                pass

        work = _run_imputation_and_outliers(work, config, report)
        restored = envelope.restore(work, fix_dtypes=config.fix_dtypes, dtype_changes=report.dtype_changes)
        report.rows_after = len(restored)
        return restored, report


class PolarsEngine:
    @staticmethod
    def execute(df: pd.DataFrame, config: CleanConfig, engine_config: EngineConfig) -> tuple[pd.DataFrame, CleanReport]:
        report = CleanReport(
            engine_used="polars",
            rows_before=len(df),
            streaming_requested=engine_config.streaming,
            streaming_executed_fully=engine_config.streaming,
            row_order_deterministic=True,
        )
        envelope = SchemaEnvelope.capture(df)

        # Inspect streaming compatibility up front
        if engine_config.streaming:
            if config.drop_duplicates and not engine_config.streaming_dedup_unordered:
                report.non_streaming_stages.append("drop_duplicates:maintain_order")
                report.streaming_executed_fully = False
            if config.drop_duplicates and config.duplicate_subset is not None:
                report.non_streaming_stages.append("drop_duplicates:subset")
                report.streaming_executed_fully = False
            if config.impute_missing and config.imputer == "knn":
                report.non_streaming_stages.append("impute_missing:knn")
                report.streaming_executed_fully = False

            if engine_config.strict_streaming and report.non_streaming_stages:
                raise StreamingUnsupportedError(
                    f"Polars strict_streaming=True failed: stages {report.non_streaming_stages} "
                    "cannot execute inside the Polars streaming engine."
                )

        if df.shape[1] == 0 or df.shape[0] == 0:
            restored = envelope.restore(df, fix_dtypes=config.fix_dtypes, dtype_changes=report.dtype_changes)
            report.rows_after = len(restored)
            return restored, report

        # Check for hostile object columns or categorical columns needing pre-normalization
        has_hostile = any(is_hostile_object_column(df[c]) for c in df.columns)
        if has_hostile:
            report.fallback_events.append({
                "stage": "full_pipeline",
                "reason": "hostile_python_objects_require_pandas_object_handling",
            })
            res, pd_rep = PandasEngine.execute(df, config, engine_config)
            pd_rep.engine_used = "polars"
            pd_rep.fallback_events = report.fallback_events
            return res, pd_rep

        work_pd = envelope.attach_row_ids(df)
        sentinels = set(config.null_sentinels)

        # Categorical normalization before Arrow bridge
        for col in df.columns:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                work_pd[col] = _canonicalize_pandas_string_col(
                    work_pd[col],
                    strip_whitespace=config.strip_whitespace,
                    normalize_sentinels=config.normalize_sentinels,
                    sentinels=sentinels,
                )

        lf = pl.from_pandas(work_pd).lazy()
        schema = lf.collect_schema()

        # Native Polars whitespace & sentinel canonicalization
        exprs: list[pl.Expr] = []
        for col_name, pl_dtype in schema.items():
            if col_name == ROW_ID_SENTINEL_COL:
                continue
            if pl_dtype == pl.String:
                e = pl.col(col_name)
                if config.strip_whitespace:
                    e = e.str.strip_chars()
                if config.normalize_sentinels:
                    e = pl.when(e.is_in(list(sentinels))).then(None).otherwise(e).alias(col_name)
                exprs.append(e.alias(col_name))
        if exprs:
            lf = lf.with_columns(exprs)

        # Native Polars duplicate removal
        data_cols = [str(c) for c in df.columns]
        dedup_cols = config.duplicate_subset if config.duplicate_subset is not None else data_cols
        if config.drop_duplicates:
            if engine_config.streaming and engine_config.streaming_dedup_unordered:
                report.row_order_deterministic = False
                lf = lf.unique(subset=dedup_cols, keep="any", maintain_order=False)
            else:
                keep_mode = "first" if config.duplicate_keep == "any" else config.duplicate_keep
                lf = lf.unique(subset=dedup_cols, keep=keep_mode, maintain_order=True)

        if engine_config.streaming:
            try:
                report.plan_explain = lf.explain(engine="streaming")
            except (TypeError, Exception):
                try:
                    report.plan_explain = lf.explain(streaming=True)
                except Exception:
                    report.plan_explain = lf.explain()
            try:
                pl_df = lf.collect(engine="streaming")
            except (TypeError, Exception):
                pl_df = lf.collect()
        else:
            pl_df = lf.collect()

        # Convert back to pandas for schema restoration & statistical imputation parity
        mid_pd = pl_df.to_pandas()
        if config.drop_duplicates:
            # Sort by surviving __freshdata_row_id__ so order matches pandas drop_duplicates exactly
            if ROW_ID_SENTINEL_COL in mid_pd.columns and report.row_order_deterministic:
                mid_pd = mid_pd.sort_values(ROW_ID_SENTINEL_COL, kind="mergesort").reset_index(drop=True)
            removed = len(df) - len(mid_pd)
            report.duplicates_removed = removed
            if removed > 0:
                report.actions.append(CleanAction("drop_duplicates", None, f"Removed {removed} duplicates", removed))
        else:
            try:
                dup_count = int(mid_pd[data_cols].duplicated(subset=config.duplicate_subset).sum())
                if dup_count > 0:
                    report.warnings.append(f"Detected {dup_count} duplicate rows (drop_duplicates=False).")
            except TypeError:
                pass

        # Restore pre-promotion extension types first so imputation operates on exact dtypes
        mid_restored = envelope.restore(mid_pd, fix_dtypes=False, dtype_changes={})
        mid_promoted = _try_promote_dtypes(mid_restored, config, report)
        final_df = _run_imputation_and_outliers(mid_promoted, config, report)
        final_df = envelope.restore(final_df, fix_dtypes=config.fix_dtypes, dtype_changes=report.dtype_changes)
        report.rows_after = len(final_df)
        return final_df, report


class DuckDBEngine:
    @staticmethod
    def execute(df: pd.DataFrame, config: CleanConfig, engine_config: EngineConfig) -> tuple[pd.DataFrame, CleanReport]:
        report = CleanReport(
            engine_used="duckdb",
            rows_before=len(df),
            streaming_requested=engine_config.streaming,
            streaming_executed_fully=False,
            row_order_deterministic=True,
        )
        envelope = SchemaEnvelope.capture(df)
        if df.shape[1] == 0 or df.shape[0] == 0:
            restored = envelope.restore(df, fix_dtypes=config.fix_dtypes, dtype_changes=report.dtype_changes)
            report.rows_after = len(restored)
            return restored, report

        has_hostile = any(is_hostile_object_column(df[c]) for c in df.columns)
        if has_hostile:
            report.fallback_events.append({
                "stage": "full_pipeline",
                "reason": "hostile_python_objects_require_pandas_object_handling",
            })
            res, pd_rep = PandasEngine.execute(df, config, engine_config)
            pd_rep.engine_used = "duckdb"
            pd_rep.fallback_events = report.fallback_events
            return res, pd_rep

        work_pd = envelope.attach_row_ids(df)
        sentinels = set(config.null_sentinels)

        for col in df.columns:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                work_pd[col] = _canonicalize_pandas_string_col(
                    work_pd[col],
                    strip_whitespace=config.strip_whitespace,
                    normalize_sentinels=config.normalize_sentinels,
                    sentinels=sentinels,
                )

        con = duckdb.connect(":memory:")
        try:
            con.register("input_tbl", work_pd)
            select_exprs: list[str] = []
            sentinel_sql_list = ", ".join(f"'{s.replace(chr(39), chr(39)*2)}'" for s in sentinels)

            for col in work_pd.columns:
                qcol = f'"{col}"'
                if col == ROW_ID_SENTINEL_COL:
                    select_exprs.append(qcol)
                    continue
                dt = df[col].dtype
                if (pd.api.types.is_string_dtype(dt) or dt == object) and not isinstance(dt, pd.CategoricalDtype):
                    expr = f"TRIM({qcol})" if config.strip_whitespace else qcol
                    if config.normalize_sentinels:
                        expr = f"CASE WHEN {expr} IN ({sentinel_sql_list}) THEN NULL ELSE {expr} END"
                    select_exprs.append(f"{expr} AS {qcol}")
                else:
                    select_exprs.append(qcol)

            stage1_sql = f"SELECT {', '.join(select_exprs)} FROM input_tbl"

            if config.drop_duplicates:
                dedup_cols = config.duplicate_subset if config.duplicate_subset is not None else [str(c) for c in df.columns]
                partition_cols = ", ".join(f'"{c}"' for c in dedup_cols)
                order_dir = "DESC" if config.duplicate_keep == "last" else "ASC"
                sql = f"""
                    WITH staged AS ({stage1_sql}),
                    numbered AS (
                        SELECT *,
                               ROW_NUMBER() OVER (
                                   PARTITION BY {partition_cols}
                                   ORDER BY "{ROW_ID_SENTINEL_COL}" {order_dir}
                               ) AS __rn
                        FROM staged
                    )
                    SELECT * EXCLUDE (__rn)
                    FROM numbered
                    WHERE __rn = 1
                    ORDER BY "{ROW_ID_SENTINEL_COL}" ASC
                """
            else:
                sql = f"{stage1_sql} ORDER BY \"{ROW_ID_SENTINEL_COL}\" ASC"

            mid_pd = con.execute(sql).df()
        finally:
            con.close()

        if config.drop_duplicates:
            removed = len(df) - len(mid_pd)
            report.duplicates_removed = removed
            if removed > 0:
                report.actions.append(CleanAction("drop_duplicates", None, f"Removed {removed} duplicates", removed))
        else:
            data_cols = [str(c) for c in df.columns]
            try:
                dup_count = int(mid_pd[data_cols].duplicated(subset=config.duplicate_subset).sum())
                if dup_count > 0:
                    report.warnings.append(f"Detected {dup_count} duplicate rows (drop_duplicates=False).")
            except TypeError:
                pass

        mid_restored = envelope.restore(mid_pd, fix_dtypes=False, dtype_changes={})
        mid_promoted = _try_promote_dtypes(mid_restored, config, report)
        final_df = _run_imputation_and_outliers(mid_promoted, config, report)
        final_df = envelope.restore(final_df, fix_dtypes=config.fix_dtypes, dtype_changes=report.dtype_changes)
        report.rows_after = len(final_df)
        return final_df, report
