# src/freshdata/schema_envelope.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


ROW_ID_SENTINEL_COL = "__freshdata_row_id__"

EXTENSION_DTYPE_TYPES = (
    pd.Int8Dtype,
    pd.Int16Dtype,
    pd.Int32Dtype,
    pd.Int64Dtype,
    pd.UInt8Dtype,
    pd.UInt16Dtype,
    pd.UInt32Dtype,
    pd.UInt64Dtype,
    pd.Float32Dtype,
    pd.Float64Dtype,
    pd.StringDtype,
    pd.BooleanDtype,
    pd.CategoricalDtype,
    pd.DatetimeTZDtype,
)


@dataclass(frozen=True)
class ColumnSchemaSpec:
    name: Any
    dtype: Any
    is_extension: bool
    categories: pd.Index | None = None
    ordered: bool = False
    string_storage: str | None = None
    tz: Any = None


@dataclass
class SchemaEnvelope:
    shape: tuple[int, int]
    columns: pd.Index
    index: pd.Index
    column_specs: dict[Any, ColumnSchemaSpec] = field(default_factory=dict)

    @classmethod
    def capture(cls, df: pd.DataFrame) -> SchemaEnvelope:
        specs: dict[Any, ColumnSchemaSpec] = {}
        for col in df.columns:
            dt = df[col].dtype
            is_ext = isinstance(dt, EXTENSION_DTYPE_TYPES)
            cats = dt.categories.copy() if isinstance(dt, pd.CategoricalDtype) else None
            ordered = bool(dt.ordered) if isinstance(dt, pd.CategoricalDtype) else False
            storage = getattr(dt, "storage", None) if isinstance(dt, pd.StringDtype) else None
            tz = getattr(dt, "tz", None) if isinstance(dt, pd.DatetimeTZDtype) else None
            specs[col] = ColumnSchemaSpec(
                name=col,
                dtype=dt,
                is_extension=is_ext,
                categories=cats,
                ordered=ordered,
                string_storage=storage,
                tz=tz,
            )
        return cls(
            shape=df.shape,
            columns=df.columns.copy(deep=True),
            index=df.index.copy(deep=True),
            column_specs=specs,
        )

    def attach_row_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] == 0:
            return df.copy(deep=True)
        out = df.copy(deep=True)
        out[ROW_ID_SENTINEL_COL] = np.arange(len(df), dtype=np.int64)
        return out

    def restore(
        self,
        result_df: pd.DataFrame,
        *,
        fix_dtypes: bool = False,
        dtype_changes: dict[str, tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        # Handle degenerate zero-column frames (N, 0)
        if self.shape[1] == 0:
            return pd.DataFrame(index=self.index.copy(deep=True), columns=self.columns.copy(deep=True))

        out = result_df.copy()

        # Restore surviving index positions via sentinel column if present
        if ROW_ID_SENTINEL_COL in out.columns:
            surviving_pos = out[ROW_ID_SENTINEL_COL].to_numpy(dtype=np.int64)
            out = out.drop(columns=[ROW_ID_SENTINEL_COL])
            out.index = self.index.take(surviving_pos)
        elif len(out) == len(self.index):
            out.index = self.index.copy(deep=True)
        elif len(out) == 0:
            out.index = self.index[:0].copy(deep=True)

        # Ensure column ordering matches original schema
        surviving_cols = [c for c in self.columns if c in out.columns]
        out = out[surviving_cols]
        out.columns = pd.Index(surviving_cols, name=self.columns.name)

        # Restore exact dtypes unless intentionally promoted by fix_dtypes
        for col in surviving_cols:
            spec = self.column_specs[col]
            col_key = str(col)
            was_promoted = bool(fix_dtypes and dtype_changes and col_key in dtype_changes)
            if was_promoted:
                continue

            out[col] = self._restore_series_dtype(out[col], spec)

        return out

    @staticmethod
    def _restore_series_dtype(series: pd.Series, spec: ColumnSchemaSpec) -> pd.Series:
        target_dtype = spec.dtype
        if isinstance(target_dtype, pd.CategoricalDtype):
            # If strings were trimmed during cleaning, update categories accordingly while preserving order
            cleaned_cats = (
                spec.categories.str.strip()
                if spec.categories is not None and pd.api.types.is_string_dtype(spec.categories.dtype)
                else spec.categories
            )
            cat_dtype = pd.CategoricalDtype(categories=cleaned_cats, ordered=spec.ordered)
            return series.astype(cat_dtype)

        if isinstance(target_dtype, pd.DatetimeTZDtype):
            s = pd.to_datetime(series, utc=(str(spec.tz) == "UTC"))
            if s.dt.tz is None:
                s = s.dt.tz_localize(spec.tz)
            else:
                s = s.dt.tz_convert(spec.tz)
            return s.astype(target_dtype)

        if isinstance(target_dtype, pd.StringDtype):
            return series.astype(pd.StringDtype(storage=spec.string_storage or "python"))

        if isinstance(target_dtype, pd.BooleanDtype):
            return series.astype("boolean")

        if isinstance(target_dtype, (
            pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype,
            pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype,
        )):
            if pd.api.types.is_float_dtype(series.dtype):
                series = series.round()
            return series.astype(target_dtype)

        if isinstance(target_dtype, (pd.Float32Dtype, pd.Float64Dtype)):
            return series.astype(target_dtype)

        if series.dtype != target_dtype:
            return series.astype(target_dtype)
        return series
