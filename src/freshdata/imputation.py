# src/freshdata/imputation.py
from __future__ import annotations

import re
import warnings
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

ID_NAME_REGEX = re.compile(
    r"(?:^|_|-)(?:id|uuid|guid|pk|key|seq|account_num|order_no|ssn|token)(?:$|_|-)",
    re.IGNORECASE,
)

NON_COERCIBLE_OBJECT_TYPES = (complex, dict, list, set, tuple, bytes, bytearray)


def is_hostile_object_column(series: pd.Series) -> bool:
    """Return True if an object column contains complex, container, or raw-bytes objects."""
    if series.dtype != object:
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    for val in non_null:
        if isinstance(val, NON_COERCIBLE_OBJECT_TYPES):
            return True
    return False


def detect_identifier_column(series: pd.Series, col_name: Any) -> tuple[bool, str | None]:
    """Detect identifier columns via name regex OR integral-float/monotonic high-cardinality heuristics."""
    name_str = str(col_name)
    if ID_NAME_REGEX.search(name_str):
        return True, "skipped_identifier_imputation:name_pattern"

    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(series.dtype):
        return False, None

    if is_hostile_object_column(series):
        return False, None

    non_null = series.dropna()
    n_valid = len(non_null)
    if n_valid < 4:
        return False, None

    # String/object high-cardinality code patterns (e.g. "ORD-101")
    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        if non_null.nunique(dropna=True) == n_valid and all(
            isinstance(x, str) and any(ch.isdigit() for ch in x) for x in non_null
        ):
            return True, "skipped_identifier_imputation:unique_alphanumeric_code"
        return False, None

    if not pd.api.types.is_numeric_dtype(series.dtype):
        return False, None

    vals = non_null.to_numpy(dtype=np.float64, na_value=np.nan)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 4:
        return False, None

    is_integral = bool(np.all(np.mod(vals, 1.0) == 0.0))
    if not is_integral:
        return False, None

    unique_ratio = len(np.unique(vals)) / float(len(vals))
    diffs = np.diff(vals)
    is_strictly_monotonic_step = bool(len(diffs) > 0 and np.all(diffs > 0) and np.all(diffs <= 2))

    if unique_ratio >= 0.95 and (np.min(np.abs(vals)) >= 100 or (is_strictly_monotonic_step and len(vals) >= 10)):
        return True, "skipped_identifier_imputation:integral_high_cardinality_or_monotonic"

    return False, None


def safe_float64_matrix(df_num: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Convert numeric/nullable-extension DataFrame columns to a finite-safe float64 matrix."""
    valid_cols: list[str] = []
    arrays: list[np.ndarray] = []
    for col in df_num.columns:
        s = df_num[col]
        if pd.api.types.is_bool_dtype(s.dtype) or isinstance(s.dtype, pd.BooleanDtype):
            continue
        if is_hostile_object_column(s):
            continue
        if not pd.api.types.is_numeric_dtype(s.dtype):
            continue
        arr = s.to_numpy(dtype=np.float64, na_value=np.nan).copy()
        arr[~np.isfinite(arr)] = np.nan
        valid_cols.append(col)
        arrays.append(arr)

    if not arrays:
        return np.empty((len(df_num), 0), dtype=np.float64), []
    return np.column_stack(arrays), valid_cols


def safe_corr_matrix(mat: np.ndarray) -> np.ndarray:
    """Compute pairwise Pearson correlation without crashing on NaNs or zero-variance columns."""
    n_rows, n_cols = mat.shape
    corr = np.eye(n_cols, dtype=np.float64)
    if n_cols <= 1 or n_rows < 2:
        return corr

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            xi = mat[:, i]
            xj = mat[:, j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if np.sum(mask) < 2:
                c = 0.0
            else:
                xi_m = xi[mask]
                xj_m = xj[mask]
                std_i = float(np.std(xi_m))
                std_j = float(np.std(xj_m))
                if std_i < 1e-12 or std_j < 1e-12:
                    c = 0.0
                else:
                    c = float(np.dot(xi_m - np.mean(xi_m), xj_m - np.mean(xj_m)) / (len(xi_m) * std_i * std_j))
                    if not np.isfinite(c):
                        c = 0.0
            corr[i, j] = c
            corr[j, i] = c
    return corr


def _knn_fill(
    df: pd.DataFrame,
    target_col: Any,
    k: int = 3,
) -> tuple[pd.Series, int]:
    """Safe KNN imputation over nullable numeric columns with correlation weighting and zero-variance protection."""
    series = df[target_col].copy()
    na_mask = series.isna().to_numpy(dtype=bool)
    if not na_mask.any() or na_mask.all():
        return series, 0

    mat, cols = safe_float64_matrix(df)
    if target_col not in cols:
        return series, 0

    target_idx = cols.index(target_col)
    n_rows, n_cols = mat.shape
    corr = safe_corr_matrix(mat)
    weights = np.abs(corr[target_idx])
    weights[target_idx] = 0.0

    # Standardize columns safely for Euclidean distance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_means = np.nanmean(mat, axis=0)
        col_stds = np.nanstd(mat, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    col_stds = np.where(np.isfinite(col_stds) & (col_stds > 1e-12), col_stds, 1.0)
    z_mat = (mat - col_means) / col_stds

    feature_indices = [idx for idx in range(n_cols) if idx != target_idx and np.isfinite(z_mat[:, idx]).any()]
    target_vals = mat[:, target_idx].copy()
    valid_row_indices = np.where(~na_mask & np.isfinite(target_vals))[0]
    missing_row_indices = np.where(na_mask)[0]

    if len(valid_row_indices) == 0:
        return series, 0

    fallback_median = float(np.median(target_vals[valid_row_indices]))
    imputed_count = 0
    is_integer_target = pd.api.types.is_integer_dtype(series.dtype)

    for r_idx in missing_row_indices:
        if not feature_indices:
            fill_val = fallback_median
        else:
            row_vec = z_mat[r_idx, feature_indices]
            cand_mat = z_mat[np.ix_(valid_row_indices, feature_indices)]
            overlap = np.isfinite(row_vec) & np.isfinite(cand_mat)
            w_sub = weights[feature_indices]
            w_sub = np.where(w_sub > 1e-6, w_sub, 1.0)

            diff_sq = np.where(overlap, (cand_mat - row_vec) ** 2 * w_sub, 0.0)
            overlap_counts = np.sum(overlap, axis=1)
            valid_cands = overlap_counts > 0
            if not np.any(valid_cands):
                fill_val = fallback_median
            else:
                dists = np.full(len(valid_row_indices), np.inf, dtype=np.float64)
                dists[valid_cands] = np.sqrt(np.sum(diff_sq[valid_cands], axis=1) / overlap_counts[valid_cands])
                top_k_local = np.argsort(dists)[: min(k, int(np.sum(valid_cands)))]
                neighbor_rows = valid_row_indices[top_k_local]
                neighbor_dists = dists[top_k_local]
                inv_w = 1.0 / np.maximum(neighbor_dists, 1e-6)
                fill_val = float(np.sum(target_vals[neighbor_rows] * inv_w) / np.sum(inv_w))

        if is_integer_target:
            series.iloc[r_idx] = int(round(fill_val))
        else:
            series.iloc[r_idx] = fill_val
        imputed_count += 1

    return series, imputed_count
