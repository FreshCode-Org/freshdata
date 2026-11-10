# src/freshdata/_core/stages.py
from __future__ import annotations
import re
from typing import Tuple, List
import numpy as np
import pandas as pd
from freshdata._core.models import ActionRecord, CleanConfig, ColumnRule

def _normalize_name(name: str) -> str:
    s = re.sub(r"[^\w\s]", "", name).strip().lower()
    return re.sub(r"[\s\-]+", "_", s)

def stage_column_names(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.normalize_names:
        return df, []
    actions: List[ActionRecord] = []
    rename_map = {}
    for col in df.columns:
        norm = _normalize_name(str(col))
        if norm != col:
            rename_map[col] = norm
            actions.append(ActionRecord("column_names", str(col), "normalize_name", {"to": norm}, len(df)))
    out = df.rename(columns=rename_map) if rename_map else df
    return out, actions

def stage_strings(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.strip_strings:
        return df, []
    actions: List[ActionRecord] = []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
            s = out[col].astype(object)
            # Only strip and replace on string instances to preserve non-string objects
            def _clean_str(x):
                if pd.isna(x) or x is None:
                    return None
                if isinstance(x, str):
                    val = x.strip()
                    if not val or val.lower() in ("nan", "none"):
                        return None
                    return val
                return x

            out[col] = s.apply(_clean_str)
            actions.append(ActionRecord("strings", str(col), "strip_strings", {}, len(out)))
    return out, actions

def stage_dtypes(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.fix_dtypes:
        return df, []
    actions: List[ActionRecord] = []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            sample = out[col].dropna().head(20).astype(str)
            if sample.empty:
                continue
            is_currency = sample.str.contains(r"^[\$€£¥]\s*[\d,]+(?:\.\d+)?$").any()
            if is_currency:
                cleaned = out[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
                out[col] = pd.to_numeric(cleaned, errors="coerce")
                actions.append(ActionRecord("dtypes", str(col), "parse_currency", {"target_dtype": "float64"}, len(out)))
    return out, actions

def stage_duplicates(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.drop_duplicates:
        return df, []
    orig_len = len(df)
    out = df.drop_duplicates()
    dropped = orig_len - len(out)
    actions = [ActionRecord("duplicates", "*", "drop_duplicates", {}, dropped)] if dropped > 0 else []
    return out, actions

def stage_missing(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    actions: List[ActionRecord] = []
    out = df.copy()
    for col in out.columns:
        if col in cfg.id_columns or col == cfg.target_column:
            continue
        rule = cfg.rules.get(col, ColumnRule())
        if rule.protected:
            continue
        impute_strategy = rule.impute or (cfg.default_impute if cfg.default_impute != "preserve" else None)
        null_count = int(out[col].isna().sum())
        if null_count > 0 and impute_strategy in ("median", "auto") and pd.api.types.is_numeric_dtype(out[col]):
            val = float(out[col].median())
            out[col] = out[col].fillna(val)
            actions.append(ActionRecord("missing", str(col), "impute_median", {"value": val}, null_count))
    return out, actions

def stage_outliers(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    actions: List[ActionRecord] = []
    out = df.copy()
    for col in out.columns:
        if col in cfg.id_columns or col == cfg.target_column:
            continue
        rule = cfg.rules.get(col, ColumnRule())
        if rule.protected:
            continue
        strategy = rule.outliers or cfg.default_outliers
        if strategy == "clip" and pd.api.types.is_numeric_dtype(out[col]):
            s = out[col].dropna()
            if len(s) < 4:
                continue
            q25, q75 = np.percentile(s, [25, 75])
            iqr = q75 - q25
            lower, upper = q25 - cfg.iqr_factor * iqr, q75 + cfg.iqr_factor * iqr
            out[col] = out[col].clip(lower=lower, upper=upper)
            actions.append(ActionRecord("outliers", str(col), "clip_iqr", {"lower": float(lower), "upper": float(upper)}, len(out)))
    return out, actions
