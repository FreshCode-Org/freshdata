# src/freshdata/_core/stages.py
from __future__ import annotations
import dataclasses
import re
from collections.abc import Container
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from freshdata._core.models import ActionRecord, CleanConfig

CURRENCY_REGEX = re.compile(r"^\s*[-+]?\s*[\$€£¥]?\s*[\d,]+(?:\.\d+)?\s*[\$€£¥]?\s*$")

def _normalize_name(name: str) -> str:
    s = re.sub(r"[^\w\s]", "", name).strip().lower()
    return re.sub(r"[\s\-]+", "_", s)

def _is_protected(col: Any, cfg: CleanConfig) -> bool:
    """Single source of truth for columns that cleaning stages must not touch.

    ``cfg`` must already be normalized (see ``normalize_config``).
    """
    name = str(col)
    if col in cfg.id_columns or col == cfg.target_column or name.endswith("_id") or name == "id":
        return True
    rule = cfg.rules.get(col)
    return rule is not None and (rule.protected or rule.role in ("id", "target"))

def _impute_strategy(col: Any, cfg: CleanConfig) -> str | None:
    """Per-field fallback: a rule that leaves ``impute`` unset uses the default."""
    rule = cfg.rules.get(col)
    default = None if cfg.default_impute == "preserve" else cfg.default_impute
    return (rule.impute if rule else None) or default

def _outlier_strategy(col: Any, cfg: CleanConfig) -> str:
    """Per-field fallback: a rule that leaves ``outliers`` unset uses the default."""
    rule = cfg.rules.get(col)
    return (rule.outliers if rule else None) or cfg.default_outliers

def flag_column(col: object, existing: Container[Any]) -> str:
    """Name of the boolean column that outliers="flag" adds. Never overwrites a column."""
    name = f"{col}_outlier"
    if name in existing:
        raise ValueError(f"outliers='flag' on {col!r} would overwrite the existing column {name!r}")
    return name

def _strip_str(x: object) -> object:
    """The one strip_strings executor, shared by planning and pandas apply_plan.

    Strips surrounding whitespace and turns empty strings into None. Missing
    values become None. Other non-string cells are returned unchanged.
    """
    if isinstance(x, str):
        return x.strip() or None
    return None if pd.isna(x) else x

def normalize_config(cfg: CleanConfig) -> CleanConfig:
    if not cfg.normalize_names:
        return cfg
    norm_rules = {_normalize_name(k): v for k, v in cfg.rules.items()}
    norm_id = tuple(_normalize_name(c) for c in cfg.id_columns)
    norm_target = _normalize_name(cfg.target_column) if cfg.target_column else None
    return dataclasses.replace(cfg, rules=norm_rules, id_columns=norm_id, target_column=norm_target)

def stage_column_names(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.normalize_names:
        return df, []
    actions: List[ActionRecord] = []
    rename_map = {}
    for col in df.columns:
        norm = _normalize_name(str(col))
        if norm != col:
            rename_map[col] = norm
            actions.append(ActionRecord("column_names", str(col), "normalize_name", {"to": norm}, 0))
    out = df.rename(columns=rename_map) if rename_map else df
    return out, actions

def stage_strings(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.strip_strings:
        return df, []
    actions: List[ActionRecord] = []
    out = df.copy()
    cfg_eff = normalize_config(cfg)
    for col in out.columns:
        if _is_protected(col, cfg_eff):
            continue
        if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
            # Series.apply (not .str) keeps pandas' result-dtype inference unchanged.
            before = out[col].astype(object)
            after = before.apply(_strip_str)
            changed = (before.astype(str) != after.astype(str)) & ~(before.isna() & after.isna())
            out[col] = after
            actions.append(ActionRecord("strings", str(col), "strip_strings", {}, int(changed.sum())))
    return out, actions

def stage_dtypes(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.fix_dtypes:
        return df, []
    actions: List[ActionRecord] = []
    out = df.copy()
    cfg_eff = normalize_config(cfg)
    for col in out.columns:
        if _is_protected(col, cfg_eff):
            continue
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            sample = out[col].dropna().head(20).astype(str)
            if sample.empty:
                continue
            has_symbol = sample.str.contains(r"[\$€£¥]").any()
            matches_pattern = sample.apply(lambda v: bool(CURRENCY_REGEX.match(v)))
            if has_symbol and (matches_pattern.sum() / len(sample)) >= 0.7:
                parsed = int(out[col].notna().sum())
                cleaned = out[col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
                out[col] = pd.to_numeric(cleaned, errors="coerce")
                actions.append(ActionRecord("dtypes", str(col), "parse_currency", {"target_dtype": "float64"}, parsed))
    return out, actions

def stage_duplicates(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.drop_duplicates:
        return df, []
    orig_len = len(df)
    mask = ~df.duplicated().to_numpy()
    out = df[mask].reset_index(drop=True)
    dropped = orig_len - len(out)
    actions = [ActionRecord("duplicates", "*", "drop_duplicates", {}, dropped)] if dropped > 0 else []
    return out, actions

def stage_missing(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    actions: List[ActionRecord] = []
    out = df.copy()
    cfg_eff = normalize_config(cfg)
    for col in out.columns:
        if _is_protected(col, cfg_eff):
            continue
        impute_strategy = _impute_strategy(col, cfg_eff)
        null_count = int(out[col].isna().sum())
        if impute_strategy == "drop" and (null_count > 0 or len(out) == 0):
            orig_len = len(out)
            mask = ~out[col].isna().to_numpy()
            out = out[mask].reset_index(drop=True)
            actions.append(ActionRecord("missing", str(col), "drop_nulls", {}, orig_len - len(out)))
        elif 0 < null_count < len(out):
            if impute_strategy in ("median", "auto") and pd.api.types.is_numeric_dtype(out[col]):
                med = out[col].median()
                if not pd.isna(med):
                    val = float(med)
                    out[col] = out[col].fillna(val)
                    actions.append(ActionRecord("missing", str(col), "impute_median", {"value": val}, null_count))
            elif impute_strategy == "mean" and pd.api.types.is_numeric_dtype(out[col]):
                mean_val = out[col].mean()
                if not pd.isna(mean_val):
                    val = float(mean_val)
                    out[col] = out[col].fillna(val)
                    actions.append(ActionRecord("missing", str(col), "impute_mean", {"value": val}, null_count))
            elif impute_strategy == "mode":
                mode_vals = out[col].mode()
                if not mode_vals.empty and not pd.isna(mode_vals.iloc[0]):
                    val = mode_vals.iloc[0]
                    val = val.item() if isinstance(val, np.generic) else val  # plain Python scalar, original type
                    out[col] = out[col].fillna(val)
                    actions.append(ActionRecord("missing", str(col), "impute_mode", {"value": val}, null_count))
            elif impute_strategy == "ffill":
                filled = out[col].ffill()
                filled_count = null_count - int(filled.isna().sum())
                if filled_count:
                    out[col] = filled
                    actions.append(ActionRecord("missing", str(col), "impute_ffill", {}, filled_count))
    return out, actions

def stage_outliers(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    actions: List[ActionRecord] = []
    out = df.copy()
    cfg_eff = normalize_config(cfg)
    for col in list(out.columns):
        if _is_protected(col, cfg_eff):
            continue
        strategy = _outlier_strategy(col, cfg_eff)
        if strategy not in ("clip", "drop", "flag") or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        s = out[col].dropna()
        if len(s) < 4:
            continue
        q25, q75 = np.percentile(s, [25, 75])
        iqr = q75 - q25
        lower, upper = q25 - cfg_eff.iqr_factor * iqr, q75 + cfg_eff.iqr_factor * iqr
        if strategy == "clip":
            clipped = int(((out[col] < lower) | (out[col] > upper)).sum())
            out[col] = out[col].clip(lower=lower, upper=upper)
            actions.append(ActionRecord("outliers", str(col), "clip_iqr", {"lower": float(lower), "upper": float(upper)}, clipped))
        elif strategy == "flag":
            flag_col = flag_column(col, out.columns)
            out[flag_col] = (out[col] < lower) | (out[col] > upper)
            actions.append(ActionRecord("outliers", str(col), "flag_iqr", {"lower": float(lower), "upper": float(upper)}, int(out[flag_col].sum())))
        else:
            mask = ((out[col] >= lower) & (out[col] <= upper) | out[col].isna()).to_numpy()
            dropped = int((~mask).sum())
            out = out[mask].reset_index(drop=True)
            actions.append(ActionRecord("outliers", str(col), "drop_outliers", {"lower": float(lower), "upper": float(upper)}, dropped))
    return out, actions
