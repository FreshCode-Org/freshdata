# src/freshdata/_core/stages.py
from __future__ import annotations
import re
from typing import Tuple, List
import numpy as np
import pandas as pd
from freshdata._core.models import ActionRecord, CleanConfig, ColumnRule

CURRENCY_REGEX = re.compile(r"^\s*[-+]?\s*[\$€£¥]?\s*[\d,]+(?:\.\d+)?\s*[\$€£¥]?\s*$")

def _normalize_name(name: str) -> str:
    s = re.sub(r"[^\w\s]", "", name).strip().lower()
    return re.sub(r"[\s\-]+", "_", s)

def normalize_config(cfg: CleanConfig) -> CleanConfig:
    if not cfg.normalize_names:
        return cfg
    norm_rules = {_normalize_name(k): v for k, v in cfg.rules.items()}
    norm_id = tuple(_normalize_name(c) for c in cfg.id_columns)
    norm_target = _normalize_name(cfg.target_column) if cfg.target_column else None
    return CleanConfig(
        strategy=cfg.strategy,
        rules=norm_rules,
        id_columns=norm_id,
        target_column=norm_target,
        normalize_names=cfg.normalize_names,
        strip_strings=cfg.strip_strings,
        fix_dtypes=cfg.fix_dtypes,
        drop_duplicates=cfg.drop_duplicates,
        default_impute=cfg.default_impute,
        default_outliers=cfg.default_outliers,
        iqr_factor=cfg.iqr_factor,
        strict_backend=cfg.strict_backend,
    )

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
            def _clean_str(x):
                if pd.isna(x) or x is None:
                    return None
                if isinstance(x, str):
                    val = re.sub(r"\s+", " ", x.strip())
                    if not val or val.lower() in ("nan", "none", "null"):
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
            has_symbol = sample.str.contains(r"[\$€£¥]").any()
            matches_pattern = sample.apply(lambda v: bool(CURRENCY_REGEX.match(v)))
            if has_symbol and (matches_pattern.sum() / len(sample)) >= 0.7:
                cleaned = out[col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
                out[col] = pd.to_numeric(cleaned, errors="coerce")
                actions.append(ActionRecord("dtypes", str(col), "parse_currency", {"target_dtype": "float64"}, len(out)))
    return out, actions

def stage_duplicates(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.drop_duplicates:
        return df, []
    orig_len = len(df)
    mask = ~df.duplicated().to_numpy()
    out = pd.DataFrame({col: df[col].to_numpy()[mask] for col in df.columns})
    dropped = orig_len - len(out)
    actions = [ActionRecord("duplicates", "*", "drop_duplicates", {}, dropped)] if dropped > 0 else []
    return out, actions

def stage_missing(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    actions: List[ActionRecord] = []
    out = df.copy()
    cfg_eff = normalize_config(cfg)
    for col in out.columns:
        if col in cfg_eff.id_columns or col == cfg_eff.target_column:
            continue
        rule = cfg_eff.rules.get(col, ColumnRule())
        if rule.protected:
            continue
        impute_strategy = rule.impute or (cfg_eff.default_impute if cfg_eff.default_impute != "preserve" else None)
        null_count = int(out[col].isna().sum())
        if null_count > 0:
            if impute_strategy in ("median", "auto") and pd.api.types.is_numeric_dtype(out[col]):
                val = float(out[col].median())
                out[col] = out[col].fillna(val)
                actions.append(ActionRecord("missing", str(col), "impute_median", {"value": val}, null_count))
            elif impute_strategy == "mean" and pd.api.types.is_numeric_dtype(out[col]):
                val = float(out[col].mean())
                out[col] = out[col].fillna(val)
                actions.append(ActionRecord("missing", str(col), "impute_mean", {"value": val}, null_count))
            elif impute_strategy == "mode":
                mode_vals = out[col].mode()
                if not mode_vals.empty:
                    val = mode_vals.iloc[0]
                    out[col] = out[col].fillna(val)
                    actions.append(ActionRecord("missing", str(col), "impute_mode", {"value": str(val)}, null_count))
            elif impute_strategy == "drop":
                mask = ~out[col].isna().to_numpy()
                out = pd.DataFrame({c: out[c].to_numpy()[mask] for c in out.columns})
                actions.append(ActionRecord("missing", str(col), "drop_nulls", {}, null_count))
    return out, actions

def stage_outliers(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    actions: List[ActionRecord] = []
    out = df.copy()
    cfg_eff = normalize_config(cfg)
    for col in out.columns:
        if col in cfg_eff.id_columns or col == cfg_eff.target_column:
            continue
        rule = cfg_eff.rules.get(col, ColumnRule())
        if rule.protected:
            continue
        strategy = rule.outliers or cfg_eff.default_outliers
        if strategy == "clip" and pd.api.types.is_numeric_dtype(out[col]):
            s = out[col].dropna()
            if len(s) < 4:
                continue
            q25, q75 = np.percentile(s, [25, 75])
            iqr = q75 - q25
            lower, upper = q25 - cfg_eff.iqr_factor * iqr, q75 + cfg_eff.iqr_factor * iqr
            out[col] = out[col].clip(lower=lower, upper=upper)
            actions.append(ActionRecord("outliers", str(col), "clip_iqr", {"lower": float(lower), "upper": float(upper)}, len(out)))
        elif strategy == "drop" and pd.api.types.is_numeric_dtype(out[col]):
            s = out[col].dropna()
            if len(s) >= 4:
                q25, q75 = np.percentile(s, [25, 75])
                iqr = q75 - q25
                lower, upper = q25 - cfg_eff.iqr_factor * iqr, q75 + cfg_eff.iqr_factor * iqr
                mask = ((out[col] >= lower) & (out[col] <= upper) | out[col].isna()).to_numpy()
                dropped = int((~mask).sum())
                out = pd.DataFrame({c: out[c].to_numpy()[mask] for c in out.columns})
                actions.append(ActionRecord("outliers", str(col), "drop_outliers", {"lower": float(lower), "upper": float(upper)}, dropped))
    return out, actions
