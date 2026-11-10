# src/freshdata/_core/engine.py
from __future__ import annotations
from typing import Any
import pandas as pd
import polars as pl
from freshdata._core.backends import resolve_backend
from freshdata._core.models import (
    ActionRecord,
    CleanConfig,
    CleanPlan,
    CleanReport,
    ColumnProfile,
    ColumnRule,
    ProfileReport,
)

def profile(df: Any) -> ProfileReport:
    return resolve_backend(df).profile(df)

def suggest_plan(df: Any, config: CleanConfig | None = None) -> CleanPlan:
    cfg = config or CleanConfig()
    return resolve_backend(df).suggest_plan(df, cfg)

def apply_plan(
    df: Any,
    plan: CleanPlan,
    config: CleanConfig | None = None,
    return_report: bool = False,
) -> Any:
    cfg = config or CleanConfig()
    cleaned = resolve_backend(df).apply_plan(df, plan, cfg)
    if return_report:
        orig_r = len(df) if isinstance(df, (pd.DataFrame, pl.DataFrame)) else 0
        clean_r = len(cleaned) if isinstance(cleaned, (pd.DataFrame, pl.DataFrame)) else 0
        orig_c = len(df.columns) if hasattr(df, "columns") else 0
        clean_c = len(cleaned.columns) if hasattr(cleaned, "columns") else 0
        rep = CleanReport(orig_r, clean_r, orig_c, clean_c, plan.actions)
        return cleaned, rep
    return cleaned

def clean(
    df: Any,
    config: CleanConfig | None = None,
    return_report: bool = False,
) -> Any:
    cfg = config or CleanConfig()
    plan = suggest_plan(df, cfg)
    return apply_plan(df, plan, cfg, return_report=return_report)

class Cleaner:
    def __init__(self, config: CleanConfig | None = None):
        self.config = config or CleanConfig()

    def profile(self, df: Any) -> ProfileReport:
        return profile(df)

    def suggest_plan(self, df: Any) -> CleanPlan:
        return suggest_plan(df, self.config)

    def apply_plan(self, df: Any, plan: CleanPlan, return_report: bool = False) -> Any:
        return apply_plan(df, plan, self.config, return_report=return_report)

    def clean(self, df: Any, return_report: bool = False) -> Any:
        return clean(df, self.config, return_report=return_report)
