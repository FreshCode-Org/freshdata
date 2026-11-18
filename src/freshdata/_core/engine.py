# src/freshdata/_core/engine.py
from __future__ import annotations
import dataclasses
from typing import Any
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

def _with_overrides(config: CleanConfig | None, overrides: dict[str, Any]) -> CleanConfig:
    cfg = config or CleanConfig()
    return dataclasses.replace(cfg, **overrides) if overrides else cfg

def _column_count(frame: Any) -> int:
    if hasattr(frame, "collect_schema"):  # Polars: schema only, no data and no PerformanceWarning
        return len(frame.collect_schema())
    return len(frame.columns) if hasattr(frame, "columns") else 0

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
    cleaned: Any = resolve_backend(df).apply_plan(df, plan, cfg)
    if return_report:
        orig_r = 0 if type(df).__name__ == "LazyFrame" else len(df)
        clean_r = 0 if type(cleaned).__name__ == "LazyFrame" else len(cleaned)
        orig_c = _column_count(df)
        clean_c = _column_count(cleaned)
        rep = CleanReport(orig_r, clean_r, orig_c, clean_c, plan.actions)
        return cleaned, rep
    return cleaned

def clean(
    df: Any,
    config: CleanConfig | None = None,
    return_report: bool = False,
    **overrides: Any,
) -> Any:
    """Plan and apply in one call. Keyword overrides replace CleanConfig fields."""
    cfg = _with_overrides(config, overrides)
    plan = suggest_plan(df, cfg)
    return apply_plan(df, plan, cfg, return_report=return_report)

class Cleaner:
    def __init__(self, config: CleanConfig | None = None, **overrides: Any):
        self.config = _with_overrides(config, overrides)

    def profile(self, df: Any) -> ProfileReport:
        return profile(df)

    def suggest_plan(self, df: Any) -> CleanPlan:
        return suggest_plan(df, self.config)

    def apply_plan(self, df: Any, plan: CleanPlan, return_report: bool = False) -> Any:
        return apply_plan(df, plan, self.config, return_report=return_report)

    def clean(self, df: Any, return_report: bool = False) -> Any:
        return clean(df, self.config, return_report=return_report)
