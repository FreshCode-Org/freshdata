# src/freshdata/api.py
from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl

from freshdata.config import CleanConfig
from freshdata.execution import DuckDBEngine, EngineConfig, PandasEngine, PolarsEngine
from freshdata.report import CleanReport


def clean(
    df: pd.DataFrame | pl.DataFrame,
    *,
    config: CleanConfig | None = None,
    engine: str | None = None,
    engine_config: EngineConfig | None = None,
    return_report: bool = False,
    **kwargs: Any,
) -> pd.DataFrame | pl.DataFrame | tuple[pd.DataFrame | pl.DataFrame, CleanReport]:
    """Explainable, schema-preserving tabular cleaning across pandas, Polars, and DuckDB."""
    cfg = config or CleanConfig()
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    input_is_polars = isinstance(df, pl.DataFrame)
    resolved_engine = engine or (engine_config.engine if engine_config else ("polars" if input_is_polars else "pandas"))
    eng_cfg = engine_config or EngineConfig(engine=resolved_engine)

    pd_in = df.to_pandas() if input_is_polars else df

    if resolved_engine == "pandas":
        cleaned_pd, report = PandasEngine.execute(pd_in, cfg, eng_cfg)
    elif resolved_engine == "polars":
        cleaned_pd, report = PolarsEngine.execute(pd_in, cfg, eng_cfg)
    elif resolved_engine == "duckdb":
        cleaned_pd, report = DuckDBEngine.execute(pd_in, cfg, eng_cfg)
    else:
        raise ValueError(f"Unsupported engine: {resolved_engine!r}")

    out_frame: pd.DataFrame | pl.DataFrame = pl.from_pandas(cleaned_pd) if input_is_polars else cleaned_pd
    if return_report:
        return out_frame, report
    return out_frame


class PipelineBuilder:
    def __init__(self) -> None:
        self.config = CleanConfig()

    def strip_whitespace(self, enabled: bool = True) -> PipelineBuilder:
        self.config.strip_whitespace = enabled
        return self

    def normalize_sentinels(self, enabled: bool = True) -> PipelineBuilder:
        self.config.normalize_sentinels = enabled
        return self

    def fix_dtypes(self, enabled: bool = True) -> PipelineBuilder:
        self.config.fix_dtypes = enabled
        return self

    def handle_missing(self, strategy: str = "balanced") -> PipelineBuilder:
        self.config.strategy = strategy
        self.config.impute_missing = True
        return self

    def run(self, df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame | pl.DataFrame:
        res = clean(df, config=self.config)
        if isinstance(res, tuple):
            return res[0]
        return res


def pipeline() -> PipelineBuilder:
    return PipelineBuilder()


class PlanReport:
    def __init__(self, report: CleanReport) -> None:
        self.report = report

    def summary(self) -> str:
        return self.report.summary()


def plan(df: pd.DataFrame | pl.DataFrame) -> PlanReport:
    res = clean(df, return_report=True)
    assert isinstance(res, tuple)
    return PlanReport(res[1])


def apply(plan_obj: PlanReport, df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame | pl.DataFrame:
    res = clean(df, return_report=False)
    if isinstance(res, tuple):
        return res[0]
    return res
