"""The cleaning pipeline and the reusable :class:`Cleaner` front-end."""

from __future__ import annotations

import dataclasses
import time

import pandas as pd

from ._util import memory_bytes
from .adapters.polars import is_polars_frame, to_pandas
from .config import CleanConfig, merge_options
from .engine import auto_missing, auto_outliers
from .engine.cache import build_engine_cache
from .report import CleanReport
from .steps.columns import normalize_column_names
from .steps.dtypes import fix_dtypes
from .steps.duplicates import drop_duplicate_rows
from .steps.memory import optimize_memory
from .steps.missing import impute_missing
from .steps.outliers import handle_outliers
from .steps.prune import drop_constant_columns, drop_empty_columns, drop_empty_rows
from .steps.strings import clean_strings


def _validate_input(df: object, config: CleanConfig) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        raise TypeError(
            "freshdata works on DataFrames; got a Series. "
            "Convert it first with s.to_frame()."
        )
    if not isinstance(df, pd.DataFrame) and not is_polars_frame(df):
        raise TypeError(
            f"expected a pandas or polars DataFrame, got {type(df).__name__}"
        )
    frame = to_pandas(df)
    if frame.columns.duplicated().any() and not config.column_names:
        dupes = sorted({str(c) for c in frame.columns[frame.columns.duplicated()]})
        raise ValueError(
            f"DataFrame has duplicate column labels {dupes}, which makes "
            "column-wise cleaning ambiguous. Rename them, or leave "
            "column_names=True to deduplicate automatically."
        )
    return frame


def run_pipeline(df: pd.DataFrame, config: CleanConfig) -> tuple[pd.DataFrame, CleanReport]:
    """Run every enabled step, in a fixed and documented order.

    With ``preserve_original=True`` (the default) the input frame is never
    mutated: the pipeline works on a shallow copy and steps only rebind whole
    columns or build new frames, so the only extra memory used is for the
    columns that actually change. With ``preserve_original=False`` the
    pipeline may write into the input frame to save memory.

    After representation repair (names, strings, sentinels, empties, dtypes,
    duplicates), ``strategy="auto"`` runs the decision engine for missing
    values and outliers; explicit ``impute=`` / ``outliers=`` settings always
    override the corresponding engine stage.
    """
    df = _validate_input(df, config)
    report = CleanReport(
        rows_before=len(df),
        cols_before=df.shape[1],
        memory_before=memory_bytes(df),
        missing_before=int(df.isna().sum().sum()),
    )
    started = time.perf_counter()

    out = df.copy(deep=False) if config.preserve_original else df
    if config.column_names:
        out = normalize_column_names(out, report)
    out = clean_strings(out, config, report)
    if config.drop_empty_columns:
        out = drop_empty_columns(out, report)
    if config.drop_empty_rows:
        out = drop_empty_rows(out, report)
    if config.fix_dtypes:
        out = fix_dtypes(out, config, report)
    if config.drop_constant_columns:
        out = drop_constant_columns(out, config, report)
    if config.drop_duplicates:
        out = drop_duplicate_rows(out, config, report)
    if config.engine_mode is not None:
        cache = build_engine_cache(out, config)
        out = auto_missing(out, config, report, contexts=cache.contexts,
                           numeric_corr=cache.numeric_corr)
        out = auto_outliers(out, config, report, contexts=cache.contexts)
    out = impute_missing(out, config, report)
    out = handle_outliers(out, config, report)
    out = optimize_memory(out, config, report)
    if config.reset_index:
        out = out.reset_index(drop=True)

    report.rows_after = len(out)
    report.cols_after = out.shape[1]
    report.memory_after = memory_bytes(out)
    report.missing_after = int(out.isna().sum().sum())
    report.duration_seconds = time.perf_counter() - started
    return out, report


class Cleaner:
    """A configured, reusable cleaning pipeline.

    Useful when the same settings are applied to many frames (e.g. every file
    in a directory), or when you want the report after the fact::

        cleaner = fd.Cleaner(impute="median", drop_constant_columns=True)
        for path in paths:
            cleaned = cleaner.clean(pd.read_csv(path))
            print(cleaner.report_.summary())

    Attributes
    ----------
    config:
        The immutable :class:`~freshdata.CleanConfig` in effect.
    report_:
        The :class:`~freshdata.CleanReport` from the most recent
        :meth:`clean` call (``None`` before the first call).
    """

    def __init__(self, config: CleanConfig | None = None, **options: object) -> None:
        self.config: CleanConfig = merge_options(config, **options)
        self.report_: CleanReport | None = None

    def clean(
        self, df: pd.DataFrame, *, report: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, CleanReport]:
        """Clean *df* and return the result (the input is left unchanged
        unless ``preserve_original=False`` was configured).

        With ``report=True``, returns ``(cleaned_df, CleanReport)`` instead.
        The latest report is always available as :attr:`report_`.
        """
        cleaned, rep = run_pipeline(df, self.config)
        self.report_ = rep
        if self.config.verbose:
            print(rep.brief())
        return (cleaned, rep) if report else cleaned

    def __repr__(self) -> str:
        defaults = CleanConfig()
        overrides = {
            f.name: getattr(self.config, f.name)
            for f in dataclasses.fields(CleanConfig)
            if getattr(self.config, f.name) != getattr(defaults, f.name)
        }
        inner = ", ".join(f"{k}={v!r}" for k, v in overrides.items())
        return f"Cleaner({inner})"
