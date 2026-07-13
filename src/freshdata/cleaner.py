"""The cleaning pipeline and the reusable :class:`Cleaner` front-end."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping

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

ProgressCallback = Callable[[dict[str, object]], None]


def _validate_input(df: object, config: CleanConfig) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        raise TypeError(
            "freshdata works on DataFrames; got a Series. Convert it first with s.to_frame()."
        )
    if not isinstance(df, pd.DataFrame) and not is_polars_frame(df):
        raise TypeError(f"expected a pandas or polars DataFrame, got {type(df).__name__}")
    frame = to_pandas(df)
    if frame.columns.duplicated().any() and not config.column_names:
        dupes = sorted({str(c) for c in frame.columns[frame.columns.duplicated()]})
        raise ValueError(
            f"DataFrame has duplicate column labels {dupes}, which makes "
            "column-wise cleaning ambiguous. Rename them, or leave "
            "column_names=True to deduplicate automatically."
        )
    return frame


def _emit_progress(
    callback: ProgressCallback | None,
    step: str,
    status: str,
    frame: pd.DataFrame,
) -> None:
    if callback is None:
        return
    callback(
        {
            "step": step,
            "status": status,
            "rows": len(frame),
            "columns": frame.shape[1],
        }
    )


def run_pipeline(  # noqa: PLR0915 - fixed-order pipeline orchestration
    df: pd.DataFrame,
    config: CleanConfig,
    *,
    memory: object | None = None,
    profile: object | None = None,
) -> tuple[pd.DataFrame, CleanReport]:
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

    ``memory`` (a :class:`~freshdata.CleaningMemory`, or ``None``) is passed
    through to the semantic stage so it can retrieve and replay compatible
    learned semantic repairs; every other step ignores it. ``profile`` (a
    :class:`~freshdata.learning.LearningProfile`, or ``None``) is likewise
    forwarded so the profile backend can replay learned value maps.
    """
    df = _validate_input(df, config)
    progress_callback = config.progress_callback
    _emit_progress(progress_callback, "input", "after", df)
    report = CleanReport(
        rows_before=len(df),
        cols_before=df.shape[1],
        memory_before=memory_bytes(df),
        missing_before=int(df.isna().sum().sum()),
    )
    started = time.perf_counter()

    if config.context is not None or config.policy is not None:
        # Compile/resolve the context policy against this frame's effective
        # schema and lower it into plain config fields. Lazily imported and
        # skipped entirely when no context is supplied (zero behaviour change).
        from .context import apply_policy_to_config  # noqa: PLC0415

        config = apply_policy_to_config(config, df=df, report=report)
        _emit_progress(progress_callback, "context", "after", df)

    out = df.copy(deep=False) if config.preserve_original else df
    if config.column_names:
        out = normalize_column_names(out, report)
        _emit_progress(progress_callback, "column_names", "after", out)

    # Hard protected-column guard (context policy / mutable=False): fold the
    # protected set into preserve_columns so drop/impute logic honors it, and
    # snapshot the columns now (post-rename) to verify byte-identity at the
    # end. Zero-cost when no context protection exists.
    from .guard import (  # noqa: PLC0415
        hard_protected_columns,
        snapshot_protected,
        verify_protected,
    )

    hard_protected = hard_protected_columns(config, out.columns)
    if hard_protected:
        missing_preserve = tuple(c for c in hard_protected if c not in config.preserve_columns)
        if missing_preserve:
            config = dataclasses.replace(
                config, preserve_columns=config.preserve_columns + missing_preserve
            )
        guard_snapshot = snapshot_protected(out, hard_protected)
    else:
        guard_snapshot = {}
    out = clean_strings(out, config, report)
    _emit_progress(progress_callback, "strings", "after", out)
    if config.drop_empty_columns:
        out = drop_empty_columns(out, report, config)
        _emit_progress(progress_callback, "empty_columns", "after", out)
    if config.drop_empty_rows:
        out = drop_empty_rows(out, report)
        _emit_progress(progress_callback, "empty_rows", "after", out)
    if config.fix_dtypes:
        out = fix_dtypes(out, config, report)
        _emit_progress(progress_callback, "dtypes", "after", out)
    if config.drop_constant_columns:
        out = drop_constant_columns(out, config, report)
        _emit_progress(progress_callback, "constant_columns", "after", out)
    if config.drop_duplicates:
        out = drop_duplicate_rows(out, config, report)
        _emit_progress(progress_callback, "duplicates", "after", out)
    if config.semantic_enabled:
        # Semantic cleaning runs after representation repair and before the
        # statistical engine, so missing/outlier logic sees repaired values.
        # Lazily imported to keep ``import freshdata`` light.
        from .semantic.apply import run_semantic  # noqa: PLC0415

        out = run_semantic(out, config, report, memory=memory, profile=profile)
        _emit_progress(progress_callback, "semantic", "after", out)
    if config.engine_mode is not None:
        cache = build_engine_cache(out, config)
        _emit_progress(progress_callback, "engine_cache", "after", out)
        out = auto_missing(
            out, config, report, contexts=cache.contexts, numeric_corr=cache.numeric_corr
        )
        _emit_progress(progress_callback, "engine_missing", "after", out)
        out = auto_outliers(out, config, report, contexts=cache.contexts)
        _emit_progress(progress_callback, "engine_outliers", "after", out)
    out = impute_missing(out, config, report)
    _emit_progress(progress_callback, "missing", "after", out)
    out = handle_outliers(out, config, report)
    _emit_progress(progress_callback, "outliers", "after", out)
    out = optimize_memory(out, config, report)
    _emit_progress(progress_callback, "memory", "after", out)
    if guard_snapshot:
        # Physical byte-identity check, before reset_index so row survivors
        # can still be aligned by their original index labels.
        verify_protected(out, guard_snapshot, report)
        _emit_progress(progress_callback, "protected_columns", "after", out)
    if config.reset_index:
        out = out.reset_index(drop=True)
        _emit_progress(progress_callback, "index", "after", out)

    report.rows_after = len(out)
    report.cols_after = out.shape[1]
    report.memory_after = memory_bytes(out)
    report.missing_after = int(out.isna().sum().sum())
    report.duration_seconds = time.perf_counter() - started
    _emit_progress(progress_callback, "complete", "after", out)
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

    def __init__(
        self,
        config: CleanConfig | Mapping[str, object] | None = None,
        **options: object,
    ) -> None:
        if isinstance(config, Mapping):
            merged = dict(config)
            merged.update(options)
            self._profile = merged.pop("profile", None)
            self.config = merge_options(None, **merged)
        else:
            self._profile = options.pop("profile", None)
            self.config = merge_options(config, **options)
        self.report_: CleanReport | None = None

    def clean(
        self,
        df: pd.DataFrame,
        *,
        report: bool = False,
        memory: object | None = None,
        profile: object | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, CleanReport]:
        """Clean *df* and return the result (the input is left unchanged
        unless ``preserve_original=False`` was configured).

        With ``report=True``, returns ``(cleaned_df, CleanReport)`` instead.
        The latest report is always available as :attr:`report_`. ``memory``
        (a :class:`~freshdata.CleaningMemory`) lets the semantic stage replay
        compatible learned repairs; see :func:`freshdata.clean`'s ``memory=``.
        ``profile`` (a :class:`~freshdata.learning.LearningProfile` or a path
        to a ``.fdprofile``) replays a learned profile; it overrides any
        profile the ``Cleaner`` was constructed with for this call.
        """
        effective_profile = profile if profile is not None else self._profile
        gate = None
        if effective_profile is not None:
            from .learning.replay import (  # noqa: PLC0415 - lazy import
                check_profile_drift,
                resolve_profile,
            )

            effective_profile = resolve_profile(effective_profile)
            gate = check_profile_drift(to_pandas(df), effective_profile)

        cleaned, rep = run_pipeline(
            df,
            self.config,
            memory=memory,
            profile=effective_profile if gate is not None and gate.ok else None,
        )
        if effective_profile is not None and gate is not None:
            from .learning.replay import annotate_profile_report  # noqa: PLC0415

            annotate_profile_report(rep, effective_profile, gate)
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
