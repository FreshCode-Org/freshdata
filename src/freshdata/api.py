"""Minimal top-level API for freshdata.

The public surface is intentionally small:

    fd.clean(df, config=None, *, report=False, **options)
    fd.profile(df, config=None, **options)

Advanced behavior belongs in :class:`freshdata.CleanConfig` or lower-level
subpackages, not in a wide ``clean`` signature.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeVar, cast, overload

import pandas as pd

from ._reportframe import ReportFrame
from .adapters.polars import from_pandas, to_pandas
from .cleaner import Cleaner
from .config import CleanConfig, merge_options
from .engine.context import build_contexts
from .engine.model_select import EngineMode, rank_missing_models
from .profile import Profile, build_profile
from .report import CleanReport

FrameT = TypeVar("FrameT")
ConfigLike = CleanConfig | Mapping[str, object] | None

_REMOVED_CLEAN_KWARGS: dict[str, str] = {
    "return_report": "return_report= was removed; use report=True.",
    "domain": (
        "domain= was removed from fd.clean; use freshdata.domains.run_domain "
        "after core cleaning."
    ),
    "column_map": "column_map= was removed from fd.clean; pass it to the domain layer.",
    "gtfs_file": "gtfs_file= was removed from fd.clean; pass it to the domain layer.",
    "fhir_resource": "fhir_resource= was removed from fd.clean; pass it to the domain layer.",
    "media_type": "media_type= was removed from fd.clean; pass it to the domain layer.",
    "finance_mode": "finance_mode= was removed from fd.clean; pass it to the domain layer.",
    "audit_include_phi": (
        "audit_include_phi= was removed from fd.clean; pass it to the domain layer."
    ),
    "domain_kwargs": "domain_kwargs= was removed from fd.clean; pass kwargs to the domain layer.",
    "engine": (
        "engine= was removed from fd.clean; use freshdata.execution.run_with_engine "
        "for out-of-core execution."
    ),
    "output_format": (
        "output_format= was removed from fd.clean; use freshdata.execution.run_with_engine "
        "for native output formats."
    ),
    "engine_config": (
        "engine_config= was removed from fd.clean; use freshdata.execution.run_with_engine."
    ),
    "source_provenance": (
        "source_provenance= was removed from fd.clean; annotate provenance after cleaning."
    ),
    "provenance_confidence_threshold": (
        "provenance_confidence_threshold= was removed from fd.clean."
    ),
    "contract": "contract= was removed from fd.clean; run schema checks before cleaning.",
    "on_unexpected": (
        "on_unexpected= was removed from fd.clean; run schema checks before cleaning."
    ),
    "on_missing": "on_missing= was removed from fd.clean; run schema checks before cleaning.",
    "memory": "memory= was removed from fd.clean; replay learned behavior outside core cleaning.",
}

_RENAMED_PROFILE_KWARGS: dict[str, str] = {
    "include_plan": "include_plan= was renamed; use plan=True.",
    "profile_sample": "profile_sample= was renamed; use sample=<rows>.",
    "lazy_report": "lazy_report= was renamed; use lazy=True.",
}


def _reject_removed_kwargs(options: Mapping[str, object]) -> None:
    """Raise targeted migration errors for kwargs removed from ``fd.clean``."""
    removed = [name for name in options if name in _REMOVED_CLEAN_KWARGS]
    if removed:
        detail = " ".join(_REMOVED_CLEAN_KWARGS[name] for name in sorted(removed))
        raise TypeError(detail)


def _reject_profile_renames(options: Mapping[str, object]) -> None:
    renamed = [name for name in options if name in _RENAMED_PROFILE_KWARGS]
    if renamed:
        detail = " ".join(_RENAMED_PROFILE_KWARGS[name] for name in sorted(renamed))
        raise TypeError(detail)


def _build_config(config: ConfigLike, options: Mapping[str, object]) -> CleanConfig:
    """Build a :class:`CleanConfig` from an object or a lightweight mapping."""
    _reject_removed_kwargs(options)
    overrides = dict(options)
    if config is None:
        return merge_options(None, **overrides)
    if isinstance(config, CleanConfig):
        return merge_options(config, **overrides)
    if isinstance(config, Mapping):
        _reject_removed_kwargs(config)
        merged = dict(config)
        merged.update(overrides)
        return merge_options(None, **merged)
    raise TypeError(
        "config must be a CleanConfig, a mapping of CleanConfig fields, or None; "
        f"got {type(config).__name__}"
    )


@overload
def clean(
    df: FrameT,
    config: ConfigLike = None,
    *,
    report: Literal[False] = False,
    **options: object,
) -> FrameT:
    ...


@overload
def clean(
    df: FrameT,
    config: ConfigLike = None,
    *,
    report: Literal[True],
    **options: object,
) -> tuple[FrameT, CleanReport]:
    ...


@overload
def clean(
    df: FrameT,
    config: ConfigLike = None,
    *,
    report: bool,
    **options: object,
) -> FrameT | tuple[FrameT, CleanReport]:
    ...


def clean(
    df: FrameT,
    config: ConfigLike = None,
    *,
    report: bool = False,
    **options: object,
) -> FrameT | tuple[FrameT, CleanReport]:
    """Clean a pandas or polars DataFrame and return the same frame backend.

    Core usage stays small and predictable:

    >>> cleaned = fd.clean(df)
    >>> cleaned, audit = fd.clean(df, report=True)
    >>> cleaned = fd.clean(df, {"strategy": "conservative"}, verbose=False)

    Advanced cleaning options are :class:`CleanConfig` fields and may be passed
    either through ``config=`` or as keyword overrides. Removed v1 routing
    keywords raise clear ``TypeError`` migration messages.
    """
    cfg = _build_config(config, options)
    cleaner = Cleaner(config=cfg)
    result = cleaner.clean(df, report=report)
    if report:
        cleaned, rep = cast(tuple[pd.DataFrame, CleanReport], result)
        return cast(FrameT, from_pandas(cleaned, df)), rep
    return cast(FrameT, from_pandas(cast(pd.DataFrame, result), df))


def _infer_roles(
    df: pd.DataFrame,
    config: ConfigLike = None,
    **options: object,
) -> pd.DataFrame:
    """Infer column roles for internal explanation/reporting surfaces."""
    cfg = _build_config(config, options)
    frame = to_pandas(df)
    contexts = build_contexts(frame, cfg)
    mode = cast(EngineMode, cfg.engine_mode or "balanced")
    rows: list[dict[str, object]] = []
    for col, ctx in sorted(contexts.items()):
        primary = None
        if ctx.missing_ratio > 0:
            primary = rank_missing_models(frame, col, ctx, cfg, mode=mode).primary
        rows.append(
            {
                "column": col,
                "role": ctx.role,
                "missing_pct": round(ctx.missing_ratio * 100, 2),
                "cardinality": ctx.nunique,
                "skew": ctx.skew,
                "domain_sensitive": ctx.domain_sensitive,
                "primary_missing_model": primary.model_id if primary else None,
            }
        )
    return ReportFrame.wrap(pd.DataFrame(rows), "infer_roles")


def profile(
    df: pd.DataFrame,
    config: ConfigLike = None,
    **options: object,
) -> Profile:
    """Inspect a DataFrame without changing it.

    Profile-only controls live in ``**options``:

    - ``plan=True`` attaches ``profile.plan``.
    - ``sample=N`` profiles a deterministic row sample.
    - ``max_columns=M`` profiles the first ``M`` columns.
    - ``lazy=True`` skips the expensive duplicate-row scan.
    """
    _reject_profile_renames(options)
    opts = dict(options)
    include_plan = bool(opts.pop("plan", False))
    sample = cast(int | None, opts.pop("sample", None))
    max_columns = cast(int | None, opts.pop("max_columns", None))
    lazy = bool(opts.pop("lazy", False))

    cfg = _build_config(config, opts)
    frame = to_pandas(df)
    prof = build_profile(frame, cfg, sample=sample, max_columns=max_columns, lazy=lazy)
    if include_plan:
        from .plan import suggest_plan  # noqa: PLC0415

        object.__setattr__(prof, "plan", suggest_plan(frame, config=cfg))
    return prof
