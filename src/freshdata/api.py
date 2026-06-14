"""Top-level convenience functions: ``fd.clean(df)`` and ``fd.profile(df)``."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .cleaner import Cleaner
from .config import CleanConfig, merge_options
from .profile import Profile, build_profile
from .report import CleanReport

#: Distinguishes "argument not given" from a meaningful ``None``
#: (e.g. ``outlier_action=None`` means "detect but preserve outliers").
_UNSET: Any = object()


def clean(
    df: pd.DataFrame,
    *,
    strategy: str | None = None,
    missing_threshold_low: float | None = None,
    missing_threshold_medium: float | None = None,
    missing_threshold_high: float | None = None,
    duplicate_threshold: float | None = None,
    outlier_method: str | None = None,
    outlier_action: str | None = _UNSET,
    preserve_original: bool | None = None,
    return_report: bool = False,
    verbose: bool | None = None,
    report: bool = False,
    config: CleanConfig | None = None,
    **options: object,
) -> pd.DataFrame | tuple[pd.DataFrame, CleanReport]:
    """Clean a DataFrame and return a new, repaired one.

    Two layers run in order. **Representation repair** always happens first:

    1.  ``column_names`` — snake_case column names, deduplicate collisions.
    2.  ``strip_whitespace`` — trim surrounding whitespace in text cells.
    3.  ``normalize_sentinels`` — turn "N/A", "null", "-", "" … into missing.
    4.  ``drop_empty_columns`` / ``drop_empty_rows`` — remove all-missing ones.
    5.  ``fix_dtypes`` — text that is really numeric / datetime / boolean gets
        the right dtype (validated; ``numeric_threshold`` of values must parse).
    6.  ``drop_duplicates`` — resolve duplicate rows (``duplicate_keep``
        chooses first/last/drop/aggregate; time-indexed frames are protected).

    Then, with ``strategy="auto"`` (the default), the **decision engine**
    profiles every column — missing ratio, dtype, skewness, cardinality,
    inferred role (id / target / datetime / text / categorical), whether
    missingness looks informative — and applies threshold rules for missing
    values and outliers. Nothing is done silently: every action (including
    deliberately preserving a column) is logged with a rationale, a risk
    level, and a confidence score. ``strategy="conservative"`` disables the
    engine; imputation and outlier handling are then opt-in via ``impute=`` /
    ``outliers=``.

    Parameters
    ----------
    df:
        The DataFrame to clean.
    strategy:
        ``"auto"`` (default — run the decision engine) or ``"conservative"``.
    missing_threshold_low / missing_threshold_medium / missing_threshold_high:
        Band edges for the missing-value rules (defaults 0.05 / 0.30 / 0.60):
        low → impute (mean/median/mode/ffill by context), medium → robust
        impute (median, KNN, sentinel), high → keep only if important else
        drop, extreme (above high) → drop unless preserved or a label.
    duplicate_threshold:
        Duplicate-row ratio above which a data-quality warning is raised
        (default 0.10).
    outlier_method:
        ``"iqr"`` (default), ``"zscore"``, ``"auto"`` (z-score for ~normal
        columns, IQR for skewed), or ``"isolation_forest"`` (needs
        scikit-learn; falls back to IQR).
    outlier_action:
        ``"auto"`` (default) — context-aware: flag under ``strategy="balanced"``,
        cap under ``"aggressive"``. ``"cap"`` (winsorize to the fences),
        ``"remove"``, and ``"flag"`` are explicit directives applied to every
        eligible numeric column (heavy-tailed columns are still acted on, with a
        warning); ``None`` detects and reports, changing nothing.
    preserve_original:
        Default True: the input frame is never mutated. False allows in-place
        reuse of the input's memory.
    return_report:
        If True, return ``(cleaned_df, CleanReport)``. The report carries
        per-action rationale/risk/confidence, missing counts before/after,
        warnings, and recommendations for manual review. (``report=True`` is
        an equivalent alias kept for backward compatibility.)
    verbose:
        Default True: print a one-line summary (plus any warnings) per clean.
    config:
        A prebuilt :class:`~freshdata.CleanConfig` to start from.
    **options:
        Any other :class:`~freshdata.CleanConfig` field as a keyword override
        (e.g. ``preserve_columns``, ``target_column``, ``duplicate_keep``,
        ``impute``, ``outliers``). Unknown names raise :class:`TypeError`.

    Examples
    --------
    >>> import freshdata as fd
    >>> cleaned = fd.clean(df)
    >>> cleaned, rep = fd.clean(df, return_report=True)
    >>> print(rep.summary())

    >>> fd.clean(df, outlier_action="flag", target_column="churn",
    ...          preserve_columns=("notes",), verbose=False)
    """
    explicit = {
        "strategy": strategy,
        "missing_threshold_low": missing_threshold_low,
        "missing_threshold_medium": missing_threshold_medium,
        "missing_threshold_high": missing_threshold_high,
        "duplicate_threshold": duplicate_threshold,
        "outlier_method": outlier_method,
        "preserve_original": preserve_original,
        "verbose": verbose,
    }
    options.update({k: v for k, v in explicit.items() if v is not None})
    if outlier_action is not _UNSET:
        options["outlier_action"] = outlier_action
    cleaner = Cleaner(config=config, **options)
    return cleaner.clean(df, report=report or return_report)


def profile(
    df: pd.DataFrame,
    *,
    config: CleanConfig | None = None,
    **options: object,
) -> Profile:
    """Inspect a DataFrame without changing it.

    Returns a :class:`~freshdata.Profile` describing shape, memory, missing
    data, duplicates, and per-column issues — including a faithful preview of
    the dtype conversions :func:`clean` would perform, computed by the same
    inference code.

    Examples
    --------
    >>> import freshdata as fd
    >>> p = fd.profile(df)
    >>> print(p)             # human-readable issue table
    >>> p.to_frame()         # one row per column, sortable in a notebook
    >>> p.to_dict()          # JSON-friendly
    """
    return build_profile(df, merge_options(config, **options))
