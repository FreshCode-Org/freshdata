"""Opt-in missing-value imputation.

Off by default: filling values changes the statistics of the data, so the user
must ask for it (``impute="auto" | "mean" | "median" | "mode"``). The "auto"
strategy uses the median for numeric columns and the mode for everything else.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .._util import exact_int_stat, exceeds_float64_exact, fill_na_exact, safe_median
from ..config import CleanConfig
from ..report import CleanReport


def _mode_value(s: pd.Series) -> Any | None:
    """Most frequent non-missing value; deterministic for ties when sortable."""
    try:
        modes = s.mode(dropna=True)
        if len(modes):
            return modes.iloc[0]
    except TypeError:
        pass  # mixed un-comparable types; fall back to frequency order
    counts = s.value_counts(dropna=True)
    return counts.index[0] if len(counts) else None


def _fill_value(s: pd.Series, strategy: str) -> Any | None:
    numeric = is_numeric_dtype(s) and not is_bool_dtype(s)
    if strategy == "auto":
        strategy = "median" if numeric else "mode"
    if strategy in ("mean", "median"):
        if not numeric:
            return None  # not defined for this dtype; caller reports the skip
        if exceeds_float64_exact(s):
            return exact_int_stat(s, strategy)  # a float statistic would lose digits
        return s.mean() if strategy == "mean" else safe_median(s)
    return _mode_value(s)


def _strategy_for_column(col: object, config: CleanConfig) -> str | None:
    if config.impute_strategy and str(col) in config.impute_strategy:
        return config.impute_strategy[str(col)]
    return config.impute


def _declared_roles(config: CleanConfig, columns: Iterable[object]) -> dict[str, str]:
    """Declared identifier and target columns present in *columns*, by role.

    Names resolve like context-protected columns (exact, else snake case), so a
    declared ``"Customer ID"`` still matches ``customer_id`` after renaming. A
    column declared as both is reported as the target.
    """
    from ..guard import _match_columns  # noqa: PLC0415 — cycle-safe lazy import

    names = [str(c) for c in columns]
    roles: dict[str, str] = {}
    if config.target_column is not None:
        for name in _match_columns([str(config.target_column)], names):
            if name in names:
                roles[name] = "target"
    for name in _match_columns([str(c) for c in config.id_columns], names):
        if name in names:
            roles.setdefault(name, "identifier")
    return roles


def impute_missing(df: pd.DataFrame, config: CleanConfig,
                   report: CleanReport) -> pd.DataFrame:
    """Fill missing values per column according to explicit impute config.

    Context-protected columns and the declared ``id_columns`` and
    ``target_column`` are never filled, whatever ``impute`` or
    ``impute_strategy`` says: imputing an identifier corrupts keys and imputing
    the target leaks into it. They can still inform ``"missforest"`` as
    features for other columns.
    """
    if config.impute is None and not config.impute_strategy:
        return df
    from ..guard import hard_protected_columns  # noqa: PLC0415 — cycle-safe lazy import

    protected = hard_protected_columns(config, df.columns)
    roles = _declared_roles(config, df.columns)
    for name, declared_role in roles.items():
        if config.impute_strategy and name in config.impute_strategy:
            report.add_warning(
                f"impute_strategy for '{name}' ignored: it is the declared "
                f"{declared_role} column")
    # MissForest applies its own role gates (target and identifier columns are
    # preserved with an audited fallback action), so declared roles stay in its
    # column list and are reported there.
    missforest_columns = [
        col for col in df.columns
        if str(col) not in protected
        and _strategy_for_column(col, config) == "missforest"
        and int(df[col].isna().sum()) > 0
    ]
    if missforest_columns:
        from ..engine.context import build_contexts  # noqa: PLC0415
        from ..imputation.missforest import MissForestImputer  # noqa: PLC0415

        df = MissForestImputer(config, report).impute(
            df,
            missforest_columns,
            build_contexts(df, config),
        )

    for col in df.columns:
        if str(col) in protected:
            continue  # context-protected columns must stay byte-identical
        strategy = _strategy_for_column(col, config)
        if strategy is None or strategy == "missforest":
            continue
        role = roles.get(str(col))
        if role is not None:
            if int(df[col].isna().sum()) and df[col].notna().any():
                report.add("impute", f"skipped: {role} column", column=str(col))
            continue
        s = df[col]
        n_missing = int(s.isna().sum())
        if n_missing == 0 or s.notna().sum() == 0:
            continue  # nothing to fill, or nothing to learn a fill value from
        value = _fill_value(s, strategy)
        if value is None or pd.isna(value):
            if strategy in ("mean", "median"):
                report.add("impute",
                           f"skipped ({strategy} is not defined for dtype {s.dtype})",
                           column=str(col))
            continue
        try:
            filled, cast_note = fill_na_exact(s, value)
        except (TypeError, ValueError):
            # e.g. value not representable in this dtype
            report.add("impute", f"skipped (could not fill dtype {s.dtype})",
                       column=str(col))
            continue
        df[col] = filled
        shown = f"{value:.6g}" if isinstance(value, float) else repr(value)
        report.add("impute",
                   f"filled {n_missing} missing value(s) with {strategy} ({shown}{cast_note})",
                   column=str(col), count=n_missing)
    return df
