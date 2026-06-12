"""Read-only data profiling: what is in this frame, and what would clean() do?

:func:`build_profile` reuses the exact inference code from the cleaning steps,
so every "would convert to …" suggestion is a faithful preview, not a guess.
Profiling never modifies the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_bool_dtype, is_numeric_dtype

from ._util import _is_stringlike_dtype, format_bytes, memory_bytes
from .config import CleanConfig
from .steps.dtypes import suggest_conversion
from .steps.outliers import _bounds
from .steps.strings import active_sentinels, normalize_text


@dataclass(frozen=True)
class ColumnProfile:
    """Statistics and detected issues for one column."""

    name: str
    dtype: str
    non_null: int
    missing: int
    missing_pct: float
    unique: int | None  # None when values are unhashable
    sample_values: list[Any]
    suggested_dtype: str | None
    issues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Profile:
    """A whole-table profile. Render with ``print(profile)``, export with
    :meth:`to_frame` or :meth:`to_dict`."""

    n_rows: int
    n_cols: int
    memory: int
    duplicate_rows: int | None  # None when rows are unhashable
    missing_cells: int
    missing_pct: float
    columns: list[ColumnProfile]

    def to_frame(self) -> pd.DataFrame:
        """One row per column — convenient to sort/filter in a notebook."""
        return pd.DataFrame(
            {
                "dtype": [c.dtype for c in self.columns],
                "non_null": [c.non_null for c in self.columns],
                "missing": [c.missing for c in self.columns],
                "missing_pct": [round(c.missing_pct, 2) for c in self.columns],
                "unique": [c.unique for c in self.columns],
                "suggested_dtype": [c.suggested_dtype for c in self.columns],
                "issues": ["; ".join(c.issues) for c in self.columns],
            },
            index=pd.Index([c.name for c in self.columns], name="column"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_bytes": self.memory,
            "duplicate_rows": self.duplicate_rows,
            "missing_cells": self.missing_cells,
            "missing_pct": self.missing_pct,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "non_null": c.non_null,
                    "missing": c.missing,
                    "missing_pct": c.missing_pct,
                    "unique": c.unique,
                    "sample_values": c.sample_values,
                    "suggested_dtype": c.suggested_dtype,
                    "issues": list(c.issues),
                }
                for c in self.columns
            ],
        }

    @property
    def n_issues(self) -> int:
        return sum(len(c.issues) for c in self.columns)

    def __str__(self) -> str:
        dup = "n/a" if self.duplicate_rows is None else f"{self.duplicate_rows:,}"
        lines = [
            f"freshdata profile — {self.n_rows:,} rows x {self.n_cols:,} columns, "
            f"{format_bytes(self.memory)}",
            f"  missing cells: {self.missing_cells:,} ({self.missing_pct:.1f}%)   "
            f"duplicate rows: {dup}",
        ]
        if not self.columns:
            return "\n".join(lines)
        name_w = min(28, max(6, *(len(c.name) for c in self.columns)))
        dtype_w = max(5, *(len(c.dtype) for c in self.columns))
        lines.append(f"  {'column':<{name_w}}  {'dtype':<{dtype_w}}  {'missing':>8}  issues")
        for c in self.columns:
            name = c.name if len(c.name) <= name_w else c.name[: name_w - 1] + "…"
            missing = f"{c.missing_pct:.0f}%" if c.missing else "-"
            issues = "; ".join(c.issues) if c.issues else "-"
            lines.append(f"  {name:<{name_w}}  {c.dtype:<{dtype_w}}  {missing:>8}  {issues}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<Profile: {self.n_rows:,} rows x {self.n_cols} cols, "
            f"{self.n_issues} issue(s) across {len(self.columns)} column(s)>"
        )


def _safe_nunique(s: pd.Series) -> int | None:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return None


def _sample_values(s: pd.Series, k: int = 3) -> list[Any]:
    nonnull = s.dropna()
    try:
        values = nonnull.unique()[:k].tolist()
    except TypeError:
        values = nonnull.head(k).tolist()
    # numpy scalars -> plain Python so to_dict() output serializes cleanly
    return [v.item() if isinstance(v, np.generic) else v for v in values]


def _profile_column(name: str, s: pd.Series, config: CleanConfig,
                    sentinels: frozenset) -> ColumnProfile:
    n = len(s)
    missing = int(s.isna().sum())
    non_null = n - missing
    issues: list[str] = []

    if missing:
        issues.append(f"{100.0 * missing / n:.1f}% missing")

    suggested: str | None = None
    kind = infer_dtype(s, skipna=True)
    if kind in ("mixed", "mixed-integer"):
        issues.append("mixed value types")

    is_textual = _is_stringlike_dtype(s.dtype)
    if is_textual and non_null:
        normalized, n_stripped, n_sentinels = normalize_text(s, config, sentinels)
        if n_stripped:
            issues.append(f"{n_stripped} value(s) with surrounding whitespace")
        if n_sentinels:
            issues.append(f"{n_sentinels} sentinel value(s) meaning missing")
        target, converted, n_coerced = suggest_conversion(normalized, config)
        if converted is not None:
            suggested = str(converted.dtype)
            note = f", {n_coerced} unparseable" if n_coerced else ""
            issues.append(f"would convert to {suggested}{note}")

    if is_numeric_dtype(s) and not is_bool_dtype(s) and non_null >= 20:
        bounds = _bounds(s, config)
        if bounds is not None:
            n_outliers = int(((s < bounds[0]) | (s > bounds[1])).sum())
            if n_outliers:
                issues.append(
                    f"{n_outliers} potential outlier(s) ({config.outlier_method})"
                )

    unique = _safe_nunique(s)
    if n > 1 and unique is not None and unique <= 1:
        issues.append("constant column")
    if is_textual and unique is not None and non_null >= 20 and unique == non_null:
        issues.append("all values unique (possible identifier)")

    return ColumnProfile(
        name=str(name),
        dtype=str(s.dtype),
        non_null=non_null,
        missing=missing,
        missing_pct=100.0 * missing / n if n else 0.0,
        unique=unique,
        sample_values=_sample_values(s),
        suggested_dtype=suggested,
        issues=issues,
    )


def build_profile(df: pd.DataFrame, config: CleanConfig) -> Profile:
    """Profile *df* without modifying it."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
    try:
        duplicate_rows: int | None = int(df.duplicated().sum())
    except TypeError:
        duplicate_rows = None
    n_cells = int(df.size)
    missing_cells = int(df.isna().sum().sum())
    sentinels = active_sentinels(config)
    # Positional access tolerates duplicate column labels.
    columns = [
        _profile_column(df.columns[i], df.iloc[:, i], config, sentinels)
        for i in range(df.shape[1])
    ]
    return Profile(
        n_rows=len(df),
        n_cols=df.shape[1],
        memory=memory_bytes(df),
        duplicate_rows=duplicate_rows,
        missing_cells=missing_cells,
        missing_pct=100.0 * missing_cells / n_cells if n_cells else 0.0,
        columns=columns,
    )
