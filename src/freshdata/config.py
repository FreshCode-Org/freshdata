# src/freshdata/config.py
from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_NULL_SENTINELS: tuple[str, ...] = (
    "",
    "null",
    "NULL",
    "None",
    "none",
    "N/A",
    "n/a",
    "NA",
    "na",
    "NaN",
    "nan",
    "-",
    "--",
    "?",
)


@dataclass
class CleanConfig:
    strategy: str = "balanced"  # "conservative" | "balanced" | "aggressive"
    strip_whitespace: bool = True
    normalize_sentinels: bool = True
    null_sentinels: tuple[str, ...] = field(default_factory=lambda: DEFAULT_NULL_SENTINELS)
    fix_dtypes: bool = True
    drop_duplicates: bool = False
    duplicate_keep: str = "first"  # "first" | "last" | "any"
    duplicate_subset: list[str] | None = None
    impute_missing: bool = True
    imputer: str = "auto"  # "auto" | "median" | "knn"
    handle_outliers: bool = False
    outlier_iqr_multiplier: float = 3.0
    target_column: str | None = None
