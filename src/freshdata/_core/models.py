# src/freshdata/_core/models.py
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

@dataclass(frozen=True)
class ColumnRule:
    role: Literal["id", "target", "numeric", "categorical", "datetime", "text"] | None = None
    impute: Literal["preserve", "drop", "mean", "median", "mode", "ffill"] | None = None
    outliers: Literal["preserve", "clip", "drop", "flag"] | None = None
    protected: bool = False

@dataclass(frozen=True)
class CleanConfig:
    strategy: Literal["conservative", "balanced", "aggressive"] = "balanced"
    rules: Mapping[str, ColumnRule] = field(default_factory=dict)
    id_columns: tuple[str, ...] = ()
    target_column: str | None = None
    normalize_names: bool = True
    strip_strings: bool = True
    fix_dtypes: bool = True
    drop_duplicates: bool = False
    default_impute: Literal["auto", "preserve", "median", "mode"] = "auto"
    default_outliers: Literal["auto", "preserve", "clip", "flag"] = "preserve"
    iqr_factor: float = 1.5
    strict_backend: bool = True

@dataclass(frozen=True)
class ActionRecord:
    stage: str
    column: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    rows_affected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class CleanPlan:
    actions: tuple[ActionRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"actions": [a.to_dict() for a in self.actions]}

    def diff(self, other: CleanPlan) -> dict[str, list[Any]]:
        self_map = {(a.stage, a.column): a for a in self.actions}
        other_map = {(a.stage, a.column): a for a in other.actions}

        added = [a for k, a in other_map.items() if k not in self_map]
        removed = [a for k, a in self_map.items() if k not in other_map]
        modified = []
        for k, a in self_map.items():
            if k in other_map and a != other_map[k]:
                modified.append({
                    "stage": a.stage,
                    "column": a.column,
                    "before": a.to_dict(),
                    "after": other_map[k].to_dict(),
                })
        return {"added": added, "removed": removed, "modified": modified}

@dataclass(frozen=True)
class ColumnProfile:
    name: str
    inferred_role: str
    null_count: int
    null_ratio: float
    unique_count: int
    sample_values: tuple[Any, ...] = ()

@dataclass(frozen=True)
class ProfileReport:
    row_count: int
    column_count: int
    columns: Mapping[str, ColumnProfile]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class CleanReport:
    original_rows: int
    cleaned_rows: int
    original_cols: int
    cleaned_cols: int
    actions: tuple[ActionRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
