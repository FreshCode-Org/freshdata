# src/freshdata/_core/models.py
from __future__ import annotations
import warnings
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any, Literal, Mapping

_ROLES = ("id", "target", "numeric", "categorical", "datetime", "text")
_IMPUTES = ("preserve", "drop", "mean", "median", "mode", "ffill")
_OUTLIERS = ("preserve", "clip", "drop", "flag")

def _check_choice(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is not valid. Use one of: {', '.join(allowed)}.")

def _warn_no_effect(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=4)

@dataclass(frozen=True)
class ColumnRule:
    role: Literal["id", "target", "numeric", "categorical", "datetime", "text"] | None = None
    impute: Literal["preserve", "drop", "mean", "median", "mode", "ffill"] | None = None
    outliers: Literal["preserve", "clip", "drop", "flag"] | None = None
    protected: bool = False

    def __post_init__(self) -> None:
        for name, allowed in (("role", _ROLES), ("impute", _IMPUTES), ("outliers", _OUTLIERS)):
            if getattr(self, name) is not None:
                _check_choice(name, getattr(self, name), allowed)
        if self.role not in (None, "id", "target"):
            _warn_no_effect(f"ColumnRule(role={self.role!r}) has no effect. Only 'id' and 'target' (protect the column) do.")

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
    default_impute: Literal["auto", "preserve", "drop", "mean", "median", "mode", "ffill"] = "auto"
    default_outliers: Literal["auto", "preserve", "clip", "drop", "flag"] = "preserve"
    iqr_factor: float = 1.5
    strict_backend: bool = True

    def __post_init__(self) -> None:
        _check_choice("strategy", self.strategy, ("conservative", "balanced", "aggressive"))
        _check_choice("default_impute", self.default_impute, ("auto", *_IMPUTES))
        _check_choice("default_outliers", self.default_outliers, ("auto", *_OUTLIERS))
        if isinstance(self.id_columns, str):
            raise ValueError(f"id_columns must be a tuple of column names, not the string {self.id_columns!r}.")
        object.__setattr__(self, "id_columns", tuple(self.id_columns))
        for col, rule in self.rules.items():
            if not isinstance(rule, ColumnRule):
                raise TypeError(f"rules[{col!r}] must be a ColumnRule, not {type(rule).__name__}.")
        if not isinstance(self.iqr_factor, Real) or isinstance(self.iqr_factor, bool) or self.iqr_factor < 0:
            raise ValueError(f"iqr_factor must be a number >= 0, not {self.iqr_factor!r}.")
        if self.strategy != "balanced":
            _warn_no_effect(f"CleanConfig(strategy={self.strategy!r}) has no effect in freshdata 3.x.")
        if self.default_outliers == "auto":
            _warn_no_effect("CleanConfig(default_outliers='auto') leaves outliers unchanged, like 'preserve'.")

@dataclass(frozen=True)
class ActionRecord:
    stage: str
    column: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    rows_affected: int = 0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class CleanPlan:
    actions: tuple[ActionRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"actions": [a.to_dict() for a in self.actions]}

    def diff(self, other: CleanPlan) -> dict[str, list[Any]]:
        self_map = {(a.stage, a.column, a.action): a for a in self.actions}
        other_map = {(a.stage, a.column, a.action): a for a in other.actions}

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

    @property
    def total_actions(self) -> int:
        return len(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
