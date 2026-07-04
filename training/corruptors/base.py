"""Corruptor core: labels, the corruptor container, application, composition.

Rules enforced here so individual corruptors cannot get them wrong:

- deterministic under ``seed`` (all randomness flows from one ``Random``);
- the input frame is never mutated;
- every mutated cell/row/header emits a machine-verifiable
  :class:`CorruptionLabel`;
- ``ambiguous`` or ``protected`` corruptions are never labeled auto-apply;
- composition never corrupts the same cell twice, so each label's
  ``clean_value`` stays the true pre-corruption value.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import pandas as pd

Kind = Literal["cell", "row", "header", "context"]

#: Transform families mirror FreshData's repair taxonomy.
TRANSFORM_FAMILIES = (
    "representation",
    "semantic_value",
    "reference_value",
    "date_time",
    "context_schema",
    "row_structure",
    "trap",
)


@dataclass(frozen=True)
class CorruptionLabel:
    """Machine-verifiable ground truth for one injected corruption."""

    raw_value: Any
    clean_value: Any
    column: str | None
    transform_family: str
    params: dict[str, Any] = field(default_factory=dict)
    should_repair: bool = True
    should_auto_apply: bool = True
    risk: str = "low"
    protected: bool = False
    ambiguous: bool = False
    corruptor: str = ""
    row: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_value": self.raw_value,
            "clean_value": self.clean_value,
            "column": self.column,
            "transform_family": self.transform_family,
            "params": dict(self.params),
            "should_repair": self.should_repair,
            "should_auto_apply": self.should_auto_apply,
            "risk": self.risk,
            "protected": self.protected,
            "ambiguous": self.ambiguous,
            "corruptor": self.corruptor,
            "row": self.row,
        }


#: A cell transform: (clean_value, rng, params) -> corrupted value, or None
#: when the corruptor does not apply to this value.
CellFn = Callable[[str, random.Random, dict[str, Any]], Any]
#: A frame transform for row/header/context corruptors:
#: (df, rng, params) -> (corrupted_df, labels).
FrameFn = Callable[
    [pd.DataFrame, random.Random, dict[str, Any]],
    "tuple[pd.DataFrame, list[CorruptionLabel]]",
]


@dataclass(frozen=True)
class Corruptor:
    """One named, parameterized corruption with fixed label semantics."""

    name: str
    family: str
    kind: Kind = "cell"
    fn: CellFn | None = None
    frame_fn: FrameFn | None = None
    risk: str = "low"
    should_repair: bool = True
    should_auto_apply: bool = True
    ambiguous: bool = False
    protected: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in TRANSFORM_FAMILIES:
            raise ValueError(f"{self.name}: unknown transform family {self.family!r}")
        if self.ambiguous and self.should_auto_apply:
            raise ValueError(f"{self.name}: ambiguous corruptors cannot be auto-apply")
        if self.protected and self.should_auto_apply:
            raise ValueError(f"{self.name}: protected traps cannot be auto-apply")
        if self.kind == "cell" and self.fn is None:
            raise ValueError(f"{self.name}: cell corruptor needs fn")
        if self.kind != "cell" and self.frame_fn is None:
            raise ValueError(f"{self.name}: {self.kind} corruptor needs frame_fn")

    def label_for(
        self, *, raw: Any, clean: Any, column: str | None, row: int | None,
        params: dict[str, Any],
    ) -> CorruptionLabel:
        return CorruptionLabel(
            raw_value=raw,
            clean_value=clean,
            column=column,
            transform_family=self.family,
            params=params,
            should_repair=self.should_repair,
            should_auto_apply=self.should_auto_apply,
            risk=self.risk,
            protected=self.protected,
            ambiguous=self.ambiguous,
            corruptor=self.name,
            row=row,
        )


def _seeded(name: str, seed: int) -> random.Random:
    return random.Random(f"freshdata-corruptor:{name}:{seed}")


def apply_corruptor(
    df: pd.DataFrame,
    corruptor: Corruptor,
    columns: Iterable[str] | None = None,
    *,
    seed: int = 0,
    share: float = 0.5,
    params: dict[str, Any] | None = None,
    skip_cells: set[tuple[int, str]] | None = None,
) -> tuple[pd.DataFrame, list[CorruptionLabel]]:
    """Apply one corruptor and return ``(corrupted_frame, labels)``.

    ``skip_cells`` (positions already corrupted by an earlier step) preserves
    ground truth under composition; :func:`compose` threads it automatically.
    """
    rng = _seeded(corruptor.name, seed)
    merged = {**corruptor.params, **(params or {})}
    if corruptor.kind != "cell":
        assert corruptor.frame_fn is not None
        out, labels = corruptor.frame_fn(df.copy(deep=True), rng, merged)
        return out, labels

    assert corruptor.fn is not None
    out = df.copy(deep=True)
    if columns is not None:
        targets = [c for c in columns if c in df.columns]
    else:
        # Default to string-shaped columns only: a bool/numeric column fed
        # through a text transform (e.g. casing) would otherwise trip a
        # dtype-widening cell write for no semantic reason.
        targets = [c for c in df.columns if df[c].map(lambda v: isinstance(v, str)).any()]
    labels: list[CorruptionLabel] = []
    skip = skip_cells or set()
    for col in targets:
        for pos, idx in enumerate(out.index):
            if (pos, str(col)) in skip:
                continue
            clean = out.at[idx, col]
            if not isinstance(clean, str):
                continue
            if pd.isna(clean) if not isinstance(clean, (list, dict)) else False:
                continue
            if rng.random() >= share:
                continue
            raw = corruptor.fn(str(clean), rng, merged)
            if raw is None or str(raw) == str(clean):
                continue
            out.at[idx, col] = raw
            # Sentinel-style corruptions destroy the original value; the only
            # valid repair target is "missing", never the unrecoverable truth.
            clean_target = None if merged.get("target") == "missing" else clean
            labels.append(corruptor.label_for(
                raw=raw, clean=clean_target, column=str(col), row=pos, params=merged,
            ))
    return out, labels


def compose(
    df: pd.DataFrame,
    steps: list[tuple[Corruptor, Iterable[str] | None, float]],
    *,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[CorruptionLabel]]:
    """Apply corruptors in sequence without double-corrupting any cell."""
    out = df
    all_labels: list[CorruptionLabel] = []
    taken: set[tuple[int, str]] = set()
    for i, (corruptor, columns, share) in enumerate(steps):
        out, labels = apply_corruptor(
            out, corruptor, columns, seed=seed + i * 1009, share=share, skip_cells=taken,
        )
        for label in labels:
            if label.row is not None and label.column is not None:
                taken.add((label.row, label.column))
        all_labels.extend(labels)
    return out, all_labels
