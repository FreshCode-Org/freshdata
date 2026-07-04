"""Paired-data builders for profile learning (T4) and vendor-map corruption."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from .base import CorruptionLabel, Corruptor, compose
from .representation import casing_change, sentinel_injection, whitespace_insertion
from .semantic_values import allowed_value_case, phone_in_zero_prefix


def _vendor_code_swap(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    """Replace a clean label with a vendor-specific code (learnable literal map)."""
    codes = params.get("vendor_map", {
        "active": "A1", "inactive": "I0", "pending": "P2", "churned": "X9",
    })
    return codes.get(v.strip().lower())


vendor_code_swap = Corruptor(
    name="vendor_code_swap", family="reference_value", fn=_vendor_code_swap,
    risk="medium", should_auto_apply=False)


def make_paired_learning_set(
    clean: pd.DataFrame,
    *,
    seed: int = 0,
    value_columns: tuple[str, ...] = ("status",),
    phone_columns: tuple[str, ...] = (),
    text_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, list[CorruptionLabel]]:
    """Build a ``(messy, clean, labels)`` pair for ``fd.learn``.

    The corruptions chosen here are exactly the ones a learning profile can
    capture: a stable vendor literal map plus representation dirt.
    """
    steps: list[tuple[Corruptor, tuple[str, ...] | None, float]] = [
        (vendor_code_swap, value_columns, 0.9),
        (allowed_value_case, value_columns, 0.3),
        (whitespace_insertion, text_columns or value_columns, 0.3),
        (casing_change, text_columns, 0.3),
        (phone_in_zero_prefix, phone_columns, 0.6),
        (sentinel_injection, text_columns, 0.05),
    ]
    steps = [(c, cols, share) for c, cols, share in steps if cols]
    messy, labels = compose(clean, steps, seed=seed)
    return messy, clean.copy(deep=True), labels
