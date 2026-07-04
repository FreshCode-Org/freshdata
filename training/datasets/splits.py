"""Deterministic dataset splits (hash-based, author-disjoint capable)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from ..common import sha256_text


def split_key(item: dict[str, Any], fields: Sequence[str]) -> str:
    return "|".join(str(item.get(f, "")) for f in fields)


def hash_split(
    items: Sequence[dict[str, Any]],
    *,
    key_fn: Callable[[dict[str, Any]], str],
    eval_fraction: float = 0.2,
    salt: str = "freshdata-split",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into (train, eval) by hashing a stable per-item key.

    Items sharing a key always land on the same side, so near-duplicates
    (paraphrases of one sentence, variants of one column) never straddle the
    boundary.
    """
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    threshold = int(eval_fraction * 0xFFFF)
    for item in items:
        bucket = int(sha256_text(f"{salt}:{key_fn(item)}")[:4], 16)
        (holdout if bucket < threshold else train).append(item)
    return train, holdout


def author_disjoint_split(
    items: Sequence[dict[str, Any]],
    *,
    eval_authors: Sequence[str],
    author_field: str = "author",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split so that eval authors never appear in training (paraphrase gate)."""
    eval_set = set(eval_authors)
    train = [i for i in items if str(i.get(author_field, "")) not in eval_set]
    holdout = [i for i in items if str(i.get(author_field, "")) in eval_set]
    return train, holdout
