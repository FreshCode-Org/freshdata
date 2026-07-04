"""Synthetic PII-shaped data generators (no real people, no real PII)."""

from .generators import (
    SYNTHETIC_SOURCE_ID,
    make_context_sentences,
    make_customers,
    make_transactions,
    seed_tables,
)

__all__ = [
    "SYNTHETIC_SOURCE_ID",
    "make_context_sentences",
    "make_customers",
    "make_transactions",
    "seed_tables",
]
