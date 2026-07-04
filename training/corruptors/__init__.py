"""Parameterized corruption engine mirroring FreshData's repair taxonomy.

Ground truth for training comes from *corruption metadata*: every corruptor
records exactly what it changed, what the clean value was, and whether the
change is safely repairable — teacher models never invent messy/clean pairs.
"""

from .base import CorruptionLabel, Corruptor, apply_corruptor, compose
from .registry import CORRUPTOR_REGISTRY, FAMILIES, get_corruptor

__all__ = [
    "CORRUPTOR_REGISTRY",
    "FAMILIES",
    "CorruptionLabel",
    "Corruptor",
    "apply_corruptor",
    "compose",
    "get_corruptor",
]
