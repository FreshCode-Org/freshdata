# src/freshdata/__init__.py
from importlib.metadata import version as _version

from freshdata._core.engine import Cleaner, apply_plan, clean, profile, suggest_plan
from freshdata._core.models import (
    ActionRecord,
    CleanConfig,
    CleanPlan,
    CleanReport,
    ColumnProfile,
    ColumnRule,
    ProfileReport,
)

__version__ = _version("freshdata-cleaner")

__all__ = [
    "ActionRecord",
    "CleanConfig",
    "Cleaner",
    "CleanPlan",
    "CleanReport",
    "ColumnProfile",
    "ColumnRule",
    "ProfileReport",
    "apply_plan",
    "clean",
    "profile",
    "suggest_plan",
]
