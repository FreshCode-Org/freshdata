"""The adversarial trap corpus shared by tests and benchmark harnesses.

``pythonpath = ["."]`` in ``pyproject.toml`` already makes ``benchmarks.*``
importable from ``tests/``, so any test can do::

    from benchmarks.corpus import TRAPS, Disposition, by_token

See :mod:`benchmarks.corpus.traps` for the corpus itself and
:mod:`benchmarks.corpus.dispositions` for the vocabulary it is scored against.
"""

from .adapters import all_cases, from_gauntlet, from_truthbench
from .dispositions import (
    MUTATING,
    REVIEW_FAMILY,
    Disposition,
    from_field_action,
    satisfies,
)
from .traps import (
    TRAPS,
    UNSET,
    TrapCase,
    by_family,
    by_role,
    by_token,
    families,
    roles,
)

__all__ = [
    "Disposition",
    "REVIEW_FAMILY",
    "MUTATING",
    "from_field_action",
    "satisfies",
    "TrapCase",
    "TRAPS",
    "UNSET",
    "by_family",
    "by_role",
    "by_token",
    "families",
    "roles",
    "all_cases",
    "from_gauntlet",
    "from_truthbench",
]
