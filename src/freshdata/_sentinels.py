"""Registry of string values that conventionally mean "missing".

All entries are stored casefolded; matching is case-insensitive and happens
after whitespace stripping, so ``" N/A "`` and ``"n/a"`` both match.
"""

from __future__ import annotations

#: Values commonly used in CSV / Excel / SQL exports to denote a missing cell.
#: Deliberately conservative: entries here are near-certain to mean "missing"
#: when they appear as the entire cell value. Domain words that merely *might*
#: mean missing (e.g. ``"unknown"``) are excluded; pass them via the
#: ``extra_sentinels`` option instead.
DEFAULT_SENTINELS: frozenset[str] = frozenset(
    {
        # empty / placeholder punctuation
        "",
        "-",
        "--",
        "---",
        "?",
        "??",
        # spelled-out missing markers
        "na",
        "n/a",
        "n\\a",
        "\\n",
        "n.a",
        "n.a.",
        "nan",
        "null",
        "none",
        "nil",
        "missing",
        "(null)",
        "(none)",
        "(blank)",
        "(empty)",
        "(missing)",
        # Excel error codes — never legitimate data
        "#n/a",
        "#n/a n/a",
        "#na",
        "#null!",
        "#div/0!",
        "#ref!",
        "#value!",
        "#name?",
        "#num!",
    }
)
