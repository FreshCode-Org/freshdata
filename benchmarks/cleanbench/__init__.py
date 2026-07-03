"""CleanBench: deterministic corruption/repair benchmark skeleton (Phase 2).

CleanBench measures what an auto-cleaner must *never* get wrong before it
measures anything else:

- **protected-column violation rate** — must be exactly ``0.0``;
- **false modification rate** — cells that were already correct and got
  changed anyway — release gate ``<= 0.1%`` on the Phase-2 fixtures;
- cell repair precision / recall / F1 on deliberately corrupted cells.

Everything is deterministic: corruptors take an explicit ``seed`` and the
fixtures are built in code (no binary blobs), so CI failures reproduce
locally byte-for-byte. This is the *skeleton* — tiers T2 (semantic values)
and T3 (context compliance) ship here; the full suite grows in later phases.
"""

from .corruptors import (
    CORRUPTORS,
    allowed_value_casing,
    allowed_value_punctuation,
    casing_corruption,
    date_phrase_corruption,
    duplicate_row_injection,
    email_at_spacing,
    email_double_at,
    phone_format_corruption,
    sentinel_injection,
    whitespace_corruption,
)
from .fixtures import (
    ECOMMERCE_CONTEXT,
    make_t2_semantic_fixture,
    make_t3_context_fixture,
)
from .metrics import (
    GATE_FALSE_MODIFICATION_RATE,
    GATE_PROTECTED_VIOLATION_RATE,
    GATE_RUNTIME_SLOWDOWN,
    cell_repair_f1,
    cell_repair_precision,
    cell_repair_recall,
    false_modification_rate,
    protected_column_violation_rate,
)

__all__ = [
    "CORRUPTORS",
    "ECOMMERCE_CONTEXT",
    "GATE_FALSE_MODIFICATION_RATE",
    "GATE_PROTECTED_VIOLATION_RATE",
    "GATE_RUNTIME_SLOWDOWN",
    "allowed_value_casing",
    "allowed_value_punctuation",
    "casing_corruption",
    "cell_repair_f1",
    "cell_repair_precision",
    "cell_repair_recall",
    "date_phrase_corruption",
    "duplicate_row_injection",
    "email_at_spacing",
    "email_double_at",
    "false_modification_rate",
    "make_t2_semantic_fixture",
    "make_t3_context_fixture",
    "phone_format_corruption",
    "protected_column_violation_rate",
    "sentinel_injection",
    "whitespace_corruption",
]
