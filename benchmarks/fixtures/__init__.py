"""Enterprise fixture library for the FreshData benchmark harness.

Each ``generate(n_rows, seed=42, defect_rate=None) -> pd.DataFrame`` generator
is deterministic and ships a ``GOLD_LABELS`` dict and a ``DEFECT_MANIFEST``
list. The gold fixture additionally exposes a :class:`~fixtures.gold.GoldBundle`
with dirty/clean frames and preservation/repair/false-repair masks.

Use :data:`REGISTRY` to look a fixture module up by name.
"""

from __future__ import annotations

from types import ModuleType

from . import crm, event_log, finance, gold, provenance, wide_schema

#: name -> fixture module. ``gold`` is intentionally included; its ``generate``
#: returns a :class:`GoldBundle` rather than a bare DataFrame, so callers that
#: only want a frame should use ``.dirty_df``.
REGISTRY: dict[str, ModuleType] = {
    "crm": crm,
    "finance": finance,
    "event_log": event_log,
    "wide_schema": wide_schema,
    "provenance": provenance,
    "gold": gold,
}

#: Fixtures whose ``generate`` returns a plain DataFrame (everything but gold).
FRAME_FIXTURES = ("crm", "finance", "event_log", "wide_schema", "provenance")

__all__ = [
    "REGISTRY",
    "FRAME_FIXTURES",
    "crm",
    "finance",
    "event_log",
    "wide_schema",
    "provenance",
    "gold",
]
