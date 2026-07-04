"""Registry of every corruptor, keyed by name, with family groupings."""

from __future__ import annotations

from .base import Corruptor
from .context import CONTEXT_CORRUPTORS
from .learning import vendor_code_swap
from .representation import REPRESENTATION_CORRUPTORS
from .semantic_values import SEMANTIC_VALUE_CORRUPTORS

_ALL: tuple[Corruptor, ...] = (
    *REPRESENTATION_CORRUPTORS,
    *SEMANTIC_VALUE_CORRUPTORS,
    *CONTEXT_CORRUPTORS,
    vendor_code_swap,
)

CORRUPTOR_REGISTRY: dict[str, Corruptor] = {c.name: c for c in _ALL}

FAMILIES: dict[str, tuple[str, ...]] = {}
for _c in _ALL:
    FAMILIES.setdefault(_c.family, ())
    FAMILIES[_c.family] = (*FAMILIES[_c.family], _c.name)

if len(CORRUPTOR_REGISTRY) != len(_ALL):  # pragma: no cover - definition bug
    raise RuntimeError("duplicate corruptor names in registry")


def get_corruptor(name: str) -> Corruptor:
    try:
        return CORRUPTOR_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(CORRUPTOR_REGISTRY))
        raise KeyError(f"unknown corruptor {name!r}; known: {known}") from None
