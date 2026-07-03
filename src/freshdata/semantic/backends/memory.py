"""The memory-replay backend — learned repairs as ordinary proposals.

Formalizes the Phase-2 replay path
(:func:`freshdata.semantic.memory.semantic_memory_proposals`) as a named
backend. It runs whenever a :class:`~freshdata.CleaningMemory` is supplied,
exactly as before backends were formalized — listing ``"memory"`` in
``semantic_backends`` does not conjure a memory out of nowhere, and omitting it
does not silently drop replay for existing memory users.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..memory import semantic_memory_proposals
from .base import BackendUnavailable, Budget

if TYPE_CHECKING:
    import pandas as pd

    from ..types import SemanticContext, SemanticProposal


class MemoryBackend:
    name = "memory"

    def __init__(self, memory: object | None) -> None:
        self._memory = memory

    @property
    def memory(self) -> object | None:
        return self._memory

    def warm_up(self) -> None:
        if self._memory is None:
            raise BackendUnavailable(
                "no cleaning memory supplied; pass memory= (or use Cleaner with memory) "
                "to enable replay proposals"
            )

    def propose(
        self, df: pd.DataFrame, ctx: SemanticContext, budget: Budget
    ) -> list[SemanticProposal]:
        del budget  # replay is local and cheap; never metered
        return list(semantic_memory_proposals(df, ctx, self._memory))
