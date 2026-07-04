"""The deterministic expert backend — always first in trust order.

A thin named wrapper over the Phase-2 expert pipeline
(:func:`freshdata.semantic.profiler.profile_proposals`): routing, experts, and
proposals are byte-for-byte what they were before backends were formalized.
Unbudgeted by design — deterministic proposals must survive any model budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..profiler import profile_proposals
from .base import Budget

if TYPE_CHECKING:
    import pandas as pd

    from ..types import SemanticContext, SemanticProposal


class DeterministicBackend:
    name = "deterministic"

    def warm_up(self) -> None:
        """Nothing to load; deterministic experts are pure Python."""

    def propose(
        self, df: pd.DataFrame, ctx: SemanticContext, budget: Budget
    ) -> list[SemanticProposal]:
        del budget  # deterministic experts are never metered
        return list(profile_proposals(df, ctx))
