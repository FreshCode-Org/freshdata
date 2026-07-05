"""Example FreshData plugin: a custom semantic *backend*.

A backend proposes over the *whole frame* (not one column at a time) and is
opted into by name via ``semantic_backends=(...)``. This one strips a shared
``"lbl_"`` prefix from any string column — a frame-wide transform an
expert (which sees one column) is a poor fit for.

Try it::

    import freshdata as fd
    from keyword_backend import KeywordBackend

    fd.testing.semantic_backend_contract(KeywordBackend())
    fd.register_backend(KeywordBackend())

    import pandas as pd
    df = pd.DataFrame({"tag": ["lbl_red", "lbl_blue", "plain"]})
    cleaned, report = fd.clean(
        df, semantic_mode="auto",
        semantic_backends=("deterministic", "keyword"),
        return_report=True,
    )

Package it::

    [project.entry-points."freshdata.backends"]
    keyword = "keyword_backend:KeywordBackend"
"""

from __future__ import annotations

import pandas as pd

from freshdata.semantic.backends.base import Budget
from freshdata.semantic.scoring import make_proposal
from freshdata.semantic.types import SemanticContext, SemanticEvidence, SemanticProposal

_PREFIX = "lbl_"


class KeywordBackend:
    name = "keyword"
    semantic_types = ("categorical",)
    max_risk = "medium"
    uses_network = False
    requires = ()

    def warm_up(self) -> None:
        # Nothing to load. Raise freshdata.semantic.backends.base.BackendUnavailable
        # here to self-disable when an optional resource is missing.
        pass

    def propose(
        self, df: pd.DataFrame, ctx: SemanticContext, budget: Budget
    ) -> list[SemanticProposal]:
        proposals: list[SemanticProposal] = []
        for col in df.columns:
            info = ctx.columns.get(str(col))
            if info is None or info.identifier_like:
                continue  # never touch id-like columns
            distinct = df[col].dropna().unique()
            # Respect the model budget the same way built-in backends do.
            if not budget.try_column(len(distinct)):
                break
            for raw in distinct:
                if isinstance(raw, str) and raw.startswith(_PREFIX):
                    proposals.append(make_proposal(
                        column=str(col),
                        raw_value=raw,
                        proposed_value=raw[len(_PREFIX):],
                        issue_type="category_synonym",
                        expert=self.name,
                        base_confidence=0.95,
                        evidence=(SemanticEvidence("pattern", f"strip {_PREFIX!r}", 0.0),),
                        count=int((df[col] == raw).sum()),
                        rationale=f"stripped {_PREFIX!r} prefix from {raw!r}",
                        info=info,
                    ))
        return proposals
