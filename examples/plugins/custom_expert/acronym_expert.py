"""Example FreshData plugin: a custom semantic *expert*.

An expert proposes value repairs for a single column, one distinct value at a
time. This one expands a small, explicit acronym table (``"NY" -> "New York"``)
in categorical columns. It is deterministic, offline, and low-risk.

Try it::

    import freshdata as fd
    from acronym_expert import AcronymExpert

    fd.testing.expert_contract(AcronymExpert())   # verify the contract first
    fd.register_expert(AcronymExpert())

    import pandas as pd
    df = pd.DataFrame({"state": ["NY", "CA", "NY", "TX"]})
    cleaned, report = fd.clean(df, semantic_mode="auto", return_report=True)

The plugin only *proposes* — the policy gate decides what applies. Because
``category_synonym`` is a medium-risk issue type, these expansions are recorded
as ``status="suggested"`` in the report (not auto-applied) unless you raise the
confidence or run a lower-risk issue type. That is the gate doing its job: a
plugin can never force an auto-apply.

Package it for entry-point discovery in your pyproject.toml::

    [project.entry-points."freshdata.experts"]
    acronym = "acronym_expert:AcronymExpert"
"""

from __future__ import annotations

import pandas as pd

from freshdata.semantic.scoring import make_proposal
from freshdata.semantic.types import SemanticColumnInfo, SemanticEvidence, SemanticProposal

_ACRONYMS = {
    "ny": "New York",
    "ca": "California",
    "tx": "Texas",
    "wa": "Washington",
}


class AcronymExpert:
    # --- required metadata -------------------------------------------------
    name = "acronym"
    issue_type = "category_synonym"  # reuses the built-in scoring/risk profile
    #: The semantic column kinds this expert claims to handle (declaration only;
    #: routing still calls applies()).
    semantic_types = ("categorical",)
    #: The highest risk this expert may ever emit; the plugin system drops any
    #: proposal scored above it before the gate sees it.
    max_risk = "medium"
    #: Deterministic + offline: no network, no optional dependencies.
    uses_network = False
    requires = ()

    def applies(self, info: SemanticColumnInfo) -> bool:
        # Only short, low-cardinality categorical columns — never an id/target.
        return (
            info.role == "categorical"
            and not info.identifier_like
            and not info.free_text
        )

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        # Operate on *distinct values* (never row-by-row) and never mutate the
        # series — return proposals; the gate decides what actually applies.
        proposals: list[SemanticProposal] = []
        for raw, count in series.value_counts(dropna=True).items():
            if not isinstance(raw, str):
                continue
            expansion = _ACRONYMS.get(raw.strip().lower())
            if expansion is None or expansion == raw:
                continue
            proposals.append(make_proposal(
                column=info.name,
                raw_value=raw,
                proposed_value=expansion,
                issue_type=self.issue_type,
                expert=self.name,
                base_confidence=0.9,
                evidence=(SemanticEvidence("lookup", f"acronym table -> {expansion!r}", 0.0),),
                count=int(count),
                rationale=f"expanded acronym {raw!r} to {expansion!r}",
                info=info,
            ))
        return proposals
