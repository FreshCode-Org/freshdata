"""Learned-profile proposal backend (Phase 4).

Turns a :class:`~freshdata.learning.LearningProfile` into
:class:`~freshdata.semantic.types.SemanticProposal` candidates:

* **value maps** — literal ``raw -> clean`` repairs learned from a paired
  dataset, proposed with their learned precision as base confidence;
* **example retrieval** — when the ``[semantic]`` extra and a local encoder
  are available, unexplained training examples are retrieved by similarity
  as *suggest-only* evidence (never auto-applied).

Like every backend this one only generates candidates: the policy gate in
:func:`freshdata.semantic.policy.decide` judges each proposal against the
current context policy and calibrated thresholds, and the executor's
byte-identity guard keeps hard-protected columns physically untouched no
matter what the profile learned.  Masked entries never replay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..scoring import make_proposal
from ..types import SemanticEvidence, SemanticProposal
from .base import Budget

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from ..types import SemanticContext

__all__ = ["ProfileBackend"]

#: Learned precision is honest evidence, not certainty: cap the base
#: confidence below 1.0 so calibration keeps head-room to demote.
_MAX_BASE_CONFIDENCE = 0.98

#: Suggest-only ceiling for example-retrieval proposals.
_RETRIEVAL_CONFIDENCE = 0.60

#: Retrieval is skipped for columns whose distinct-value count exceeds this
#: (high-cardinality free text is where retrieval hallucinates).
_RETRIEVAL_MAX_DISTINCT = 200

_RETRIEVAL_SIMILARITY = 0.90


class ProfileBackend:
    """Propose repairs learned by ``fd.learn`` for the current frame."""

    name = "profile"

    def __init__(self, profile: Any) -> None:
        self._profile = profile

    def warm_up(self) -> None:  # pragma: no cover - nothing to warm
        return None

    # -- proposal generation -------------------------------------------------

    def propose(
        self, df: pd.DataFrame, ctx: SemanticContext, budget: Budget
    ) -> list[SemanticProposal]:
        proposals = self._value_map_proposals(df, ctx, budget)
        proposals.extend(self._retrieval_proposals(df, ctx, budget))
        return proposals

    def _base_provenance(self) -> dict[str, object]:
        return {
            "profile_influenced": True,
            "profile_id": self._profile.profile_id,
        }

    def _value_map_proposals(
        self, df: pd.DataFrame, ctx: SemanticContext, budget: Budget
    ) -> list[SemanticProposal]:
        out: list[SemanticProposal] = []
        for column, value_map in sorted(self._profile.value_maps.items()):
            if column not in df.columns:
                continue
            info = ctx.columns.get(column)
            # mutable is tri-state: None means "unspecified" (replayable);
            # only an explicit mutable=False (protected) blocks replay.
            if info is not None and getattr(info, "mutable", None) is False:
                continue
            lookup = value_map.lookup()
            if not lookup:
                continue
            counts = df[column].value_counts(dropna=True)
            for raw_value, entry in lookup.items():
                count = int(counts.get(raw_value, 0))
                if count == 0 or budget.exhausted:
                    continue
                provenance = self._base_provenance()
                provenance.update(
                    {
                        "support": entry.support,
                        "learned_precision": entry.precision,
                        "transform_family": entry.transform_family,
                    }
                )
                out.append(
                    make_proposal(
                        column=column,
                        raw_value=raw_value,
                        proposed_value=entry.clean_value,
                        issue_type=entry.transform_family,
                        expert=f"profile:{entry.transform_family}",
                        base_confidence=min(entry.precision, _MAX_BASE_CONFIDENCE),
                        evidence=(
                            SemanticEvidence(
                                "profile_replay",
                                f"learned from {entry.support} training example(s) "
                                f"in profile {self._profile.profile_id} "
                                f"(precision {entry.precision:.3f})",
                                0.0,
                            ),
                        ),
                        count=count,
                        rationale=(
                            f"Replayed learned repair from profile {self._profile.profile_id}"
                        ),
                        info=info,
                        backend="profile",
                        provenance=provenance,
                    )
                )
        return out

    # -- example retrieval (optional, suggest-only) --------------------------

    def _retrieval_proposals(
        self, df: pd.DataFrame, ctx: SemanticContext, budget: Budget
    ) -> list[SemanticProposal]:
        bank = self._profile.examples
        vectors = self._profile.vectors
        if bank is None or vectors is None or not bank.vectors_path:
            return []
        candidates = [e for e in bank.examples if not e.masked and isinstance(e.raw_value, str)]
        if not candidates or len(candidates) != len(vectors):
            return []
        try:
            from ...models.runtime import get_encoder  # noqa: PLC0415 - optional path

            encoder = get_encoder()
        except Exception:  # noqa: BLE001 - retrieval is strictly optional
            return []
        if encoder is None:
            return []

        import numpy as np  # noqa: PLC0415

        bank_matrix = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(bank_matrix, axis=1)
        norms[norms == 0] = 1.0

        out: list[SemanticProposal] = []
        example_columns = {e.column for e in candidates}
        for column in sorted(example_columns):
            if column not in df.columns or budget.exhausted:
                continue
            info = ctx.columns.get(column)
            # mutable is tri-state: None means "unspecified" (replayable);
            # only an explicit mutable=False (protected) blocks replay.
            if info is not None and getattr(info, "mutable", None) is False:
                continue
            series = df[column].dropna()
            distinct = series.unique()
            if len(distinct) > _RETRIEVAL_MAX_DISTINCT:
                continue  # high-cardinality free text: retrieval off by default
            column_ids = [i for i, e in enumerate(candidates) if e.column == column]
            if not column_ids:
                continue
            texts = [str(v) for v in distinct if isinstance(v, str)]
            if not texts:
                continue
            try:
                queries = np.asarray(encoder.encode_texts(texts), dtype="float32")
            except Exception:  # noqa: BLE001
                return out
            counts = series.value_counts()
            sub = bank_matrix[column_ids]
            sub_norms = norms[column_ids]
            for text, query in zip(texts, queries):
                q_norm = float(np.linalg.norm(query)) or 1.0
                sims = (sub @ query) / (sub_norms * q_norm)
                best = int(np.argmax(sims))
                if float(sims[best]) < _RETRIEVAL_SIMILARITY:
                    continue
                example = candidates[column_ids[best]]
                if str(example.raw_value) == text or example.clean_value is None:
                    continue
                provenance = self._base_provenance()
                provenance.update(
                    {
                        "support": example.support,
                        "learned_precision": None,
                        "transform_family": example.transform_family,
                        "retrieval_similarity": round(float(sims[best]), 4),
                    }
                )
                out.append(
                    make_proposal(
                        column=column,
                        raw_value=text,
                        proposed_value=example.clean_value,
                        issue_type=example.transform_family,
                        expert="profile:example_retrieval",
                        base_confidence=_RETRIEVAL_CONFIDENCE,
                        evidence=(
                            SemanticEvidence(
                                "profile_example",
                                f"similar to training example "
                                f"{example.raw_value!r} -> {example.clean_value!r} "
                                f"(similarity {float(sims[best]):.2f})",
                                0.0,
                            ),
                        ),
                        count=int(counts.get(text, 1)),
                        rationale=(
                            "Similar unexplained example in learned profile "
                            f"{self._profile.profile_id}; suggest-only"
                        ),
                        info=info,
                        backend="profile",
                        provenance=provenance,
                    )
                )
        return out
