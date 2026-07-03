"""The optional embedding rescue rung of the column resolver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.context.compiler import resolve_policy
from freshdata.context.resolve import resolve_reference
from freshdata.models import runtime as model_runtime

SCHEMA = ["cust_id", "email_addr", "mob_no", "monthly_revenue"]


def _scorer(table: dict[str, list[tuple[str, float]]]):
    def scorer(ref: str, columns):
        ranked = table.get(ref, [])
        by_name = dict(ranked)
        return [(c, by_name.get(c, 0.0)) for c in columns]

    return scorer


def test_no_scorer_is_byte_identical():
    plain = resolve_reference("Phone numbers", SCHEMA)
    with_kwargs = resolve_reference(
        "Phone numbers", SCHEMA, embedding_scorer=None, embedding_model_id=None
    )
    assert plain == with_kwargs
    assert plain.method == "unresolved"
    assert plain.model_id is None


def test_embedding_rescues_unresolved_alias():
    scorer = _scorer({"Phone numbers": [("mob_no", 0.83), ("email_addr", 0.30)]})
    resolution = resolve_reference(
        "Phone numbers", SCHEMA, embedding_scorer=scorer, embedding_model_id="fd-col-encoder-v1"
    )
    assert resolution.resolved
    assert resolution.column == "mob_no"
    assert resolution.method == "embedding"
    assert resolution.confidence == pytest.approx(0.83)
    assert resolution.model_id == "fd-col-encoder-v1"
    # Evidence: ranked candidates including the runner-up.
    assert resolution.candidates[0] == ("mob_no", 0.83)
    assert ("email_addr", 0.30) in resolution.candidates


def test_embedding_never_overrides_exact_or_alias():
    scorer = _scorer(
        {
            "cust_id": [("monthly_revenue", 0.99)],
            "CustomerID": [("monthly_revenue", 0.99)],
        }
    )
    exact = resolve_reference("cust_id", SCHEMA, embedding_scorer=scorer)
    assert (exact.column, exact.method) == ("cust_id", "exact")
    alias = resolve_reference("CustomerID", SCHEMA, embedding_scorer=scorer)
    assert (alias.column, alias.method) == ("cust_id", "alias")


def test_embedding_close_candidates_stay_unresolved():
    scorer = _scorer({"Phone numbers": [("mob_no", 0.82), ("email_addr", 0.80)]})
    resolution = resolve_reference(
        "Phone numbers", SCHEMA, embedding_scorer=scorer, embedding_model_id="fd-col-encoder-v1"
    )
    assert not resolution.resolved
    assert "embedding also ambiguous" in resolution.reason
    assert resolution.candidates[0][0] == "mob_no"


def test_embedding_below_threshold_stays_unresolved():
    scorer = _scorer({"Phone numbers": [("mob_no", 0.45), ("email_addr", 0.10)]})
    resolution = resolve_reference("Phone numbers", SCHEMA, embedding_scorer=scorer)
    assert not resolution.resolved
    assert "no column matches" in resolution.reason


def test_strict_mode_still_raises_on_unresolved():
    scorer = _scorer({})  # scores everything 0.0
    df = pd.DataFrame({c: [1] for c in SCHEMA})
    with pytest.raises(fd.PolicyError, match="Fax"):
        fd.compile_context("Fax numbers must be valid.", df=df, strict=True)
    del scorer  # strict path exercised through the public API without a model


class _NameEncoder:
    """Maps snake-spaced names to fixed axes: 'phone number' ~ 'mob no'."""

    model_id = "fd-col-encoder-v1"
    model_sha256 = "resolver-test"
    dim = 16

    _AXES = {
        "phone numbers": 0,
        "phone number": 0,
        "mob no": None,  # blended below
        "cust id": 2,
        "email addr": 3,
        "monthly revenue": 4,
    }

    def encode_texts(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if text in ("mob no",):
                vec = np.zeros(self.dim, dtype=np.float32)
                vec[0], vec[9] = 0.9, np.sqrt(1 - 0.81)
                out[i] = vec
            else:
                axis = self._AXES.get(text)
                vec = np.zeros(self.dim, dtype=np.float32)
                vec[axis if axis is not None else 10 + (i % 5)] = 1.0
                out[i] = vec
        return out


def test_compile_context_uses_embedding_rung_end_to_end():
    model_runtime.set_encoder_factory(lambda model_id: _NameEncoder())
    try:
        df = pd.DataFrame(
            {
                "cust_id": ["C1"],
                "email_addr": ["a@b.co"],
                "mob_no": ["98765 43210"],
                "monthly_revenue": ["100"],
            }
        )
        config = fd.CleanConfig(
            semantic_mode="auto", semantic_backends=("deterministic", "embedding")
        )
        policy = fd.compile_context("Phone numbers are Indian.", df=df, config=config)
        phone = [c for c in policy.constraints if c.rule == "locale_format"]
        assert len(phone) == 1
        assert phone[0].column == "mob_no"
        assert phone[0].resolution_confidence == pytest.approx(0.9, abs=1e-4)
        evidence = phone[0].params["resolution_evidence"]
        assert evidence["method"] == "embedding"
        assert evidence["model_id"] == "fd-col-encoder-v1"
        assert evidence["candidates"][0][0] == "mob_no"

        # Without the embedding backend in the tuple, the same compile stays
        # deterministic and the reference remains unresolved.
        plain_policy = fd.compile_context("Phone numbers are Indian.", df=df)
        assert not [c for c in plain_policy.constraints if c.rule == "locale_format"]
        assert plain_policy.unresolved
    finally:
        model_runtime.set_encoder_factory(None)


def test_resolution_survives_policy_roundtrip():
    model_runtime.set_encoder_factory(lambda model_id: _NameEncoder())
    try:
        config = fd.CleanConfig(
            semantic_mode="auto", semantic_backends=("deterministic", "embedding")
        )
        # Schema-free compile defers resolution; resolve_policy picks up the rung.
        policy = fd.compile_context("Phone numbers are Indian.", config=config)
        resolved = resolve_policy(
            policy, ["cust_id", "email_addr", "mob_no", "monthly_revenue"], config=config
        )
        phone = [c for c in resolved.constraints if c.rule == "locale_format"]
        assert phone and phone[0].column == "mob_no"
        assert phone[0].params["resolution_evidence"]["method"] == "embedding"
    finally:
        model_runtime.set_encoder_factory(None)
