"""Contract tests for FreshData plugins (``fd.testing``).

Call one of these on your plugin to assert it satisfies the contract *before*
you register it. They are plain functions that raise ``AssertionError`` with an
actionable message on the first violation and return ``None`` on success, so
they work inside pytest or as a standalone smoke check::

    import freshdata as fd
    fd.testing.expert_contract(MyExpert())
    fd.testing.semantic_backend_contract(MyBackend())
    fd.testing.validator_contract(MyValidator())

What they verify (the guarantees ``freshdata.plugins`` relies on):

- the required protocol methods exist and return the right *types*;
- metadata is well-formed (``max_risk`` in {low, medium, high}, ``uses_network``
  a bool, ``requires`` / ``semantic_types`` iterable);
- the plugin does **not mutate** the input it is handed;
- experts/backends only ever emit proposals for the column/frame they were
  given, and never above their declared ``max_risk``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .findings import QualityFinding
from .plugins import _RISK_RANK
from .semantic.types import SemanticColumnInfo, SemanticProposal

__all__ = ["expert_contract", "semantic_backend_contract", "validator_contract"]


def _check_metadata(plugin: Any, *, want_semantic_types: bool) -> None:
    max_risk = getattr(plugin, "max_risk", "high")
    assert max_risk in _RISK_RANK, (
        f"max_risk must be one of {sorted(_RISK_RANK)}, got {max_risk!r}"
    )
    uses_network = getattr(plugin, "uses_network", False)
    assert isinstance(uses_network, bool), "uses_network must be a bool"
    requires = getattr(plugin, "requires", ())
    assert isinstance(requires, (list, tuple, str)), "requires must be a str or sequence of str"
    if want_semantic_types:
        semantic_types = getattr(plugin, "semantic_types", ())
        assert isinstance(semantic_types, (list, tuple, str)), (
            "semantic_types must be a str or sequence of str"
        )


def _synthetic_info(name: str = "value") -> SemanticColumnInfo:
    """A deliberately permissive column info so most experts' ``applies`` fire."""
    return SemanticColumnInfo(
        name=name, role="categorical", n_nonnull=6, nunique=4, high_cardinality=False,
        preserve=False, free_text=False, numeric_like=True, boolean_like=True,
        money_like=True, unit_like=True, identifier_like=False, date_like=True,
        email_like=True, phone_like=True,
    )


def expert_contract(expert: Any) -> None:
    """Assert *expert* satisfies the semantic-expert plugin contract."""
    assert isinstance(getattr(expert, "name", None), str) and expert.name, (
        "expert must expose a non-empty string `name`"
    )
    assert isinstance(getattr(expert, "issue_type", None), str) and expert.issue_type, (
        "expert must expose a non-empty string `issue_type`"
    )
    assert callable(getattr(expert, "applies", None)), "expert must define applies(info)"
    assert callable(getattr(expert, "propose", None)), "expert must define propose(series, info)"
    _check_metadata(expert, want_semantic_types=True)

    info = _synthetic_info()
    applies = expert.applies(info)
    assert isinstance(applies, bool), "applies(info) must return a bool"

    series = pd.Series(["one", "two", "3", "yes", "$4", "10 kg"], name=info.name)
    before = series.copy(deep=True)
    proposals = expert.propose(series, info)
    assert isinstance(proposals, list), "propose(...) must return a list"
    pd.testing.assert_series_equal(series, before, obj="propose() must not mutate the series")

    max_rank = _RISK_RANK[getattr(expert, "max_risk", "high")]
    for p in proposals:
        assert isinstance(p, SemanticProposal), (
            "every proposal must be a freshdata.semantic.SemanticProposal "
            "(build it with freshdata.semantic.scoring.make_proposal)"
        )
        assert p.column == info.name, (
            f"expert proposed for column {p.column!r} but was given {info.name!r}"
        )
        assert _RISK_RANK.get(p.risk, 2) <= max_rank, (
            f"proposal risk {p.risk!r} exceeds declared max_risk "
            f"{getattr(expert, 'max_risk', 'high')!r}"
        )


def semantic_backend_contract(backend: Any) -> None:
    """Assert *backend* satisfies the semantic-backend plugin contract."""
    from .semantic.backends.base import Budget  # noqa: PLC0415
    from .semantic.context import build_semantic_context  # noqa: PLC0415

    assert isinstance(getattr(backend, "name", None), str) and backend.name, (
        "backend must expose a non-empty string `name`"
    )
    assert callable(getattr(backend, "propose", None)), (
        "backend must define propose(df, ctx, budget)"
    )
    _check_metadata(backend, want_semantic_types=True)

    warm = getattr(backend, "warm_up", None)
    if warm is not None:
        assert callable(warm), "warm_up must be callable when defined"

    from .config import CleanConfig  # noqa: PLC0415

    df = pd.DataFrame({"value": ["one", "two", "3", "yes", "$4", "10 kg"]})
    before = df.copy(deep=True)
    ctx = build_semantic_context(df, CleanConfig(semantic_mode="assist"))
    if warm is not None:
        try:
            warm()
        except Exception as exc:  # noqa: BLE0001 - surfaced as a contract failure
            from .semantic.backends.base import BackendUnavailable  # noqa: PLC0415

            assert isinstance(exc, BackendUnavailable), (
                f"warm_up must only raise BackendUnavailable, got {type(exc).__name__}: {exc}"
            )
            return  # a self-disabling backend is contract-valid; nothing more to check
    proposals = backend.propose(df, ctx, Budget())
    assert isinstance(proposals, list), "propose(...) must return a list"
    pd.testing.assert_frame_equal(df, before, obj="propose() must not mutate the frame")

    max_rank = _RISK_RANK[getattr(backend, "max_risk", "high")]
    for p in proposals:
        assert isinstance(p, SemanticProposal), "every proposal must be a SemanticProposal"
        assert p.column in df.columns, f"backend proposed for unknown column {p.column!r}"
        assert _RISK_RANK.get(p.risk, 2) <= max_rank, (
            f"proposal risk {p.risk!r} exceeds declared max_risk"
        )


def validator_contract(validator: Any) -> None:
    """Assert *validator* satisfies the validator plugin contract."""
    assert isinstance(getattr(validator, "name", None), str) and validator.name, (
        "validator must expose a non-empty string `name`"
    )
    assert callable(getattr(validator, "validate", None)), (
        "validator must define validate(df, policy, ctx)"
    )
    _check_metadata(validator, want_semantic_types=False)

    df = pd.DataFrame({"value": [1, 2, 2, 3], "label": ["a", "b", "b", "c"]})
    before = df.copy(deep=True)
    findings = validator.validate(df, _StubPolicy(), _StubCtx())
    assert isinstance(findings, list), "validate(...) must return a list"
    pd.testing.assert_frame_equal(df, before, obj="validate() must not mutate the frame")
    for f in findings:
        assert isinstance(f, QualityFinding), (
            "every finding must be a freshdata.QualityFinding "
            "(build it with QualityFinding.create(...))"
        )


class _StubPolicy:
    """A minimal, empty policy stand-in for the validator contract test."""

    constraints: tuple[Any, ...] = ()
    unresolved: tuple[Any, ...] = ()
    issues: tuple[Any, ...] = ()


class _StubCtx:
    """A minimal config/context stand-in for the validator contract test."""
