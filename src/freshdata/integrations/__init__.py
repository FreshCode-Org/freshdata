"""freshdata.integrations — run freshdata's clean + trust gate inside orchestrators.

The framework-agnostic core (:func:`evaluate_trust_gate`, :class:`TrustGateResult`)
is always importable and has no optional dependencies. The orchestrator adapters are
opt-in submodules, each guarded so the *module* imports cleanly even when its
framework is absent; the framework is required only when you actually use it:

>>> from freshdata.integrations import evaluate_trust_gate          # always works
>>> from freshdata.integrations.dagster import freshdata_asset_check  # needs dagster
>>> from freshdata.integrations.airflow import FreshDataCleanOperator # needs airflow
>>> from freshdata.integrations.dbt import FreshDataDbtTransform       # needs dbt-core

Install the extras as needed, e.g. ``pip install "freshdata[dagster]"`` or
``pip install "freshdata[integrations]"`` for all three.
"""

from __future__ import annotations

from ._core import (
    OnLowScore,
    TrustGateError,
    TrustGateResult,
    evaluate_trust_gate,
)

__all__ = [
    "OnLowScore",
    "TrustGateError",
    "TrustGateResult",
    "evaluate_trust_gate",
]
