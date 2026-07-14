"""TruthBench backend-parity contracts.

These tests deliberately tamper the collected evidence.  They do not depend on
an optional dependency being absent: a release must reject an incomplete or
dishonest observation even in a fully provisioned development environment.
"""

from __future__ import annotations

from dataclasses import replace
from importlib.metadata import PackageNotFoundError

import pandas as pd
import pytest
from benchmarks.truthbench.fixtures.base import FixtureBuilder
from benchmarks.truthbench.models import Disposition
from benchmarks.truthbench.surfaces import backends
from benchmarks.truthbench.surfaces.backends import (
    EXTENDED_BACKEND_CONTRACTS,
    BackendParityAdapter,
    BackendUnavailableError,
    evaluate_backend_parity,
    preflight_required_backends,
)

import freshdata as fd


def _fixture():
    frame = pd.DataFrame({"name": [" alpha ", "beta"], "amount": [1, 2]}, index=["a", "b"])
    builder = FixtureBuilder("v1", "parity", frame)
    builder.inject(
        "a", "name", " alpha ", Disposition.REPAIR, expected="alpha", family="whitespace"
    )
    return builder.build()


def _config() -> fd.CleanConfig:
    return fd.CleanConfig(
        strategy="conservative", fix_dtypes=False, drop_duplicates=False, verbose=False
    )


def _passing_result():
    return BackendParityAdapter().observe(_fixture(), _config())


def test_required_backends_are_preflighted_without_importing_them(monkeypatch) -> None:
    def missing(name: str) -> str:
        if name == "duckdb":
            raise PackageNotFoundError(name)
        return "test-version"

    monkeypatch.setattr(backends.metadata, "version", missing)
    with pytest.raises(BackendUnavailableError, match="duckdb"):
        preflight_required_backends()


def test_real_native_subset_has_no_fallback_and_passes_parity() -> None:
    result = _passing_result()
    assert result.passed, result.failures
    assert tuple(result.executions) == ("pandas", "polars", "duckdb")
    for backend, execution in result.executions.items():
        assert execution.requested_backend == backend
        assert execution.actual_backend == backend
        assert execution.fallback_events == ()
        assert execution.gold_failures == ()


@pytest.mark.parametrize(
    ("mutator", "needle"),
    [
        (
            lambda e: replace(e, actual_backend="pandas"),
            "requested/actual backend mismatch",
        ),
        (
            lambda e: replace(e, fallback_events=("native delegated",)),
            "unexpected fallback",
        ),
        (
            lambda e: replace(e, row_ids=tuple(reversed(e.row_ids))),
            "row identity/order mismatch",
        ),
        (
            lambda e: replace(e, dtypes={**e.dtypes, "amount": "string"}),
            "dtype divergence",
        ),
        (
            lambda e: replace(e, actions=((*e.actions, ("tampered",)),)),
            "action divergence",
        ),
        (
            lambda e: replace(e, backend_differences=({"detail": "undisclosed"},)),
            "undisclosed backend difference",
        ),
    ],
)
def test_tampered_backend_evidence_fails_parity(mutator, needle: str) -> None:
    result = _passing_result()
    executions = dict(result.executions)
    executions["polars"] = mutator(executions["polars"])
    tampered = evaluate_backend_parity(executions)
    assert not tampered.passed
    assert any(needle in failure for failure in tampered.failures)


def test_value_divergence_fails_even_when_a_report_mentions_it() -> None:
    result = _passing_result()
    executions = dict(result.executions)
    target = executions["duckdb"]
    changed = target.output_frame.copy(deep=True)
    changed.loc[0, "amount"] = 99
    executions["duckdb"] = replace(
        target,
        output_frame=changed,
        backend_differences=({"step": "clean", "detail": "changed amount"},),
    )
    tampered = evaluate_backend_parity(executions)
    assert not tampered.passed
    assert any("value divergence" in failure for failure in tampered.failures)


def test_pandas_is_scored_against_gold_before_comparison() -> None:
    result = _passing_result()
    executions = dict(result.executions)
    reference = executions["pandas"]
    broken = reference.output_frame.copy(deep=True)
    broken.loc[0, "name"] = "wrong"
    executions["pandas"] = replace(
        reference,
        output_frame=broken,
        gold_failures=("v1:parity:a:name",),
    )
    outcome = evaluate_backend_parity(executions)
    assert not outcome.passed
    assert any("pandas failed gold" in failure for failure in outcome.failures)


def test_extended_backends_are_explicitly_marked_as_infrastructure_contracts() -> None:
    assert {contract.backend for contract in EXTENDED_BACKEND_CONTRACTS} == {"spark", "freshcore"}
    assert all(contract.requires_infrastructure for contract in EXTENDED_BACKEND_CONTRACTS)
