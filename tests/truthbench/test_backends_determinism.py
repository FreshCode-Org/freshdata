"""TruthBench backend-parity contracts.

These tests deliberately tamper the collected evidence.  They do not depend on
an optional dependency being absent: a release must reject an incomplete or
dishonest observation even in a fully provisioned development environment.
"""

from __future__ import annotations

from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pandas as pd
import pytest
from benchmarks.truthbench.fixtures.base import FixtureBuilder
from benchmarks.truthbench.models import Disposition
from benchmarks.truthbench.surfaces import backends
from benchmarks.truthbench.surfaces.backends import (
    EXTENDED_BACKEND_CONTRACTS,
    BackendParityAdapter,
    BackendProvenanceError,
    BackendUnavailableError,
    evaluate_backend_parity,
    exercise_extended_backend_contract,
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


def test_shared_row_reorder_fails_against_fixture_order_not_only_pandas() -> None:
    result = _passing_result()
    executions = {
        backend: replace(execution, row_ids=tuple(reversed(execution.row_ids)))
        for backend, execution in result.executions.items()
    }
    outcome = evaluate_backend_parity(executions)
    assert not outcome.passed
    assert {
        failure for failure in outcome.failures if "source row identity/order mismatch" in failure
    } == {
        "source row identity/order mismatch for pandas",
        "source row identity/order mismatch for polars",
        "source row identity/order mismatch for duckdb",
    }


def test_missing_backend_provenance_fails_instead_of_being_inferred() -> None:
    def unprovenanced_cleaner(source, **_kwargs):
        return source, SimpleNamespace(actions=(), fallback_events=(), backend_differences=())

    with pytest.raises(BackendProvenanceError, match="requested_backend"):
        BackendParityAdapter(cleaner=unprovenanced_cleaner).observe(_fixture(), _config())


def test_tampered_report_decision_audit_evidence_fails_parity() -> None:
    result = _passing_result()
    executions = dict(result.executions)
    target = executions["duckdb"]
    executions["duckdb"] = replace(
        target,
        decision_audit={**target.decision_audit, "coerced_cells": {"amount": {"a": "raw"}}},
    )
    outcome = evaluate_backend_parity(executions)
    assert not outcome.passed
    assert any("decision/audit divergence" in failure for failure in outcome.failures)


def test_extended_backends_are_exercised_with_fake_public_cleaner() -> None:
    assert {contract.backend for contract in EXTENDED_BACKEND_CONTRACTS} == {"spark", "freshcore"}
    assert all(contract.requires_infrastructure for contract in EXTENDED_BACKEND_CONTRACTS)

    calls = []

    def fake_cleaner(source, **kwargs):
        calls.append(kwargs)
        backend = kwargs["engine"]
        return source.copy(deep=True), SimpleNamespace(
            requested_backend=backend,
            backend=backend,
            actions=(),
            fallback_events=(),
            backend_differences=(),
            coerced_cells={},
            columns_dropped=(),
            columns_imputed=(),
            columns_preserved=(),
            warnings=(),
            recommendations=(),
            domain_findings=(),
            domain_repairs=(),
            domain_trust_score=None,
            decisions_hash=None,
            contract_violations=None,
        )

    for contract in EXTENDED_BACKEND_CONTRACTS:
        result = exercise_extended_backend_contract(
            contract,
            pd.DataFrame({"value": [1, 2]}),
            _config(),
            cleaner=fake_cleaner,
        )
        assert result.passed, result.failures
    assert [call["engine"] for call in calls] == ["spark", "freshcore"]
    assert all(call["fallback_policy"] == "error" for call in calls)
