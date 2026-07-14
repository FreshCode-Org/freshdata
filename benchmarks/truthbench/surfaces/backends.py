"""Strict, reproducible parity checks for FreshData's native backends.

TruthBench never treats the pandas result as truth merely because it is the
reference representation.  Each backend is first checked against the fixture
oracle, then compared with the other independently observed executions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import pandas as pd

import freshdata as fd

from ..exact import encode_typed, exact_equal
from ..models import Disposition

REQUIRED_BACKENDS: tuple[str, ...] = ("pandas", "polars", "duckdb")
_BACKEND_PACKAGES = {"pandas": "pandas", "polars": "polars", "duckdb": "duckdb"}
_ROW_ID_PREFIX = "truthbench_row_id"


class BackendUnavailableError(RuntimeError):
    """A required backend is not installed for this mandatory profile."""


class BackendParityError(AssertionError):
    """Raised by :func:`assert_backend_parity` when evidence diverges."""


class BackendProvenanceError(RuntimeError):
    """A public cleaning report omitted mandatory backend provenance."""


@dataclass(frozen=True)
class ExtendedBackendContract:
    """An integration-only backend contract, intentionally excluded from CI."""

    backend: str
    required_packages: tuple[str, ...]
    requires_infrastructure: bool
    profile: str = "extended"


EXTENDED_BACKEND_CONTRACTS: tuple[ExtendedBackendContract, ...] = (
    ExtendedBackendContract("spark", ("pyspark",), True),
    ExtendedBackendContract("freshcore", ("freshcore",), True),
)


@dataclass(frozen=True)
class BackendExecution:
    """A normalized record of one backend's public ``fd.clean`` invocation."""

    requested_backend: str
    actual_backend: str
    output_frame: pd.DataFrame
    source_row_ids: tuple[str, ...]
    row_ids: tuple[str, ...]
    dtypes: Mapping[str, str]
    actions: tuple[tuple[Any, ...], ...]
    decision_audit: Mapping[str, Any]
    fallback_events: tuple[Any, ...]
    backend_differences: tuple[Mapping[str, Any], ...]
    gold_failures: tuple[str, ...]
    dependency_version: str


@dataclass(frozen=True)
class BackendParityResult:
    """Collected executions and every reason they cannot be declared equal."""

    executions: Mapping[str, BackendExecution]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def preflight_required_backends(
    required_backends: Sequence[str] = REQUIRED_BACKENDS,
) -> dict[str, str]:
    """Resolve distribution versions without importing optional backends.

    A missing distribution is a release failure, never a skip.  The metadata
    lookup is deliberately injectable through ``importlib.metadata.version``
    in tests, so the test suite does not need to uninstall anything.
    """

    versions: dict[str, str] = {}
    for backend in required_backends:
        package = _BACKEND_PACKAGES.get(backend)
        if package is None:
            raise ValueError(f"unsupported TruthBench backend: {backend!r}")
        try:
            versions[backend] = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise BackendUnavailableError(
                f"required TruthBench backend {backend!r} is unavailable "
                f"(missing distribution {package!r})"
            ) from exc
    return versions


def common_native_config() -> fd.CleanConfig:
    """Return the explicit lowest-common-denominator native cleaning config."""

    return fd.CleanConfig(
        strategy="conservative",
        fix_dtypes=False,
        drop_duplicates=False,
        verbose=False,
    )


def _fixture_frame(fixture: Any) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    frame = getattr(fixture, "frame", fixture)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("backend parity requires a pandas TruthFixture frame")
    source = frame.copy(deep=True)
    row_id_column = _ROW_ID_PREFIX
    suffix = 1
    while row_id_column in source.columns:
        row_id_column = f"{_ROW_ID_PREFIX}_{suffix}"
        suffix += 1
    # Native engines cannot faithfully carry a pandas index.  Keep immutable
    # row identity as a real column and require it to come back in the same
    # order.  This prevents a sort/reset from being misclassified as parity.
    source_row_ids = tuple(str(value) for value in source.index)
    source.insert(0, row_id_column, source_row_ids)
    source.index = pd.RangeIndex(len(source))
    return source, row_id_column, source_row_ids


def _to_pandas(output: Any) -> pd.DataFrame:
    if isinstance(output, pd.DataFrame):
        return output.copy(deep=True)
    method = getattr(output, "to_pandas", None)
    if callable(method):
        converted = method()
        if isinstance(converted, pd.DataFrame):
            return converted.copy(deep=True)
    for name in ("fetchdf", "to_df", "df"):
        method = getattr(output, name, None)
        if callable(method):
            converted = method()
            if isinstance(converted, pd.DataFrame):
                return converted.copy(deep=True)
    raise TypeError(f"backend output cannot be normalized to pandas: {type(output).__name__}")


def _action_fingerprint(report: Any) -> tuple[tuple[Any, ...], ...]:
    """Use the public action fields that express a decision, not timing data."""

    return tuple(
        (
            getattr(action, "step", None),
            getattr(action, "column", None),
            getattr(action, "count", None),
            getattr(action, "rationale", None),
            getattr(action, "risk", None),
            getattr(action, "confidence", None),
            getattr(action, "status", None),
            getattr(action, "human_review", None),
        )
        for action in getattr(report, "actions", ())
    )


def _decision_audit_evidence(report: Any) -> dict[str, Any]:
    """Snapshot public report fields which explain a backend's decisions.

    Timing and memory counters are intentionally excluded: they vary by runner
    and do not explain a data-quality decision.  This is an in-memory
    comparison snapshot; later TruthBench stages scan persisted evidence.
    """

    names = (
        "coerced_cells",
        "columns_dropped",
        "columns_imputed",
        "columns_preserved",
        "warnings",
        "recommendations",
        "domain_findings",
        "domain_repairs",
        "domain_trust_score",
        "decisions_hash",
        "contract_violations",
    )
    return {name: deepcopy(getattr(report, name, None)) for name in names}


def _gold_failures(fixture: Any, output: pd.DataFrame, row_id_column: str) -> tuple[str, ...]:
    """Score every labelled cell before comparing backend outputs.

    ``flag`` and ``review`` do not imply a mutation: their default gold value
    remains the adversarial input.  The later normalizer grades their routing
    evidence; this adapter is limited to value/dtype parity.
    """

    if not hasattr(fixture, "cells"):
        return ()
    if row_id_column not in output.columns:
        return tuple(cell.cell_id for cell in fixture.cells)
    positions = {str(row_id): index for index, row_id in enumerate(output[row_id_column])}
    failures: list[str] = []
    for cell in fixture.cells:
        position = positions.get(str(cell.row_id))
        if position is None or cell.column not in output.columns:
            failures.append(cell.cell_id)
            continue
        actual = output.iloc[position][cell.column]
        actual_dtype = output[cell.column].dtype
        if cell.disposition is Disposition.REPAIR:
            expected = cell.expected_output
            if expected is None:
                failures.append(cell.cell_id)
                continue
            # A repair oracle only constrains dtype when fixture authors opted
            # in with ``expected_dtype``.  Otherwise it constrains the exact
            # repaired scalar while representation parity is checked below.
            dtype = actual_dtype if expected.dtype is not None else None
            if encode_typed(actual, dtype=dtype) != expected:
                failures.append(cell.cell_id)
            continue
        expected = fixture.frame.at[cell.row_id, cell.column]
        if not exact_equal(
            actual,
            expected,
            left_dtype=actual_dtype,
            right_dtype=fixture.frame[cell.column].dtype,
        ):
            failures.append(cell.cell_id)
    return tuple(failures)


def _execute_public_clean(
    fixture: Any,
    config: fd.CleanConfig,
    backend: str,
    version: str,
    cleaner: Callable[..., Any],
) -> BackendExecution:
    source, row_id_column, source_row_ids = _fixture_frame(fixture)
    kwargs: dict[str, Any] = {
        "config": config,
        "engine": backend,
        "return_report": True,
        "engine_config": fd.EngineConfig(
            engine=backend,
            output_format="pandas",
            fallback_policy="error" if backend != "pandas" else "allow",
        ),
    }
    if backend != "pandas":
        kwargs["fallback_policy"] = "error"
    output, report = cleaner(source, **kwargs)
    normalized = _to_pandas(output).reset_index(drop=True)
    requested = getattr(report, "requested_backend", None)
    actual = getattr(report, "backend", None)
    if not isinstance(requested, str) or not requested:
        raise BackendProvenanceError(
            f"backend {backend!r} report omitted requested_backend provenance"
        )
    if not isinstance(actual, str) or not actual:
        raise BackendProvenanceError(
            f"backend {backend!r} report omitted actual backend provenance"
        )
    row_ids = (
        tuple(str(value) for value in normalized[row_id_column])
        if row_id_column in normalized.columns
        else ()
    )
    differences = tuple(getattr(report, "backend_differences", ()) or ())
    return BackendExecution(
        requested_backend=str(requested),
        actual_backend=str(actual),
        output_frame=normalized,
        source_row_ids=source_row_ids,
        row_ids=row_ids,
        dtypes={str(column): str(dtype) for column, dtype in normalized.dtypes.items()},
        actions=_action_fingerprint(report),
        decision_audit=_decision_audit_evidence(report),
        fallback_events=tuple(getattr(report, "fallback_events", ()) or ()),
        backend_differences=tuple(dict(item) for item in differences),
        gold_failures=_gold_failures(fixture, normalized, row_id_column),
        dependency_version=version,
    )


def _approved_dtype_equivalent(
    left: str, right: str, left_values: pd.Series, right_values: pd.Series
) -> bool:
    if left == right:
        return True
    # Only nullable-vs-NumPy forms of the *same* logical family are approved.
    pairs = {
        frozenset(("int64", "Int64")),
        frozenset(("int32", "Int32")),
        frozenset(("float64", "Float64")),
        frozenset(("float32", "Float32")),
        frozenset(("bool", "boolean")),
        frozenset(("object", "string")),
    }
    if frozenset((left, right)) not in pairs:
        return False
    if {left, right} == {"object", "string"}:
        return all(isinstance(value, str) for value in left_values.dropna()) and all(
            isinstance(value, str) for value in right_values.dropna()
        )
    return True


def _frames_equal(
    reference: BackendExecution, candidate: BackendExecution
) -> tuple[bool, str | None]:
    left, right = reference.output_frame, candidate.output_frame
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False, "schema divergence"
    for column in left.columns:
        if not _approved_dtype_equivalent(
            reference.dtypes[str(column)],
            candidate.dtypes[str(column)],
            left[column],
            right[column],
        ):
            return False, f"dtype divergence in {column!r}"
    try:
        pd.testing.assert_frame_equal(
            left, right, check_dtype=False, check_like=False, check_exact=True
        )
    except AssertionError:
        return False, "value divergence"
    return True, None


def evaluate_backend_parity(
    executions: Mapping[str, BackendExecution],
    *,
    required_backends: Sequence[str] = REQUIRED_BACKENDS,
) -> BackendParityResult:
    """Evaluate parity without trusting any backend as an unscored oracle."""

    failures: list[str] = []
    for backend in required_backends:
        execution = executions.get(backend)
        if execution is None:
            failures.append(f"missing required backend execution: {backend}")
            continue
        if execution.requested_backend != backend or execution.actual_backend != backend:
            failures.append(f"requested/actual backend mismatch for {backend}")
        if execution.fallback_events:
            failures.append(f"unexpected fallback for {backend}")
        if execution.gold_failures:
            failures.append(f"{backend} failed gold: {', '.join(execution.gold_failures)}")
        if execution.backend_differences:
            failures.append(f"undisclosed backend difference for {backend}")
        if execution.row_ids != execution.source_row_ids:
            failures.append(f"source row identity/order mismatch for {backend}")

    reference = executions.get("pandas")
    if reference is not None:
        for backend in required_backends:
            if backend == "pandas" or backend not in executions:
                continue
            candidate = executions[backend]
            if candidate.row_ids != reference.row_ids:
                failures.append(f"row identity/order mismatch for {backend}")
            if candidate.actions != reference.actions:
                failures.append(f"action divergence for {backend}")
            if candidate.decision_audit != reference.decision_audit:
                failures.append(f"decision/audit divergence for {backend}")
            equal, reason = _frames_equal(reference, candidate)
            if not equal:
                failures.append(f"{reason} for {backend}")
    return BackendParityResult(executions=dict(executions), failures=tuple(failures))


def assert_backend_parity(
    executions: Mapping[str, BackendExecution],
    *,
    required_backends: Sequence[str] = REQUIRED_BACKENDS,
) -> BackendParityResult:
    """Return a passing result or raise a single audit-friendly assertion."""

    result = evaluate_backend_parity(executions, required_backends=required_backends)
    if not result.passed:
        raise BackendParityError("; ".join(result.failures))
    return result


def exercise_extended_backend_contract(
    contract: ExtendedBackendContract,
    fixture: Any,
    config: fd.CleanConfig | None = None,
    *,
    cleaner: Callable[..., Any] = fd.clean,
    dependency_version: str = "extended-fake",
) -> BackendParityResult:
    """Exercise Spark/FreshCore adapter and gate behavior with a supplied fake.

    The extended CI profile supplies an actual cleaner and infrastructure.  Unit
    tests deliberately inject a fake public-call boundary, retaining strict
    provenance/fallback/gold checks without requiring Spark or FreshCore here.
    """

    if contract.backend not in {item.backend for item in EXTENDED_BACKEND_CONTRACTS}:
        raise ValueError(f"unknown extended TruthBench backend: {contract.backend!r}")
    execution = _execute_public_clean(
        fixture,
        config or common_native_config(),
        contract.backend,
        dependency_version,
        cleaner,
    )
    return evaluate_backend_parity(
        {contract.backend: execution}, required_backends=(contract.backend,)
    )


class BackendParityAdapter:
    """Run all mandatory backends through the public FreshData API only."""

    name = "backend_parity"

    def __init__(self, cleaner: Callable[..., Any] = fd.clean) -> None:
        self._cleaner = cleaner

    def observe(self, fixture: Any, config: fd.CleanConfig | None = None) -> BackendParityResult:
        chosen = config or common_native_config()
        versions = preflight_required_backends()
        executions = {
            backend: _execute_public_clean(
                fixture, chosen, backend, versions[backend], self._cleaner
            )
            for backend in REQUIRED_BACKENDS
        }
        return evaluate_backend_parity(executions)


# A short alias keeps integrations that use one-adapter-per-surface naming tidy.
BackendAdapter = BackendParityAdapter


__all__ = [
    "BackendAdapter",
    "BackendExecution",
    "BackendParityAdapter",
    "BackendParityError",
    "BackendParityResult",
    "BackendProvenanceError",
    "BackendUnavailableError",
    "EXTENDED_BACKEND_CONTRACTS",
    "ExtendedBackendContract",
    "REQUIRED_BACKENDS",
    "assert_backend_parity",
    "common_native_config",
    "evaluate_backend_parity",
    "exercise_extended_backend_contract",
    "preflight_required_backends",
]
