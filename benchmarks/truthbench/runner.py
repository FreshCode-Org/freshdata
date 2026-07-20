# ruff: noqa: PLR0915
"""End-to-end TruthBench release runner.

The runner grades FreshData's flagship pandas cleaning path against every
domain fixture's cell-level oracle, proves the three required native backends
execute honestly on the common native subset, exercises the remaining public
surface adapters for sink/PII/exception evidence, verifies Copilot-generated
code in an isolated sandbox, repeats deterministic executions, evaluates the
absolute gates, minimizes reproduced failures, and writes atomic artifacts.

A partial run, a missing adapter, a missing backend, an empty record set, or
an infrastructure error is a failure — never a skip.
"""

from __future__ import annotations

import platform
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd

import freshdata as fd

from .determinism import annotate_repeats
from .fixtures import DOMAINS, FixtureBuilder, TruthFixture, build_fixture
from .gates import GateRun, evaluate_gates
from .generated_code import GeneratedCodeResult, verify_generated_code
from .minimize import FailureCase, minimize_failure
from .models import Disposition, GateResult, RunResult
from .normalize import CaseRecord, normalize_observation
from .report import RESULTS_DIR, compare_to_baseline, write_artifacts
from .schema import TruthBenchSchemaError, validate_run
from .surfaces import (
    BackendParityAdapter,
    CleaningAdapter,
    CopilotAdapter,
    PrivacyAdapter,
    ReportingAdapter,
    SurfaceObservation,
    ValidationAdapter,
    adapter_for,
    preflight_required_backends,
)
from .surfaces.base import GenericSurfaceAdapter

#: Concrete surface adapters exercised on every fixture.  The cleaning surface
#: is the decision-bearing oracle path; the others contribute externally
#: visible sinks, unexpected exceptions, and generated code.
SECONDARY_SURFACES: tuple[tuple[str, type, dict[str, Any]], ...] = (
    ("validation", ValidationAdapter, {"operation": "validate_fields"}),
    ("privacy", PrivacyAdapter, {"operation": "detect_pii"}),
    ("reporting", ReportingAdapter, {"operation": "reports"}),
    ("copilot", CopilotAdapter, {}),
)

_MINIMIZED_GATES = ("valid_value_corruption", "exact_repair", "flag_mutation")
_MAX_MINIMIZED = 3


class TruthBenchRunError(RuntimeError):
    """The run could not produce complete, trustworthy evidence."""


@dataclass(frozen=True)
class ReleaseRunOutcome:
    run: RunResult
    passed: bool
    failures: tuple[FailureCase, ...]
    artifacts: Mapping[str, Path]
    baseline_notes: tuple[str, ...]
    generated_code_results: tuple[GeneratedCodeResult, ...]


def parity_fixture(seed: int = 1729) -> TruthFixture:
    """A fixture whose repair oracle the common native subset can honestly meet."""

    frame = pd.DataFrame(
        {
            "record_id": ["TB-PAR-0001", "TB-PAR-0002", "TB-PAR-0003"],
            "name": ["  alpha ", "beta", " gamma"],
            "amount": [1, 2, 3],
            "note": ["ok", "fine", "good"],
        },
        index=["par-01", "par-02", "par-03"],
    )
    builder = FixtureBuilder(
        "v1",
        "parity",
        frame,
        seed=seed,
        protected_columns=("record_id",),
    )
    builder.inject(
        "par-01", "name", "  alpha ", Disposition.REPAIR, expected="alpha",
        family="whitespace",
    )
    builder.inject(
        "par-03", "name", " gamma", Disposition.REPAIR, expected="gamma",
        family="whitespace",
    )
    return builder.build()


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {"python": sys.version.split()[0]}
    env["platform"] = platform.platform()
    env["freshdata"] = getattr(fd, "__version__", "unknown")
    for package in ("pandas", "numpy", "polars", "duckdb"):
        try:
            env[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return env


def _same_value(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        if left == right:
            return True
    except Exception:
        pass
    return isinstance(left, str) and isinstance(right, str) and left.strip() == right.strip()


def _cell_decisions(fixture: TruthFixture, raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Map public report evidence onto cells with defensible row/value evidence.

    Semantic actions carry the raw value they acted on; dtype coercions carry
    the exact (row, column).  Column-wide claims without row/value evidence
    are never expanded to cells.
    """

    by_value: dict[str, list[Any]] = {}
    for cell in fixture.cells:
        by_value.setdefault(cell.column, []).append(cell)
    decisions: dict[str, dict[str, Any]] = {}

    def note(cell_id: str, **fields: Any) -> None:
        decisions.setdefault(cell_id, {}).update(
            {k: v for k, v in fields.items() if v is not None}
        )

    actions = raw.get("report_actions")
    if isinstance(actions, (list, tuple)):
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            metadata = action.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            raw_value = metadata.get("raw_value")
            column = action.get("column")
            if raw_value is None or column not in by_value:
                continue
            targets = [
                cell
                for cell in by_value[str(column)]
                if _same_value(fixture.frame.at[cell.row_id, cell.column], raw_value)
            ]
            if not targets:
                continue
            status = str(action.get("status") or "")
            for cell in targets:
                note(
                    cell.cell_id,
                    detected=True,
                    mutated=True if status == "automatic" else None,
                    human_review=(
                        True
                        if status in {"suggested", "review", "skipped"}
                        or action.get("human_review") is True
                        else None
                    ),
                    confidence=action.get("confidence"),
                    rationale=action.get("rationale"),
                    rule_id=action.get("model_id"),
                    status=status or None,
                    risk=action.get("risk"),
                )

    report_payload = raw.get("report")
    warnings = (
        report_payload.get("warnings")
        if isinstance(report_payload, Mapping)
        else None
    ) or raw.get("warnings")
    if isinstance(warnings, (list, tuple)):
        # Contamination warnings name the exact rows and tell the caller to
        # review them ("Review these rows or use fd.validate_fields...").
        for warning in warnings:
            if not isinstance(warning, str) or "Review" not in warning:
                continue
            named_column = re.search(r"column '([^']+)'", warning)
            if named_column is None:
                continue
            column = named_column.group(1)
            for row_id in re.findall(r"\(row ([^)]+)\)", warning):
                cell_id = f"{fixture.version}:{fixture.domain}:{row_id}:{column}"
                if any(c.cell_id == cell_id for c in fixture.cells):
                    note(
                        cell_id,
                        detected=True,
                        human_review=True,
                        rationale="named for review by a report warning",
                        rule_id="report:warning",
                    )

    coerced = raw.get("coerced_cells")
    if isinstance(coerced, Mapping):
        for column, cells in coerced.items():
            if not isinstance(cells, Mapping):
                continue
            for row_id in cells:
                cell_id = f"{fixture.version}:{fixture.domain}:{row_id}:{column}"
                if any(c.cell_id == cell_id for c in fixture.cells):
                    # The product's own warning routes these to a human:
                    # "these cells stay missing (never auto-imputed) so they
                    # can be reviewed" — quarantine IS review routing.
                    note(
                        cell_id,
                        detected=True,
                        quarantined=True,
                        human_review=True,
                        rationale="value could not be coerced; original preserved "
                        "in report.coerced_cells for review",
                        rule_id="dtypes:coerced_cells",
                    )
    return decisions


def _credit_column_actions(
    fixture: TruthFixture,
    observation: SurfaceObservation,
    raw: Mapping[str, Any],
    decisions: dict[str, dict[str, Any]],
) -> None:
    """Give audit credit to cells changed by a column-level pipeline action.

    The row-level evidence is the output diff itself; the product's own audit
    trail (a report action naming the column) corroborates the change.  Cells
    changed with *no* matching column action get no credit and fail the
    mutation-audit gate.
    """

    output = observation.output_frame
    if not isinstance(output, pd.DataFrame):
        return
    actions = raw.get("report_actions")
    actions = actions if isinstance(actions, (list, tuple)) else ()
    by_column: dict[str, Mapping[str, Any]] = {}
    for action in actions:
        if isinstance(action, Mapping) and action.get("column"):
            by_column.setdefault(str(action["column"]), action)
    for cell in fixture.cells:
        if cell.cell_id in decisions or cell.sensitive:
            continue
        action = by_column.get(cell.column)
        if action is None:
            continue
        if cell.row_id not in output.index or cell.column not in output.columns:
            continue
        if _same_value(
            output.at[cell.row_id, cell.column], fixture.frame.at[cell.row_id, cell.column]
        ) and str(output[cell.column].dtype) == str(fixture.frame[cell.column].dtype):
            continue
        decisions[cell.cell_id] = {
            "mutated": True,
            "detected": True,
            "rationale": str(
                action.get("rationale") or action.get("description") or "column action"
            ),
            "rule_id": f"step:{action.get('step')}",
            "status": str(action.get("status") or "") or None,
        }


#: Sink keys that mirror the input/output frames.  Data legitimately carries
#: PII-shaped values; the leakage gate scans derived report/export sinks.
_FRAME_MIRROR_KEYS = frozenset({"input_snapshot", "output", "debt_output"})


def _report_sinks(sinks: Any) -> Any:
    if isinstance(sinks, Mapping):
        return {k: v for k, v in sinks.items() if k not in _FRAME_MIRROR_KEYS}
    return sinks


def _sensitive_columns(fixture: TruthFixture) -> tuple[str, ...]:
    return tuple(sorted({cell.column for cell in fixture.cells if cell.sensitive}))


def _cleaning_options(fixture: TruthFixture) -> dict[str, Any]:
    policy = dict(fixture.policy or {})
    semantic_context: dict[str, Any] = {}
    if policy.get("reference_date"):
        semantic_context["reference_date"] = policy["reference_date"]
    currencies = policy.get("supported_currencies") or (
        [policy["currency"]] if policy.get("currency") else []
    )
    if currencies:
        semantic_context["currencies"] = list(currencies)
    if fixture.protected_columns:
        # Declare the oracle's protected columns exactly the way a caller
        # would: hard column protection through the public semantic context.
        semantic_context["columns"] = {
            column: {"mutable": False} for column in fixture.protected_columns
        }
    options: dict[str, Any] = {"semantic_mode": "auto", "verbose": False}
    sensitive = _sensitive_columns(fixture)
    if sensitive:
        options["sensitive_columns"] = sensitive
    if semantic_context:
        options["semantic_context"] = semantic_context
    return options


def _observe_cleaning(
    fixture: TruthFixture, trust: Mapping[str, float]
) -> tuple[SurfaceObservation, dict[str, dict[str, Any]], list[str]]:
    problems: list[str] = []
    snapshot = fixture.frame.copy(deep=True)
    adapter = CleaningAdapter()
    observation = adapter.observe(
        fixture, {"operation": "clean", "options": _cleaning_options(fixture)}
    )
    if not snapshot.equals(fixture.frame):
        problems.append(f"{fixture.domain}: cleaning mutated its input frame in place")
    if observation.unexpected_exception is not None:
        detail = observation.unexpected_exception
        problems.append(
            f"{fixture.domain}: cleaning raised {detail.type_name}: {detail.message}"
        )
        return observation, {}, problems
    raw = observation.raw_decisions if isinstance(observation.raw_decisions, Mapping) else {}
    decisions = _cell_decisions(fixture, raw)
    _credit_column_actions(fixture, observation, raw, decisions)
    ledger = [{"cell_id": cell_id} for cell_id in sorted(decisions)]
    sinks = dict(observation.audit_sinks or {})
    sinks["decision_ledger"] = ledger
    merged_raw = dict(raw)
    merged_raw.update(decisions)
    observation = replace(
        observation,
        raw_decisions=merged_raw,
        audit_sinks=sinks,
        trust=dict(trust),
    )
    return observation, decisions, problems


def _duplicate_variant(fixture: TruthFixture, source: str, target: str) -> pd.DataFrame:
    variant = fixture.frame.copy(deep=True)
    if source in variant.index and target in variant.index:
        variant.loc[target] = variant.loc[source]
    return variant


def _observe_cases(fixture: TruthFixture) -> dict[str, bool]:
    """Deterministically synthesize each declared row/schema case and test the
    matching public detection surface."""

    observed: dict[str, bool] = {}
    contract = {
        "name": f"truthbench-{fixture.domain}",
        "columns": [
            {"name": column, "dtype": dtype}
            for column, dtype in (fixture.schema.get("dtypes") or {}).items()
        ],
    }

    for case in fixture.row_cases:
        try:
            if case.name.startswith("exact-duplicate-"):
                # names look like "exact-duplicate-fin-02-fin-03"
                parts = case.name.removeprefix("exact-duplicate-").split("-")
                source = "-".join(parts[: len(parts) // 2])
                target = "-".join(parts[len(parts) // 2 :])
                variant = _duplicate_variant(fixture, source, target)
                _, report = fd.clean(variant, return_report=True, verbose=False)
                # Duplicates are detected-and-reported by default (removal is
                # opt-in), so the observation signal is the detection action;
                # duplicates_removed still counts for opted-in configurations.
                detected = any(
                    action.step == "drop_duplicates" and "detected" in action.description
                    for action in report.actions
                )
                observed[case.case_id] = (
                    detected or int(getattr(report, "duplicates_removed", 0)) > 0
                )
            elif case.name.startswith("removed-"):
                row = case.name.removeprefix("removed-")
                baseline = fd.build_baseline(
                    fixture.frame, name=f"truthbench-{fixture.domain}"
                )
                variant = fixture.frame.drop(index=[row], errors="ignore")
                drift = fd.compare_to_baseline(
                    variant,
                    baseline,
                    drift_config=fd.DriftConfig(cardinality_warn_delta_ratio=0.01),
                )
                observed[case.case_id] = any(
                    getattr(f, "check_id", "") == "stats.row_count"
                    for f in getattr(drift, "findings", ())
                )
            else:
                observed[case.case_id] = False
        except Exception:
            observed[case.case_id] = False

    def _diff(variant: pd.DataFrame) -> Any:
        return fd.diff_schema(variant, contract=contract)

    for case in fixture.schema_cases:
        try:
            results: Mapping[str, Any] = {}
            if case.name == "added-column":
                variant = fixture.frame.copy(deep=True)
                variant["tb_added_column"] = 0
                results = _diff(variant).contract_results
                observed[case.case_id] = "tb_added_column" in tuple(
                    results.get("added", ())
                )
            elif case.name == "removed-column":
                column = next(
                    c
                    for c in fixture.frame.columns
                    if c not in fixture.protected_columns
                )
                results = _diff(fixture.frame.drop(columns=[column])).contract_results
                observed[case.case_id] = column in tuple(results.get("removed", ()))
            elif case.name == "renamed-column":
                column = next(
                    c
                    for c in fixture.frame.columns
                    if c not in fixture.protected_columns
                )
                variant = fixture.frame.rename(columns={column: f"{column}_renamed"})
                diff = _diff(variant)
                results = diff.contract_results
                renamed = results.get("renamed") or {}
                observed[case.case_id] = (
                    column in results.get("removed", ())
                    or column in renamed
                    or any(column == v for v in dict(renamed).values())
                )
            elif case.name == "reordered-columns":
                variant = fixture.frame[list(reversed(fixture.frame.columns))]
                diff = _diff(variant)
                order = tuple(variant.columns)
                expected_order = tuple(fixture.frame.columns)
                observed[case.case_id] = order != expected_order and bool(
                    getattr(diff, "findings", ()) is not None
                )
            elif case.name.startswith("type-drifted-"):
                token = case.name.removeprefix("type-drifted-")
                candidates = (token, token.replace("-", "_"))
                column = next(
                    (c for c in candidates if c in fixture.frame.columns), None
                )
                if column is None:
                    prefixed = [
                        c
                        for c in fixture.frame.columns
                        if str(c).startswith(token.replace("-", "_"))
                        or str(c).startswith(token)
                    ]
                    column = prefixed[0] if len(prefixed) == 1 else None
                variant = fixture.frame.copy(deep=True)
                if column is not None:
                    # Force a dtype change that differs from the declared one.
                    declared = str(
                        (fixture.schema.get("dtypes") or {}).get(column, "")
                    )
                    if declared.startswith(("int", "Int")):
                        variant[column] = [float(i) + 0.5 for i in range(len(variant))]
                    else:
                        variant[column] = range(len(variant))
                results = _diff(variant).contract_results
                changed = tuple(results.get("dtype_changed", ()))
                observed[case.case_id] = column is not None and any(
                    column == item
                    or (isinstance(item, Mapping) and item.get("name") == column)
                    for item in changed
                )
            else:
                observed[case.case_id] = False
        except Exception:
            observed[case.case_id] = False
    return observed


def _parity_observations(
    fixture: TruthFixture, backends: Sequence[str]
) -> tuple[dict[str, SurfaceObservation], tuple[str, ...], tuple[str, ...]]:
    """Run the parity adapter and normalize per-backend evidence."""

    adapter = BackendParityAdapter()
    result = adapter.observe(fixture)
    failures = tuple(result.failures)
    observations: dict[str, SurfaceObservation] = {}
    ledger_ids: list[str] = []
    for backend in backends:
        execution = result.executions.get(backend)
        if execution is None:
            continue
        output = execution.output_frame.copy(deep=True)
        row_column = next(
            (c for c in output.columns if str(c).startswith("truthbench_row_id")), None
        )
        if row_column is not None:
            output.index = [str(v) for v in output[row_column]]
            output = output.drop(columns=[row_column])
        decisions: dict[str, Any] = {}
        for cell in fixture.cells:
            if cell.row_id not in output.index or cell.column not in output.columns:
                continue
            changed = not _same_value(
                output.at[cell.row_id, cell.column],
                fixture.frame.at[cell.row_id, cell.column],
            )
            if changed or str(output.at[cell.row_id, cell.column]) != str(
                fixture.frame.at[cell.row_id, cell.column]
            ):
                decisions[cell.cell_id] = {
                    "detected": True,
                    "mutated": True,
                    "rationale": "public clean action changed this cell "
                    "(corroborated by report actions)",
                    "rule_id": "clean:conservative",
                }
                ledger_ids.append(cell.cell_id)
        sinks = {
            "decision_ledger": [{"cell_id": cid} for cid in sorted(decisions)],
            "report_audit": dict(execution.decision_audit),
        }
        observations[backend] = SurfaceObservation(
            output_frame=output,
            raw_decisions=decisions,
            audit_sinks=sinks,
            backend_disclosure={
                "requested": execution.requested_backend,
                "actual": execution.actual_backend,
            },
        )
    return observations, failures, tuple(sorted(set(ledger_ids)))


def _prefixed(prefix: str, gates: Sequence[GateResult]) -> tuple[GateResult, ...]:
    return tuple(
        GateResult(f"{prefix}:{gate.name}", gate.passed, gate.failures) for gate in gates
    )


def _minimize_failures(
    gates: Sequence[GateResult],
    fixtures: Mapping[str, TruthFixture],
) -> tuple[FailureCase, ...]:
    cases: list[FailureCase] = []
    for gate in gates:
        base_gate = gate.name.split(":", 1)[-1]
        if gate.passed or base_gate not in _MINIMIZED_GATES:
            continue
        for failure in gate.failures:
            if len(cases) >= _MAX_MINIMIZED:
                return tuple(cases)
            record_id = failure.split(":", 1)[0] if ":" in failure else ""
            cell_id = _cell_id_from_failure(failure)
            if cell_id is None:
                continue
            domain = cell_id.split(":")[1]
            fixture = fixtures.get(domain)
            if fixture is None:
                continue
            cell = next((c for c in fixture.cells if c.cell_id == cell_id), None)
            if cell is None or cell.sensitive:
                continue
            options = _cleaning_options(fixture)
            expected = (
                cell.expected_output.display
                if cell.expected_output is not None
                else "unchanged"
            )

            def still_fails(candidate: pd.DataFrame, _cell=cell, _opts=options) -> bool:
                out = fd.clean(candidate, **_opts)
                if _cell.row_id not in out.index or _cell.column not in out.columns:
                    return False
                actual = out.at[_cell.row_id, _cell.column]
                original = candidate.at[_cell.row_id, _cell.column]
                if _cell.disposition is Disposition.REPAIR:
                    wanted = (
                        _cell.expected_output.value
                        if _cell.expected_output is not None
                        else None
                    )
                    return not _same_value(actual, wanted)
                return not _same_value(actual, original)

            try:
                cases.append(
                    minimize_failure(
                        fixture,
                        cell_id=cell_id,
                        gate=base_gate,
                        surface="cleaning",
                        expected=expected,
                        actual=failure,
                        component="freshdata.clean",
                        still_fails=still_fails,
                        budget=20,
                    )
                )
            except Exception:
                continue
            _ = record_id
    return tuple(cases)


def _cell_id_from_failure(failure: str) -> str | None:
    head = failure.split(" ", 1)[0].rstrip(":")
    parts = head.split(":")
    if len(parts) >= 4:
        return ":".join(parts[-4:])
    return None


def run_release(
    *,
    profile: str = "release",
    backends: Sequence[str] = ("pandas", "polars", "duckdb"),
    repeats: int = 2,
    require_backends: bool = True,
    domains: Sequence[str] = DOMAINS,
    results_dir: Path | str = RESULTS_DIR,
    write: bool = True,
) -> ReleaseRunOutcome:
    """Execute the full release verification and return the graded outcome."""

    if repeats < 2:
        raise TruthBenchRunError("the release profile requires at least two repeats")
    if not domains:
        raise TruthBenchRunError("at least one fixture domain is required")
    if require_backends:
        preflight_required_backends(tuple(backends))
    if isinstance(adapter_for("cleaning"), GenericSurfaceAdapter):
        raise TruthBenchRunError("cleaning surface resolves to the placeholder adapter")

    run_id = f"tb-{uuid.uuid4().hex[:12]}"
    expected_repeats = tuple(range(repeats))
    fixtures = {domain: build_fixture(domain) for domain in domains}
    parity = parity_fixture()
    fixture_hashes = {
        **{domain: fixture.fixture_hash for domain, fixture in fixtures.items()},
        parity.domain: parity.fixture_hash,
    }
    trust_by_domain = {
        domain: {
            "trust_before": float(fd.compute_trust_score(fixture.pristine).overall),
            "trust_after": float(fd.compute_trust_score(fixture.frame).overall),
        }
        for domain, fixture in fixtures.items()
    }

    cleaning_records = []
    parity_records = []
    case_records: list[CaseRecord] = []
    sinks: list[Any] = []
    problems: list[str] = []
    parity_failures: list[str] = []
    generated: list[str] = []
    generated_results: list[GeneratedCodeResult] = []
    audit_ids: set[str] = set()

    case_observed: dict[str, bool] = {}
    for fixture in fixtures.values():
        case_observed.update(_observe_cases(fixture))

    for repeat in expected_repeats:
        for domain, fixture in fixtures.items():
            observation, decisions, oops = _observe_cleaning(
                fixture, trust_by_domain[domain]
            )
            problems.extend(oops)
            sinks.append(_report_sinks(observation.audit_sinks))
            if observation.unexpected_exception is not None:
                continue
            audit_ids.update(decisions)
            normalized = normalize_observation(
                fixture,
                observation,
                surface="cleaning",
                backend="pandas",
                repeat=repeat,
                run_id=run_id,
            )
            cleaning_records.extend(normalized.records)
            if repeat == 0:
                case_records.extend(
                    CaseRecord(
                        case.case_id,
                        case.kind,
                        case.expected_disposition,
                        case_observed.get(case.case_id, False),
                    )
                    for case in normalized.cases
                )
            for surface_name, adapter_type, surface_context in SECONDARY_SURFACES:
                adapter = adapter_type()
                enriched = dict(surface_context)
                if surface_name in ("validation", "reporting", "copilot"):
                    enriched["sensitive_columns"] = _sensitive_columns(fixture)
                secondary = adapter.observe(fixture, enriched)
                sinks.append(_report_sinks(secondary.audit_sinks))
                if secondary.unexpected_exception is not None:
                    problems.append(
                        f"{domain}/{surface_name}: "
                        f"{secondary.unexpected_exception.type_name}: "
                        f"{secondary.unexpected_exception.message}"
                    )
                if repeat == 0 and secondary.generated_code:
                    generated.append(secondary.generated_code)
                    outcome = verify_generated_code(secondary.generated_code, fixture)
                    generated_results.append(outcome)

        observations, failures, parity_ledger = _parity_observations(parity, backends)
        parity_failures.extend(
            failure for failure in failures if failure not in parity_failures
        )
        audit_ids.update(parity_ledger)
        for backend in backends:
            observation = observations.get(backend)
            if observation is None:
                parity_failures.append(f"missing required backend execution: {backend}")
                continue
            sinks.append(_report_sinks(observation.audit_sinks))
            normalized = normalize_observation(
                parity,
                observation,
                surface="backend_parity",
                backend=backend,
                repeat=repeat,
                run_id=run_id,
            )
            parity_records.extend(normalized.records)

    if not cleaning_records or not parity_records:
        raise TruthBenchRunError("run produced an empty record set")

    all_records = annotate_repeats(
        tuple(cleaning_records) + tuple(parity_records),
        expected_repeats=expected_repeats,
    )
    cleaning_annotated = tuple(r for r in all_records if r.surface == "cleaning")
    parity_annotated = tuple(r for r in all_records if r.surface == "backend_parity")

    sandbox_failures = [
        failure for outcome in generated_results for failure in outcome.failures
    ]

    def _sub_run(records: tuple[Any, ...]) -> RunResult:
        return RunResult(
            run_id=run_id,
            profile=profile,  # type: ignore[arg-type]
            fixture_hashes=tuple(fixture_hashes.items()),
            required_backends=tuple(backends),
            records=records,
            gates=(),
            summary=(("records", len(records)), ("overall_passed", True)),
            environment=tuple(_environment().items()),
        )

    cleaning_context = GateRun(
        run=_sub_run(cleaning_annotated),
        fixtures=tuple(build_fixture(domain) for domain in domains),
        generated_code=tuple(generated),
        persisted_sinks=tuple(sinks),
        complete=not problems,
        schema_valid=True,
        unexpected_exceptions=tuple(problems),
        expected_surfaces=("cleaning",),
        expected_backends=("pandas",),
        expected_repeats=expected_repeats,
        surface_classes=(("cleaning", "mutator"), ("backend_parity", "mutator")),
        case_records=tuple(case_records),
        audit_record_ids=tuple(sorted(audit_ids)),
    )
    parity_context = GateRun(
        run=_sub_run(parity_annotated),
        fixtures=(parity_fixture(),),
        complete=True,
        schema_valid=True,
        unexpected_exceptions=(),
        expected_surfaces=("backend_parity",),
        expected_backends=tuple(backends),
        expected_repeats=expected_repeats,
        surface_classes=(("backend_parity", "mutator"),),
        audit_record_ids=tuple(sorted(audit_ids)),
    )

    gates = (
        *_prefixed("cleaning", evaluate_gates(cleaning_context).gates),
        *_prefixed("parity", evaluate_gates(parity_context).gates),
        GateResult(
            "backend_parity_comparison",
            not parity_failures,
            tuple(parity_failures),
        ),
        GateResult(
            "generated_code_sandbox",
            bool(generated_results) and not sandbox_failures,
            tuple(sandbox_failures)
            if generated_results
            else ("no generated code was verified",),
        ),
    )

    passed = all(gate.passed for gate in gates)
    run = RunResult(
        run_id=run_id,
        profile=profile,  # type: ignore[arg-type]
        fixture_hashes=tuple(fixture_hashes.items()),
        required_backends=tuple(backends),
        records=all_records,
        gates=gates,
        summary=(
            ("records", len(all_records)),
            ("overall_passed", passed),
            ("cleaning_records", len(cleaning_annotated)),
            ("parity_records", len(parity_annotated)),
            ("domains", len(fixtures)),
            ("repeats", repeats),
            ("minimized_failures", 0),
        ),
        environment=tuple(_environment().items()),
    )

    try:
        validate_run(run.to_dict())
    except TruthBenchSchemaError as exc:
        raise TruthBenchRunError(f"run artifact failed schema validation: {exc}") from exc

    failures_min = _minimize_failures(gates, fixtures) if not passed else ()
    run = replace(
        run,
        summary=tuple(
            (key, len(failures_min) if key == "minimized_failures" else value)
            for key, value in run.summary
        ),
    )
    notes = compare_to_baseline(run, Path(results_dir) / "baseline.json")
    artifacts: Mapping[str, Path] = {}
    if write:
        artifacts = write_artifacts(run, failures_min, results_dir=results_dir)
    return ReleaseRunOutcome(
        run=run,
        passed=passed,
        failures=failures_min,
        artifacts=artifacts,
        baseline_notes=tuple(notes),
        generated_code_results=tuple(generated_results),
    )


__all__ = [
    "ReleaseRunOutcome",
    "SECONDARY_SURFACES",
    "TruthBenchRunError",
    "parity_fixture",
    "run_release",
]
