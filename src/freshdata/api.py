"""Top-level convenience functions: ``fd.clean(df)`` and ``fd.profile(df)``."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import pandas as pd

from ._reportframe import ReportFrame
from .adapters.polars import from_pandas, to_pandas
from .cleaner import Cleaner, run_pipeline
from .config import CleanConfig, merge_options
from .domains import SEVERITY_TO_RISK, DomainOutcome, run_domain, validator_class
from .engine.context import build_contexts
from .engine.model_select import EngineMode, rank_missing_models
from .execution import run_with_engine
from .parsers.registry import get_parser
from .plan import suggest_plan
from .profile import Profile, build_profile
from .report import CleanReport
from .result import CleanResult
from .steps.columns import normalized_column_labels
from .streaming import StreamingCleaner, TimeSeriesCleanConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .execution import EngineConfig
    from .parsers import ParseResult


def _is_native_engine_source(df: object) -> bool:
    """True for non-pandas frames an execution backend handles natively.

    Arrow, Spark, and DuckDB-relation inputs are routed through the engine layer
    (with ``engine="auto"``) rather than the in-memory pandas pipeline, which
    only accepts pandas DataFrames. Polars frames are intentionally excluded:
    they keep their existing adapter round-trip (``polars in -> polars out``).
    """
    if isinstance(df, (pd.DataFrame, dict, str)) or df is None:
        return False
    for module, attrs in (
        ("pyarrow", ("Table", "RecordBatch")),
        ("pyspark.sql", ("DataFrame",)),
        ("duckdb", ("DuckDBPyRelation",)),
    ):
        try:
            mod = __import__(module, fromlist=list(attrs))
        except ImportError:
            continue
        if isinstance(df, tuple(getattr(mod, a) for a in attrs)):
            return True
    return False


def _fold_context_options(
    options: dict[str, object],
    *,
    context: str | None,
    policy: object | None,
    strict: bool,
) -> None:
    """Fold the context-policy keywords into the plain config options.

    ``context``/``policy``/``strict`` are ordinary :class:`CleanConfig` fields;
    the explicit keywords exist for discoverability and early both-passed
    errors. Defaults are not folded so an existing ``config=`` value survives.
    """
    if context is not None and policy is not None:
        raise TypeError(
            "context= and policy= are mutually exclusive: pass the raw text or "
            "a pre-compiled ContextPolicy, not both"
        )
    if context is not None:
        options["context"] = context
    if policy is not None:
        options["policy"] = policy
    if strict:
        options["strict"] = strict


def _fold_profile(
    df: object,
    profile: object,
    options: dict[str, object],
    memory: object | None,
    *,
    engine: str,
    output_format: str,
    engine_config: object | None,
) -> tuple[object, dict[str, object], object | None]:
    """Learned-profile replay (Phase 4): drift-gate, then fold learned config
    deltas into the options (user options and policy always win) and let the
    profile's value maps / embedded memory / examples propose through the
    standard semantic gates. Returns (resolved profile, options, memory).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("profile= requires an in-memory pandas DataFrame")
    if engine != "pandas" or output_format != "pandas" or engine_config is not None:
        raise TypeError("profile= is only supported on the in-memory pandas engine")
    from .learning.replay import (  # noqa: PLC0415
        check_profile_drift,
        fold_profile_options,
        resolve_profile,
    )

    resolved = resolve_profile(profile)
    gate = check_profile_drift(df, resolved)
    options = fold_profile_options(resolved, dict(options), gate)
    if gate.ok and memory is None and resolved.memory is not None:
        memory = resolved.memory
    return resolved, options, memory


def _normalize_clean_call(
    config: CleanConfig | Mapping[str, object] | None,
    options: dict[str, object],
    return_report: bool,
) -> tuple[CleanConfig | None, dict[str, object], bool]:
    """Fold compact-call conveniences into the legacy-compatible implementation."""
    if "report" in options:
        return_report = bool(options.pop("report"))
    if isinstance(config, Mapping):
        mapped_options = dict(config)
        mapped_options.update(options)
        return None, mapped_options, return_report
    return config, options, return_report


def _clean_result(cleaned: pd.DataFrame, source: object, report: CleanReport) -> object:
    converted = from_pandas(cleaned, source)
    if isinstance(converted, pd.DataFrame):
        return CleanResult.wrap(converted, report)
    return converted


def clean(
    df: pd.DataFrame,
    config: CleanConfig | Mapping[str, object] | None = None,
    *,
    return_report: bool = False,
    source_provenance: dict[str, object] | None = None,
    provenance_confidence_threshold: float = 0.7,
    contract: object | None = None,
    on_unexpected: str = "warn",
    on_missing: str = "fail",
    domain: str | None = None,
    column_map: dict[str, str] | None = None,
    gtfs_file: str | None = None,
    fhir_resource: str | None = None,
    media_type: str | None = None,
    finance_mode: str | None = None,
    audit_include_phi: bool = False,
    domain_kwargs: dict[str, object] | None = None,
    engine: str = "pandas",
    output_format: str = "pandas",
    engine_config: EngineConfig | None = None,
    memory: object | None = None,
    profile: object | None = None,
    context: str | None = None,
    policy: object | None = None,
    strict: bool = False,
    **options: object,
) -> pd.DataFrame | tuple[pd.DataFrame, CleanReport]:
    """Clean a DataFrame and return a new, repaired one.

    Two layers run in order. **Representation repair** always happens first:

    1.  ``column_names`` — snake_case column names, deduplicate collisions.
    2.  ``strip_whitespace`` — trim surrounding whitespace in text cells.
    3.  ``normalize_sentinels`` — turn "N/A", "null", "-", "" … into missing.
    4.  ``drop_empty_columns`` / ``drop_empty_rows`` — remove all-missing ones.
    5.  ``fix_dtypes`` — text that is really numeric / datetime / boolean gets
        the right dtype (validated; ``numeric_threshold`` of values must parse).
    6.  ``drop_duplicates`` — resolve duplicate rows (``duplicate_keep``
        chooses first/last/drop/aggregate; time-indexed frames are protected).

    Then, with ``strategy="auto"`` (the default), the **decision engine**
    profiles every column — missing ratio, dtype, skewness, cardinality,
    inferred role (id / target / datetime / text / categorical), whether
    missingness looks informative — and applies threshold rules for missing
    values and outliers. Nothing is done silently: every action (including
    deliberately preserving a column) is logged with a rationale, a risk
    level, and a confidence score. ``strategy="conservative"`` disables the
    engine; imputation and outlier handling are then opt-in via ``impute=`` /
    ``outliers=``.

    Parameters
    ----------
    df:
        The DataFrame to clean.
    config:
        A prebuilt :class:`~freshdata.CleanConfig` to start from.
    return_report:
        If True, return ``(cleaned_df, CleanReport)``. The report carries
        per-action rationale/risk/confidence, missing counts before/after,
        warnings, and recommendations for manual review.
    domain:
        Optional domain validator pack (e.g. ``"finance"``). When set, generic
        cleaning runs first (defaulting to ``strategy="conservative"`` so the
        statistical engine never silently alters ledgers/IDs unless you pass an
        explicit ``strategy``), then the pack validates in layers and repairs
        separately; findings and a ``domain_trust_score`` are folded into the
        report. Unknown names raise :class:`~freshdata.domains.UnknownDomainError`.
    column_map:
        Optional ``{actual_column: canonical_field}`` overrides for the domain
        pack's column detection. Requires ``domain`` to be set.
    gtfs_file:
        File selector for a single-frame feed-domain run, such as ``"stops.txt"``
        with ``domain="transport"``. Full feeds can instead be passed as a dict.
    fhir_resource:
        FHIR resource selector for ``domain="healthcare"`` (``"Patient"``,
        ``"Observation"``, ``"Encounter"``); auto-detected from columns if omitted.
    media_type:
        Sub-schema selector for ``domain="media"`` (``"content"`` / ``"release"``);
        auto-detected from columns if omitted.
    audit_include_phi:
        For PHI-aware packs (healthcare, education), include raw PHI values in the
        audit trail instead of masking them as ``[PHI]``. Defaults to False.
    domain_kwargs:
        Optional pack-specific constructor arguments. These are forwarded for
        both single-frame and feed-domain runs.
    **options:
        Any :class:`~freshdata.CleanConfig` field as a keyword override — e.g.
        ``strategy`` (``"balanced"`` default / ``"aggressive"`` / ``"conservative"``),
        ``missing_threshold_low``/``_medium``/``_high``, ``duplicate_threshold``,
        ``outlier_method``, ``outlier_action``, ``preserve_original``, ``verbose``,
        ``progress_callback``, ``preserve_columns``, ``target_column``,
        ``duplicate_keep``, ``impute``, ``outliers``. Unknown names raise
        :class:`TypeError`.

    Examples
    --------
    >>> import freshdata as fd
    >>> cleaned = fd.clean(df)
    >>> cleaned, rep = fd.clean(df, return_report=True)
    >>> print(rep.summary())

    >>> fd.clean(df, outlier_action="flag", target_column="churn",
    ...          preserve_columns=("notes",), verbose=False)

    >>> ledger = fd.clean(df, domain="finance")          # validate + repair
    >>> ledger, rep = fd.clean(df, domain="finance", return_report=True)
    >>> rep.domain_trust_score                            # 0–1

    Natural-language cleaning rules compile deterministically into a
    :class:`~freshdata.ContextPolicy` that governs the run (protected columns,
    id columns, per-column semantic hints)::

    >>> cleaned = fd.clean(df, context="CustomerID is unique. Never modify revenue.")
    >>> policy = fd.compile_context("...", df=df)         # inspect/review first
    >>> cleaned = fd.clean(df, policy=policy)             # skip parsing, use as-is
    """
    config, options, return_report = _normalize_clean_call(config, options, return_report)

    _fold_context_options(options, context=context, policy=policy, strict=strict)
    domain_kwargs = _merge_pack_selectors(
        domain_kwargs,
        domain,
        fhir_resource=fhir_resource,
        media_type=media_type,
        finance_mode=finance_mode,
        audit_include_phi=audit_include_phi,
    )
    if domain is not None:
        if isinstance(df, dict) or gtfs_file is not None:
            return _clean_feed(
                df,
                domain,
                gtfs_file,
                column_map,
                domain_kwargs,
                config,
                return_report,
                options,
            )
        if getattr(validator_class(domain), "multi_frame", False):
            raise TypeError(
                f"domain {domain!r} requires a feed dict or a single frame with gtfs_file="
            )
        return _clean_with_domain(
            df, domain, column_map, domain_kwargs, config, return_report, options
        )
    if column_map is not None:
        raise TypeError("column_map requires a domain= to be set")
    if gtfs_file is not None:
        raise TypeError("gtfs_file requires domain='transport' (or another feed domain)")

    # Contract gate (F1c): explain incoming schema drift *before* repair.
    # In-memory pandas only — keeps the gate predictable and reproducible.
    contract_diff = None
    if contract is not None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("contract= requires an in-memory pandas DataFrame")
        if engine != "pandas" or output_format != "pandas" or engine_config is not None:
            raise TypeError("contract= is only supported on the in-memory pandas engine")
        from .enterprise.contracts import ContractViolation, diff_schema as _diff_schema  # noqa: I001, PLC0415

        contract_diff = _diff_schema(
            df,
            contract=contract,  # type: ignore[arg-type]
            on_unexpected=on_unexpected,  # type: ignore[arg-type]
            on_missing=on_missing,  # type: ignore[arg-type]
        )
        if contract_diff.n_errors > 0:
            raise ContractViolation(contract_diff)

    # Cleaning-memory replay (F3a): apply previously accepted decisions when the
    # new data still matches what the memory learned; otherwise it is ignored and
    # the report explains why. In-memory pandas only, like the contract gate.
    mem_match = None
    if memory is not None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("memory= requires an in-memory pandas DataFrame")
        if engine != "pandas" or output_format != "pandas" or engine_config is not None:
            raise TypeError("memory= is only supported on the in-memory pandas engine")
        from .memory import CleaningMemory, apply_memory  # noqa: PLC0415

        if not isinstance(memory, CleaningMemory):
            raise TypeError("memory= must be a CleaningMemory (see fd.learn_cleaning_memory)")
        options, mem_match = apply_memory(df, memory, dict(options))

    if profile is not None:
        # Drift-gating, replay, and report annotation happen inside
        # Cleaner.clean; here we only fold learned config deltas into the
        # options and adopt the embedded memory when none was supplied.
        profile, options, memory = _fold_profile(
            df,
            profile,
            options,
            memory,
            engine=engine,
            output_format=output_format,
            engine_config=engine_config,
        )

    native_source = _is_native_engine_source(df)
    if (
        engine != "pandas"
        or output_format != "pandas"
        or engine_config is not None
        or isinstance(df, str)
        or native_source
    ):
        return _clean_out_of_core(
            df,
            config,
            options,
            engine="auto" if native_source and engine == "pandas" else engine,
            output_format=output_format,
            engine_config=engine_config,
            return_report=return_report,
        )

    if source_provenance is not None and not return_report:
        raise ValueError("source_provenance requires return_report=True")

    cleaner = Cleaner(config=config, **options)
    cleaned, rep = cleaner.clean(df, report=True, memory=memory, profile=profile)
    if memory is not None and mem_match is not None:
        from .memory import CleaningMemory, annotate_report  # noqa: PLC0415

        annotate_report(rep, cast(CleaningMemory, memory), mem_match)
    if source_provenance is not None:
        from .provenance import (  # noqa: PLC0415
            annotate_provenance,
        )

        annotate_provenance(
            rep,
            source_provenance,
            confidence_threshold=provenance_confidence_threshold,
        )
    if contract_diff is not None:
        rep.contract_violations = contract_diff.to_dict()
    if return_report:
        return _clean_result(cleaned, df, rep), rep
    return _clean_result(cleaned, df, rep)


def _clean_out_of_core(
    df: Any,
    config: CleanConfig | None,
    options: dict[str, object],
    *,
    engine: str,
    output_format: str,
    engine_config: EngineConfig | None,
    return_report: bool,
) -> Any:
    """Out-of-core / Arrow-native path: run the clean on Polars, DuckDB, or
    Spark, or read a file path. Default callers passing an in-memory pandas
    frame (engine="pandas", output_format="pandas") never reach here, so the
    existing in-memory behaviour is unchanged. A non-pandas frame (polars /
    Arrow / Spark) with default options is routed to engine="auto"."""
    cfg = merge_options(config, **options)
    if cfg.context is not None or cfg.policy is not None:
        raise TypeError(
            "context=/policy= are only supported on the in-memory pandas "
            "engine (compile the policy with fd.compile_context and lower "
            "it into a config yourself for out-of-core runs)"
        )
    return run_with_engine(
        df,
        cfg,
        engine=engine,
        output_format=output_format,
        engine_config=engine_config,
        return_report=return_report,
    )


def clean_csv(
    path: str | Path,
    config: CleanConfig | Mapping[str, object] | None = None,
    *,
    output_path: str | Path | None = None,
    return_report: bool = False,
    read_csv_kwargs: dict[str, object] | None = None,
    to_csv_kwargs: dict[str, object] | None = None,
    context: str | None = None,
    policy: object | None = None,
    strict: bool = False,
    profile: object | None = None,
    **options: object,
) -> pd.DataFrame | tuple[pd.DataFrame, CleanReport]:
    """Read a CSV file, clean it, and optionally write the result to disk.

    Parameters
    ----------
    path:
        Path to the input CSV file.
    output_path:
        Optional path to write the cleaned CSV.
    return_report:
        If True, return ``(cleaned_df, CleanReport)``.
    read_csv_kwargs:
        Optional keyword arguments forwarded to ``pandas.read_csv``.
    to_csv_kwargs:
        Optional keyword arguments forwarded to ``DataFrame.to_csv``.
        ``index`` defaults to False unless explicitly overridden.
    context / policy / strict:
        Natural-language rules or a pre-compiled
        :class:`~freshdata.ContextPolicy`, forwarded to :func:`freshdata.clean`.
    profile:
        A learned :class:`~freshdata.learning.LearningProfile` (or path to a
        ``.fdprofile``), forwarded to :func:`freshdata.clean`.
    **options:
        Any :class:`~freshdata.CleanConfig` field accepted by
        :func:`freshdata.clean`.

    Examples
    --------
    >>> import freshdata as fd
    >>> cleaned = fd.clean_csv("input.csv")
    >>> fd.clean_csv("input.csv", output_path="cleaned.csv")
    >>> cleaned, report = fd.clean_csv("input.csv", return_report=True)
    >>> fd.clean_csv("input.csv", context="Emails must be valid.")
    """
    if "report" in options:
        return_report = bool(options.pop("report"))
    df = pd.read_csv(path, **(read_csv_kwargs or {}))
    result = clean(
        df,
        config=config,
        return_report=return_report,
        context=context,
        policy=policy,
        strict=strict,
        profile=profile,
        **options,  # type: ignore[arg-type]
    )
    cleaned_df = cast(pd.DataFrame, result[0] if return_report else result)
    if output_path is not None:
        cleaned_df.to_csv(output_path, **{"index": False, **(to_csv_kwargs or {})})
    return result


def compile_context(
    text: str,
    df: pd.DataFrame | None = None,
    *,
    columns: Sequence[str] | None = None,
    config: CleanConfig | None = None,
    strict: bool = False,
) -> Any:
    """Compile natural-language cleaning rules into a :class:`~freshdata.ContextPolicy`.

    Deterministic, model-free, and offline: the same text always compiles to the
    same policy. With a frame (or explicit ``columns``) column phrases are
    resolved against the effective post-normalization schema; without one the
    policy compiles schema-free and resolves when it meets a frame. The result
    is inspectable (``policy.summary()``), reviewable (``policy.to_json()``),
    and reusable (``fd.clean(df, policy=policy)``).

    Examples
    --------
    >>> import freshdata as fd
    >>> policy = fd.compile_context(
    ...     "CustomerID is unique. Never modify revenue values.", df=df)
    >>> print(policy.summary())
    >>> policy.to_json("policy.json")
    >>> cleaned = fd.clean(df, policy=policy)
    """
    from .context import compile_context as _compile  # noqa: PLC0415

    return _compile(text, df=df, columns=columns, config=config, strict=strict)


def validate(
    df: pd.DataFrame,
    *,
    context: str | None = None,
    policy: object | None = None,
    config: CleanConfig | None = None,
    strict: bool = False,
    **options: object,
) -> Any:
    """Check *df* against a context policy without mutating anything.

    Returns a :class:`~freshdata.FindingList` (a plain ``list`` of
    :class:`~freshdata.QualityFinding` with ``.errors`` / ``.warnings``
    shortcuts) covering unresolved references, compile issues, protected
    columns, and unique / allowed-values / range violations.

    Examples
    --------
    >>> findings = fd.validate(df, context="CustomerID is unique.")
    >>> assert not findings.errors
    """
    _fold_context_options(options, context=context, policy=policy, strict=strict)
    cfg = merge_options(config, **options)
    if cfg.context is None and cfg.policy is None:
        raise TypeError("fd.validate needs context= (rules text) or policy= (a ContextPolicy)")
    from .context.validate import validate_frame  # noqa: PLC0415

    return validate_frame(to_pandas(df), cfg)


def apply_plan(
    df: pd.DataFrame,
    plan: object,
    *,
    keep_undo: bool = False,
    allow_drift: bool = False,
    undo_cell_limit: int = 100_000,
) -> tuple[pd.DataFrame, CleanReport]:
    """Execute exactly the approved actions of a repair plan against *df*.

    The plan is a :class:`~freshdata.RepairPlan` (or a
    :class:`~freshdata.CleanPlan` from :func:`freshdata.suggest_plan`, whose
    ``repair_plan`` is used). Nothing is re-profiled and nothing is
    re-decided: approved actions run, rejected and blocked actions do not,
    pending actions are recorded as suggestions. Before returning, every
    protected column (context-protected, ``preserve_columns``,
    ``target_column``, id columns) is verified byte-identical — a violation
    raises :class:`~freshdata.ProtectedColumnError` and your input frame is
    left untouched.

    If *df* changed since the plan was suggested, :class:`~freshdata.PlanDriftError`
    is raised; pass ``allow_drift=True`` to apply anyway (stale actions are
    skipped and recorded). With ``keep_undo=True`` the report keeps a compact
    undo log (capped at ``undo_cell_limit`` cells) and cell-scoped actions can
    be reverted:

    >>> plan = fd.suggest_plan(df, context=rules, semantic_mode="auto").repair_plan
    >>> plan.approve_all(max_risk="low")
    >>> cleaned, report = fd.apply_plan(df, plan, keep_undo=True)
    >>> report.decisions_hash  # stable audit digest
    >>> restored = report.revert(cleaned, action_ids=[plan.actions[0].id])
    """
    from .repairplan import RepairPlan, execute_plan  # noqa: PLC0415 — lazy import

    repair_plan = getattr(plan, "repair_plan", plan)
    if repair_plan is None:
        raise TypeError(
            "this CleanPlan has no repair_plan — call suggest_plan with a "
            "context=/policy= or semantic_mode= so planned actions exist"
        )
    if not isinstance(repair_plan, RepairPlan):
        raise TypeError(f"plan must be a RepairPlan or CleanPlan, got {type(plan).__name__}")
    return execute_plan(
        to_pandas(df),
        repair_plan,
        keep_undo=keep_undo,
        allow_drift=allow_drift,
        undo_cell_limit=undo_cell_limit,
    )


def clean_timeseries(
    df: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    time_series_config: TimeSeriesCleanConfig | None = None,
    config: CleanConfig | None = None,
    return_report: bool = False,
    return_exceptions: bool = False,
    **options: object,
) -> Any:
    """Clean an ordered, timestamped frame with time-series-aware policies.

    A convenience wrapper that runs the data through a single-batch
    :class:`~freshdata.StreamingCleaner` configured with a
    :class:`~freshdata.TimeSeriesCleanConfig`: representation repair, then short-gap
    interpolation, optional seasonal imputation, ordered dedupe, watermark-based
    late-data handling, and windowed anomaly detection — every step audited in the
    returned :class:`~freshdata.CleanReport`. Generic statistical imputation is skipped
    for the numeric series columns so the time-series policy owns their gaps.

    Either pass a ready ``time_series_config`` or a ``timestamp_column`` plus any
    :class:`~freshdata.TimeSeriesCleanConfig` field as a keyword (e.g.
    ``entity_id_columns=("sensor_id",)``, ``max_interpolation_gap=2``). Remaining keyword
    options are forwarded to the cleaner (``CleanConfig`` fields, ``strategy``, etc.).

    Returns the cleaned frame; add ``return_report`` and/or ``return_exceptions`` to also
    get the report and the quarantined-rows frame (``(cleaned, report)``,
    ``(cleaned, exceptions)``, or ``(cleaned, report, exceptions)``).

    Examples
    --------
    >>> import pandas as pd
    >>> import freshdata as fd
    >>> df = pd.DataFrame({
    ...     "t": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00",
    ...                          "2024-01-01 02:00", "2024-01-01 03:00"]),
    ...     "v": [1.0, None, 3.0, 4.0],
    ... })
    >>> out = fd.clean_timeseries(df, timestamp_column="t", max_interpolation_gap=1)
    >>> out["v"].tolist()
    [1.0, 2.0, 3.0, 4.0]
    """
    if time_series_config is None:
        if timestamp_column is None:
            raise ValueError("clean_timeseries needs timestamp_column or time_series_config")
        ts_field_names = {f.name for f in dataclasses.fields(TimeSeriesCleanConfig)}
        ts_kwargs = {k: options.pop(k) for k in list(options) if k in ts_field_names}
        time_series_config = TimeSeriesCleanConfig(
            timestamp_column=timestamp_column,
            **ts_kwargs,  # type: ignore[arg-type]
        )

    cleaner = StreamingCleaner(
        config=config,
        time_series_config=time_series_config,
        warmup_batches=0,
        **options,  # type: ignore[arg-type]
    )
    cleaned, report = cleaner.clean_batch(df)
    exceptions = cleaner.exceptions_

    if return_report and return_exceptions:
        return cleaned, report, exceptions
    if return_report:
        return cleaned, report
    if return_exceptions:
        return cleaned, exceptions
    return cleaned


def _merge_pack_selectors(
    domain_kwargs: dict[str, object] | None,
    domain: str | None,
    *,
    fhir_resource: str | None,
    media_type: str | None,
    finance_mode: str | None,
    audit_include_phi: bool,
) -> dict[str, object] | None:
    """Fold pack selectors into ``domain_kwargs`` forwarded to the validator constructor.

    ``fhir_resource`` (healthcare), ``media_type`` (media), ``finance_mode`` (finance),
    and ``audit_include_phi`` (healthcare/education) are promoted to top-level ``clean``
    kwargs for ergonomics, mirroring ``gtfs_file``. Each requires ``domain=`` to be set.
    """
    selectors: dict[str, object] = {}
    if fhir_resource is not None:
        selectors["fhir_resource"] = fhir_resource
    if media_type is not None:
        selectors["media_type"] = media_type
    if finance_mode is not None:
        selectors["finance_mode"] = finance_mode
    if audit_include_phi:
        selectors["audit_include_phi"] = True
    if not selectors:
        return domain_kwargs
    if domain is None:
        raise TypeError(
            "fhir_resource=, media_type=, finance_mode=, and audit_include_phi= "
            "require a domain= to be set"
        )
    if finance_mode is not None and domain != "finance":
        raise TypeError(f"finance_mode= requires domain='finance', got domain={domain!r}")
    return {**(domain_kwargs or {}), **selectors}


def _clean_with_domain(
    df: pd.DataFrame,
    domain: str,
    column_map: dict[str, str] | None,
    domain_kwargs: dict[str, object] | None,
    config: CleanConfig | None,
    return_report: bool,
    options: dict[str, object],
) -> pd.DataFrame | tuple[pd.DataFrame, CleanReport]:
    """Generic clean (conservative by default) then domain validate + repair."""
    # With an explicit config the caller owns every setting. Otherwise default to
    # a conservative base that does *not* infer dtypes: the domain pack owns
    # format validation/coercion (per its audited rules), and generic dtype
    # inference would otherwise silently retype dates/amounts before validation.
    if config is None:
        options = {
            "strategy": "conservative",
            "fix_dtypes": False,
            **options,  # explicit caller options win
        }
    cfg = merge_options(config, **options)
    cleaned, rep = run_pipeline(df, cfg)
    effective_map = _normalized_column_map(df, cfg, column_map)
    repaired, outcome = run_domain(
        cleaned, domain, column_map=effective_map, **(domain_kwargs or {})
    )
    _fold_domain_outcome(rep, outcome)
    if cfg.verbose:
        print(rep.brief())
    out = from_pandas(repaired, df)
    return (out, rep) if return_report else out


def _fold_domain_outcome(rep: CleanReport, outcome: DomainOutcome) -> None:
    """Merge a domain run's findings/repairs into the existing CleanReport."""
    report = outcome.report
    rep.domain = outcome.domain
    rep.domain_trust_score = outcome.trust_score
    rep.domain_findings = [r.to_dict() for r in report.results]
    rep.domain_repairs = [a.to_dict() for a in outcome.repairs.actions]
    for result in report.results:
        if not result.violated:
            continue
        col = report.mapping.actual(result.fields[0]) if result.fields else None
        rep.add(
            step=f"domain:{outcome.domain}:{result.rule_id}",
            description=result.message or result.name,
            column=col,
            count=result.n_violations,
            risk=SEVERITY_TO_RISK.get(result.severity, "low"),
            rationale=result.name,
        )
        if result.severity == "error":
            rep.add_warning(
                f"[{outcome.domain}] {result.rule_id}: {result.message or result.name}"
            )
    applied = sum(1 for a in outcome.repairs.actions if a.status == "applied")
    if applied:
        rep.add_recommendation(
            f"{outcome.domain}: {applied} domain repair(s) applied — see domain_repairs"
        )


def _clean_feed(
    data: Any,
    domain: str,
    gtfs_file: str | None,
    column_map: dict[str, str] | None,
    domain_kwargs: dict[str, object] | None,
    config: CleanConfig | None,
    return_report: bool,
    options: dict[str, object],
) -> Any:
    """Validate + repair a multi-frame feed (e.g. GTFS), one file at a time.

    Accepts either a dict of ``{file: frame}`` (full feed) or a single frame plus
    ``gtfs_file`` (one file). Each frame is conservatively cleaned, then validated
    and repaired with the other frames available as cross-file context. Returns the
    same shape it was given (frame in → frame out; dict in → dict out).
    """
    cls = validator_class(domain)  # raises UnknownDomainError for unknown names
    if not getattr(cls, "multi_frame", False):
        raise TypeError(f"domain {domain!r} does not accept feed input (a dict or gtfs_file)")
    if isinstance(data, dict):
        frames = dict(data)
        single: str | None = None
    else:
        if gtfs_file is None:
            raise TypeError("a single frame for a feed domain requires gtfs_file=")
        frames = {gtfs_file: data}
        single = gtfs_file

    base = (
        {"strategy": "conservative", "fix_dtypes": False, **options} if config is None else options
    )
    cfg = merge_options(config, **base)
    cleaned = {name: run_pipeline(frame, cfg)[0] for name, frame in frames.items()}
    effective_maps = {
        name: _normalized_column_map(frames[name], cfg, column_map) for name in frames
    }

    repaired: dict[str, pd.DataFrame] = {}
    outcomes: dict[str, DomainOutcome] = {}
    unvalidated: list[str] = []
    for name, frame in cleaned.items():
        supports_file = getattr(cls, "supports_file", None)
        if single is None and callable(supports_file) and not supports_file(name):
            repaired[name] = frame
            unvalidated.append(str(name))
            continue
        kwargs = dict(domain_kwargs or {})
        kwargs.update({"gtfs_file": name, "feed": cleaned})
        rep_df, outcome = run_domain(frame, domain, column_map=effective_maps[name], **kwargs)
        repaired[name] = rep_df
        outcomes[name] = outcome

    report = CleanReport(
        rows_before=sum(len(f) for f in frames.values()),
        rows_after=sum(len(f) for f in repaired.values()),
    )
    _fold_feed_outcomes(report, domain, outcomes)
    for name in unvalidated:
        report.add_warning(f"[{domain}:{name}] file is not covered by this domain pack")
    if cfg.verbose:
        print(report.brief())

    if single is not None:
        out = from_pandas(repaired[single], frames[single])
        return (out, report) if return_report else out
    result = {name: from_pandas(repaired[name], frames[name]) for name in frames}
    return (result, report) if return_report else result


def _normalized_column_map(
    original: Any,
    cfg: CleanConfig,
    column_map: dict[str, str] | None,
) -> dict[str, str] | None:
    """Translate overrides from input labels to labels seen by a domain pack."""
    if column_map is None or not cfg.column_names:
        return column_map
    original_columns = list(to_pandas(original).columns)
    normalized_columns = normalized_column_labels(original_columns)
    translated: dict[str, str] = {}
    for actual, canonical in column_map.items():
        try:
            position = original_columns.index(actual)
        except ValueError:
            translated[actual] = canonical
        else:
            translated[str(normalized_columns[position])] = canonical
    return translated


def _fold_feed_outcomes(rep: CleanReport, domain: str, outcomes: dict[str, DomainOutcome]) -> None:
    """Merge per-file domain outcomes into one CleanReport (findings tagged by file)."""
    rep.domain = domain
    findings: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    scores: list[float] = []
    for file, outcome in outcomes.items():
        scores.append(outcome.trust_score)
        mapping = outcome.report.mapping
        for result in outcome.report.results:
            entry = result.to_dict()
            entry["file"] = file
            findings.append(entry)
            if not result.violated:
                continue
            col = mapping.actual(result.fields[0]) if result.fields else None
            rep.add(
                step=f"domain:{domain}:{file}:{result.rule_id}",
                description=result.message or result.name,
                column=col,
                count=result.n_violations,
                risk=SEVERITY_TO_RISK.get(result.severity, "low"),
                rationale=f"{file}: {result.name}",
            )
            if result.severity == "error":
                rep.add_warning(
                    f"[{domain}:{file}] {result.rule_id}: {result.message or result.name}"
                )
        for action in outcome.repairs.actions:
            entry = action.to_dict()
            entry["file"] = file
            repairs.append(entry)
    rep.domain_findings = findings
    rep.domain_repairs = repairs
    rep.domain_trust_score = round(sum(scores) / len(scores), 4) if scores else 1.0
    applied = sum(1 for a in repairs if a["status"] == "applied")
    if applied:
        rep.add_recommendation(
            f"{domain}: {applied} domain repair(s) applied — see domain_repairs"
        )


def _engine_mode(cfg: CleanConfig) -> EngineMode:
    mode = cfg.engine_mode or "balanced"
    return "balanced" if mode == "balanced" else "aggressive"


def infer_roles(
    df: pd.DataFrame,
    *,
    strategy: str = "balanced",
    config: CleanConfig | None = None,
    **options: object,
) -> pd.DataFrame:
    """Infer column roles and primary missing models without mutating data.

    Also reports a per-column ``semantic_type`` (email/phone/url/... —
    see :data:`freshdata.semantic.semantic_types.SEMANTIC_TYPES`) with a
    confidence and a compact evidence string. Semantic-type inference is
    deterministic and model-free; an explicit ``semantic_context`` hint always
    wins. These three columns are additive — the original output is unchanged.
    """
    from .semantic.semantic_types import infer_semantic_type  # noqa: PLC0415 — lazy

    cfg = merge_options(config, strategy=strategy, **options)
    frame = to_pandas(df)
    contexts = build_contexts(frame, cfg)
    mode = _engine_mode(cfg)
    hints = cfg.semantic_context if isinstance(cfg.semantic_context, dict) else {}
    column_hints = hints.get("columns", {}) if isinstance(hints.get("columns"), dict) else {}
    rows = []
    for col, ctx in sorted(contexts.items()):
        primary = None
        if ctx.missing_ratio > 0:
            primary = rank_missing_models(frame, col, ctx, cfg, mode=mode).primary
        hint = None
        col_hint = column_hints.get(col)
        if isinstance(col_hint, dict):
            hint = col_hint.get("semantic_type")
        inferred = infer_semantic_type(
            col,
            frame[col],
            role=ctx.role,
            hint=str(hint) if hint else None,
            sample_size=cfg.semantic_sample_size,
        )
        rows.append(
            {
                "column": col,
                "role": ctx.role,
                "missing_pct": round(ctx.missing_ratio * 100, 2),
                "cardinality": ctx.nunique,
                "skew": ctx.skew,
                "domain_sensitive": ctx.domain_sensitive,
                "primary_missing_model": primary.model_id if primary else None,
                "semantic_type": inferred.semantic_type,
                "semantic_type_confidence": round(inferred.confidence, 4),
                "semantic_type_evidence": "; ".join(
                    f"{e.kind}: {e.detail}" for e in inferred.evidence
                ),
            }
        )
    return ReportFrame.wrap(pd.DataFrame(rows), "infer_roles")


def profile(
    df: pd.DataFrame,
    *,
    config: CleanConfig | None = None,
    include_plan: bool = False,
    lazy_report: bool = False,
    max_columns: int | None = None,
    profile_sample: int | None = None,
    **options: object,
) -> Profile:
    """Inspect a DataFrame without changing it.

    Returns a :class:`~freshdata.Profile` describing shape, memory, missing
    data, duplicates, and per-column issues — including a faithful preview of
    the dtype conversions :func:`clean` would perform, computed by the same
    inference code.

    With ``include_plan=True``, attaches a :class:`~freshdata.CleanPlan` at
    ``profile.plan`` previewing engine model choices.

    Wide-schema / large-frame perf controls: ``profile_sample=N`` profiles a
    deterministic N-row sample (stats become estimates), ``max_columns=M`` caps
    profiling to the first M columns, and ``lazy_report=True`` skips the
    expensive full-frame duplicate-row scan. When any is used the profile
    describes the profiled *subset* and records totals at
    ``profile.materialization`` (also in ``.to_dict()``).

    Examples
    --------
    >>> import freshdata as fd
    >>> p = fd.profile(df)
    >>> print(p)             # human-readable issue table
    >>> p.to_frame()         # one row per column, sortable in a notebook
    >>> p.to_dict()          # JSON-friendly
    >>> p = fd.profile(wide_df, profile_sample=10_000, max_columns=200, lazy_report=True)
    >>> print(p.materialization)
    """
    if "plan" in options:
        include_plan = bool(options.pop("plan"))
    if "sample" in options:
        profile_sample = cast(Optional[int], options.pop("sample"))
    if "lazy" in options:
        lazy_report = bool(options.pop("lazy"))
    cfg = merge_options(config, **options)
    prof = build_profile(
        to_pandas(df),
        cfg,
        sample=profile_sample,
        max_columns=max_columns,
        lazy=lazy_report,
    )
    if include_plan:
        object.__setattr__(prof, "plan", suggest_plan(to_pandas(df), config=cfg))
    return prof


def parse_domain(source: Any, *, format: str) -> ParseResult:  # noqa: A002
    """Parse a raw message or file into DataFrames using the named *format* parser.

    *source* may be raw text/bytes content, a file-like object, or a
    :class:`pathlib.Path` for filesystem input. String values are treated as content;
    use :func:`clean_domain_file` for the convenience file-path workflow. *format* is a
    registered parser name (``"hl7v2"``, ``"gpx"``, ``"sdmx"``, ``"edifact"`` — see
    :func:`freshdata.parsers.available`).

    Returns a :class:`~freshdata.parsers.ParseResult` carrying the parsed frames,
    a suggested domain, metadata, and any audit warnings.

    Examples
    --------
    >>> import freshdata as fd
    >>> result = fd.parse_domain(hl7_text, format="hl7v2")   # doctest: +SKIP
    >>> result.frames["observation"]                          # doctest: +SKIP
    """
    return get_parser(format).parse(source)


def clean_domain_file(
    path: Any,
    *,
    format: str,  # noqa: A002
    domain: str | None = None,
    frame: str | None = None,
    return_report: bool = False,
    **clean_kwargs: Any,
) -> Any:
    """Parse *path* with the *format* parser, then optionally clean a frame by *domain*.

    With no *domain* (and none suggested by the parser), the :class:`ParseResult` is
    returned as-is. Otherwise the chosen *frame* — or the sole non-empty frame — is run
    through :func:`clean` with *domain* and any extra cleaning keyword arguments. When a
    parser yields several non-empty frames, pass ``frame=`` to pick one.
    """
    source = Path(path) if isinstance(path, str) and Path(path).exists() else path
    result = parse_domain(source, format=format)
    if domain is None:
        # suggested_domain is advisory metadata, not an instruction to clean.
        return result

    non_empty = {name: df for name, df in result.frames.items() if not df.empty}
    if frame is not None:
        target = frame
    elif len(non_empty) == 1:
        target = next(iter(non_empty))
    elif not non_empty:
        return result
    else:
        raise ValueError(
            f"{format} produced multiple non-empty frames {sorted(non_empty)}; "
            "pass frame=<name> to choose which to clean"
        )
    return clean(result.frames[target], domain=domain, return_report=return_report, **clean_kwargs)
