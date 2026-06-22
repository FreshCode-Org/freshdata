"""Top-level convenience functions: ``fd.clean(df)`` and ``fd.profile(df)``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from .adapters.polars import from_pandas, to_pandas
from .cleaner import Cleaner, run_pipeline
from .config import CleanConfig, merge_options
from .domains import SEVERITY_TO_RISK, DomainOutcome, run_domain, validator_class
from .engine.context import build_contexts
from .engine.model_select import EngineMode, rank_missing_models
from .plan import suggest_plan
from .profile import Profile, build_profile
from .report import CleanReport
from .steps.columns import normalized_column_labels


def clean(
    df: pd.DataFrame,
    *,
    config: CleanConfig | None = None,
    return_report: bool = False,
    domain: str | None = None,
    column_map: dict[str, str] | None = None,
    gtfs_file: str | None = None,
    fhir_resource: str | None = None,
    media_type: str | None = None,
    audit_include_phi: bool = False,
    domain_kwargs: dict[str, object] | None = None,
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
        ``preserve_columns``, ``target_column``, ``duplicate_keep``, ``impute``,
        ``outliers``. Unknown names raise :class:`TypeError`.

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
    """
    domain_kwargs = _merge_pack_selectors(
        domain_kwargs,
        domain,
        fhir_resource=fhir_resource,
        media_type=media_type,
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
    cleaner = Cleaner(config=config, **options)
    result = cleaner.clean(df, report=return_report)
    if return_report:
        cleaned, rep = result
        return from_pandas(cleaned, df), rep
    return from_pandas(result, df)


def clean_csv(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    return_report: bool = False,
    read_csv_kwargs: dict[str, object] | None = None,
    to_csv_kwargs: dict[str, object] | None = None,
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
    **options:
        Any :class:`~freshdata.CleanConfig` field accepted by
        :func:`freshdata.clean`.

    Examples
    --------
    >>> import freshdata as fd
    >>> cleaned = fd.clean_csv("input.csv")
    >>> fd.clean_csv("input.csv", output_path="cleaned.csv")
    >>> cleaned, report = fd.clean_csv("input.csv", return_report=True)
    """
    df = pd.read_csv(path, **(read_csv_kwargs or {}))
    result = clean(
        df,
        return_report=return_report,
        **options,  # type: ignore[arg-type]
    )
    cleaned_df = cast(pd.DataFrame, result[0] if return_report else result)
    if output_path is not None:
        cleaned_df.to_csv(output_path, **{"index": False, **(to_csv_kwargs or {})})
    return result


def _merge_pack_selectors(
    domain_kwargs: dict[str, object] | None,
    domain: str | None,
    *,
    fhir_resource: str | None,
    media_type: str | None,
    audit_include_phi: bool,
) -> dict[str, object] | None:
    """Fold pack selectors into ``domain_kwargs`` forwarded to the validator constructor.

    ``fhir_resource`` (healthcare), ``media_type`` (media), and ``audit_include_phi``
    (healthcare/education) are promoted to top-level ``clean`` kwargs for ergonomics,
    mirroring ``gtfs_file``. Each requires ``domain=`` to be set.
    """
    selectors: dict[str, object] = {}
    if fhir_resource is not None:
        selectors["fhir_resource"] = fhir_resource
    if media_type is not None:
        selectors["media_type"] = media_type
    if audit_include_phi:
        selectors["audit_include_phi"] = True
    if not selectors:
        return domain_kwargs
    if domain is None:
        raise TypeError(
            "fhir_resource=, media_type=, and audit_include_phi= require a domain= to be set"
        )
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
        {"strategy": "conservative", "fix_dtypes": False, **options}
        if config is None else options
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
        rep_df, outcome = run_domain(
            frame, domain, column_map=effective_maps[name], **kwargs
        )
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


def _fold_feed_outcomes(
    rep: CleanReport, domain: str, outcomes: dict[str, DomainOutcome]
) -> None:
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
    """Infer column roles and primary missing models without mutating data."""
    cfg = merge_options(config, strategy=strategy, **options)
    frame = to_pandas(df)
    contexts = build_contexts(frame, cfg)
    mode = _engine_mode(cfg)
    rows = []
    for col, ctx in sorted(contexts.items()):
        primary = None
        if ctx.missing_ratio > 0:
            primary = rank_missing_models(frame, col, ctx, cfg, mode=mode).primary
        rows.append({
            "column": col,
            "role": ctx.role,
            "missing_pct": round(ctx.missing_ratio * 100, 2),
            "cardinality": ctx.nunique,
            "skew": ctx.skew,
            "domain_sensitive": ctx.domain_sensitive,
            "primary_missing_model": primary.model_id if primary else None,
        })
    return pd.DataFrame(rows)


def profile(
    df: pd.DataFrame,
    *,
    config: CleanConfig | None = None,
    include_plan: bool = False,
    **options: object,
) -> Profile:
    """Inspect a DataFrame without changing it.

    Returns a :class:`~freshdata.Profile` describing shape, memory, missing
    data, duplicates, and per-column issues — including a faithful preview of
    the dtype conversions :func:`clean` would perform, computed by the same
    inference code.

    With ``include_plan=True``, attaches a :class:`~freshdata.CleanPlan` at
    ``profile.plan`` previewing engine model choices.

    Examples
    --------
    >>> import freshdata as fd
    >>> p = fd.profile(df)
    >>> print(p)             # human-readable issue table
    >>> p.to_frame()         # one row per column, sortable in a notebook
    >>> p.to_dict()          # JSON-friendly
    """
    cfg = merge_options(config, **options)
    prof = build_profile(to_pandas(df), cfg)
    if include_plan:
        object.__setattr__(prof, "plan", suggest_plan(to_pandas(df), config=cfg))
    return prof
