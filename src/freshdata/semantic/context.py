"""Build the :class:`SemanticContext` from config, data shape, and user hints.

This is the only place that decides "what does this column *mean*?" — the
experts then trust the resulting :class:`SemanticColumnInfo` flags. It reuses the
engine's role inference (:func:`freshdata.engine.context.build_contexts`) and the
value parsers from :mod:`freshdata.semantic.experts`, importing nothing heavier
than pandas.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import pandas as pd

from ..config import CleanConfig
from ..engine.context import build_contexts
from .experts import (
    column_name_is_identifier,
    is_plain_number,
    looks_like_date_value,
    parse_boolean,
    parse_currency,
    parse_number_words,
    parse_unit,
)
from .types import SemanticColumnInfo, SemanticContext

_MONEY_NAME = re.compile(
    r"price|amount|cost|salary|revenue|fee|balance|payment|charge|wage|income|"
    r"spend|budget|paid|total|usd|eur|inr|gbp|cash|fare",
    re.I,
)
_DATE_NAME = re.compile(
    r"date|time|timestamp|created_at|updated_at|dob|birth|joined|signup|registered",
    re.I,
)
_EMAIL_NAME = re.compile(r"e[-_]?mail|email", re.I)
_EMAIL_VALUE = re.compile(r"^[^@\s]+@[^@\s]+$")


def _share(values: Sequence[object], predicate) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if predicate(v)) / len(values)


def _column_hints(
    semantic_context: object,
) -> tuple[str | None, dict[str, dict], str | None]:
    """Extract ``(dataset, {column: hint_dict}, reference_date)`` from user context."""
    if not isinstance(semantic_context, Mapping):
        return None, {}, None
    dataset = semantic_context.get("dataset")
    reference_date = semantic_context.get("reference_date")
    columns = semantic_context.get("columns", {})
    if not isinstance(columns, Mapping):
        columns = {}
    hints = {str(k): dict(v) for k, v in columns.items() if isinstance(v, Mapping)}
    return (
        (str(dataset) if dataset is not None else None),
        hints,
        (str(reference_date) if reference_date is not None else None),
    )


def _build_info(
    df: pd.DataFrame,
    col: object,
    ctx,
    config: CleanConfig,
    hint: dict,
    reference_date: str | None,
    n_nonnull_override: int | None = None,
) -> SemanticColumnInfo:
    name = str(col)
    series = df[col]
    # On the native distinct path *series* holds only distinct values, so the
    # true non-null count is supplied by the caller (drives the count/n_nonnull
    # "share" evidence in scoring); otherwise measure it directly.
    n_nonnull = n_nonnull_override if n_nonnull_override is not None else int(series.notna().sum())

    # Cheap value-shape stats on a bounded sample of distinct string values.
    sample = series.dropna()
    if len(sample) > config.semantic_sample_size:
        sample = sample.sample(n=config.semantic_sample_size, random_state=config.random_state)
    try:
        distinct = list(pd.unique(sample))
    except TypeError:
        distinct = []
    strings = [v for v in distinct if isinstance(v, str)]

    numeric_share = _share(distinct, is_plain_number)
    word_share = _share(strings, lambda v: parse_number_words(v) is not None)
    bool_share = _share(distinct, lambda v: parse_boolean(str(v)) is not None)
    currency_share = _share(strings, lambda v: parse_currency(v) is not None)
    units = [parse_unit(v) for v in strings]
    parsed_units = [u for u in units if u is not None]
    unit_share = (len(parsed_units) / len(strings)) if strings else 0.0
    dominant_unit: str | None = None
    if parsed_units:
        counts: dict[str, int] = {}
        for _, unit in parsed_units:
            counts[unit] = counts.get(unit, 0) + 1
        dominant_unit = max(counts, key=lambda k: counts[k])

    semantic_type = hint.get("semantic_type")
    unit_hint = hint.get("unit")
    allowed = tuple(hint.get("allowed_values", ()) or ())
    mutable = hint.get("mutable")
    region = hint.get("region")
    dayfirst = hint.get("dayfirst")
    if not isinstance(dayfirst, bool) and isinstance(config.dayfirst, bool):
        # No per-column override: fall back to the existing top-level
        # day/month-order setting used elsewhere for the same ambiguity.
        dayfirst = config.dayfirst

    free_text = ctx.role == "text"
    nunique = ctx.nunique

    # Phase-2 experts: email columns are recognised from an explicit hint or
    # from an email-suggesting name whose values are dominated by addresses;
    # phone columns *only* from an explicit hint (never guessed from data).
    email_share = _share(strings, lambda v: bool(_EMAIL_VALUE.match(v.strip())))
    email_like = semantic_type == "email" or (
        semantic_type is None and bool(_EMAIL_NAME.search(name)) and email_share >= 0.6
    )
    phone_like = semantic_type == "phone"

    # ``mutable=True`` in the user context overrides identifier protection.
    # Email/phone columns are naturally near-unique, so the role/cardinality
    # identifier heuristics (and the "phone" name token) must not veto them —
    # the user's email/phone semantics are authoritative. Hard identifier
    # signals (an explicit hint, membership in id_columns) still win.
    identifier_like = mutable is not True and (
        semantic_type == "identifier"
        or name in config.id_columns
        or (
            not (email_like or phone_like)
            and (
                ctx.role == "id"
                or column_name_is_identifier(name)
                or (ctx.high_cardinality and ctx.role in ("text", "categorical"))
            )
        )
    )

    numeric_like = (
        not free_text
        and not identifier_like
        and (
            semantic_type == "number"
            or ctx.role == "numeric"
            or numeric_share >= 0.5
            or word_share >= 0.6
        )
    )
    boolean_like = (
        not free_text
        and ctx.role != "numeric"  # leave numeric 0/1 columns alone
        and (
            semantic_type == "boolean"
            or ctx.role == "boolean"
            or (bool_share >= 0.7 and (nunique or 0) <= 8)
        )
    )
    money_like = (
        not free_text
        and not identifier_like
        and (
            semantic_type in ("money", "currency")
            or bool(_MONEY_NAME.search(name))
            or currency_share >= 0.5
        )
    )
    dominant_share = 0.0
    if parsed_units and dominant_unit is not None:
        dominant_share = sum(1 for _, u in parsed_units if u == dominant_unit) / len(parsed_units)
    unit_like = (
        not free_text
        and not identifier_like
        and (bool(unit_hint) or (unit_share >= 0.6 and dominant_share >= 0.6))
    )
    date_share = _share(strings, looks_like_date_value)
    date_like = (
        not free_text
        and not identifier_like
        and (
            semantic_type in ("date", "datetime")
            or ctx.role == "datetime"
            or bool(_DATE_NAME.search(name))
            or date_share >= 0.6
        )
    )

    return SemanticColumnInfo(
        name=name,
        role=ctx.role,
        n_nonnull=n_nonnull,
        nunique=nunique,
        high_cardinality=ctx.high_cardinality,
        preserve=ctx.preserve,
        free_text=free_text,
        numeric_like=numeric_like,
        boolean_like=boolean_like,
        money_like=money_like,
        unit_like=unit_like,
        identifier_like=identifier_like,
        dominant_unit=dominant_unit,
        date_like=date_like,
        semantic_type=semantic_type,
        unit=unit_hint or (dominant_unit if unit_like else None),
        allowed_values=allowed,
        mutable=mutable,
        region=(str(region).upper() if isinstance(region, str) else None),
        email_like=email_like,
        phone_like=phone_like,
        dayfirst=dayfirst if isinstance(dayfirst, bool) else None,
        reference_date=reference_date,
    )


def build_semantic_context(
    df: pd.DataFrame,
    config: CleanConfig,
    *,
    stats: dict[object, tuple[int, int, int | None]] | None = None,
) -> SemanticContext:
    """Assemble the semantic context for *df* under *config*.

    ``stats`` maps a column label to its full-column ``(n_rows, n_nonnull,
    nunique)``. When supplied (the native-engine semantic path), *df* holds only
    a bounded distinct sample of each column and these true stats drive role
    inference, cardinality flags, and the non-null count — so the distinct
    sample is scored exactly as the full column would be.
    """
    engine_contexts = build_contexts(df, config, stats=stats)
    dataset, hints, reference_date = _column_hints(config.semantic_context)
    columns: dict[str, SemanticColumnInfo] = {}
    for col in df.columns:
        name = str(col)
        col_stats = (stats or {}).get(col)
        columns[name] = _build_info(
            df,
            col,
            engine_contexts[col],
            config,
            hints.get(name, {}),
            reference_date,
            n_nonnull_override=col_stats[1] if col_stats is not None else None,
        )
    return SemanticContext(
        dataset=dataset,
        columns=columns,
        auto_threshold=config.semantic_auto_threshold,
        review_threshold=config.semantic_review_threshold,
        max_distinct_values=config.semantic_max_distinct_values,
        sample_size=config.semantic_sample_size,
        privacy_policy=config.semantic_privacy_policy,
        mode=config.semantic_mode or "off",
        id_columns=frozenset(config.id_columns),
        preserve_columns=frozenset(config.preserve_columns),
        target_column=config.target_column,
    )
