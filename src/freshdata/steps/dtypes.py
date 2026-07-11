"""Type inference for text columns: boolean, numeric, datetime.

Strategy per column, in priority order (first hit wins):

1. **boolean** — every non-missing value is in a small true/false vocabulary.
2. **numeric** — plain ``to_numeric``, with a second pass that understands
   thousands separators and currency symbols (``"$1,234.56"``).
3. **datetime** — guarded by a cheap "looks like a date" heuristic so we never
   pay datetime parsing on obviously non-date columns.

Conversions are *validated*: they only stick if at least ``*_threshold`` of
the non-missing values parse, and every value coerced to missing is counted
and reported. Large columns are pre-screened on a sample so hopeless
conversions are rejected at O(sample) instead of O(n).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable

import pandas as pd
from pandas.api.types import infer_dtype

from .._util import PANDAS_MAJOR, sample_series, stringlike_columns
from ..config import CleanConfig
from ..report import CleanReport

_TRUE_WORDS = frozenset({"true", "t", "yes", "y"})
_FALSE_WORDS = frozenset({"false", "f", "no", "n"})
_BOOL_WORDS = _TRUE_WORDS | _FALSE_WORDS

# Currency symbols stripped from numbers dressed up as text.
_CURRENCY = "$€£₹"

# A zero-padded number ("007", "02115") — a leading zero followed by another
# digit. These are almost always identifiers (ZIP, phone, padded keys) where
# coercion to int silently destroys the padding, so we keep them as text.
_LEADING_ZERO = re.compile(r"^\s*[+-]?0\d")


def _number_format(
    config: CleanConfig,
) -> tuple[re.Pattern, re.Pattern, Callable[[pd.Series], pd.Series]]:
    """Build (formatted-number matcher, noise detector, cleanup) for the locale.

    The defaults (``thousands=","``, ``decimal="."``) reproduce the original
    US-style ``"$1,234.56"`` behavior exactly; other locales (e.g. European
    ``"1.234,56"``) are supported by setting the separators on the config.
    """
    tho = re.escape(config.thousands)
    dec = re.escape(config.decimal)
    formatted = re.compile(
        rf"\s*[+-]?\s*[{_CURRENCY}]?\s*"
        rf"(?:\d{{1,3}}(?:{tho}\d{{3}})+|\d+)(?:{dec}\d+)?\s*[{_CURRENCY}]?\s*"
    )
    # Characters that hint a value is a *formatted* number worth a second pass.
    noise_chars = _CURRENCY + config.thousands + (config.decimal if config.decimal != "." else "")
    noise = re.compile(rf"[{re.escape(noise_chars)}]")
    strip = re.compile(rf"[\s{_CURRENCY}{tho}]")

    def cleanup(s: pd.Series) -> pd.Series:
        out = s.str.replace(strip, "", regex=True)
        if config.decimal != ".":
            out = out.str.replace(config.decimal, ".", regex=False)
        return out

    return formatted, noise, cleanup


def _has_leading_zero_ids(sample: pd.Series) -> bool:
    """True if the sample contains zero-padded numeric strings (ZIP/IDs)."""
    return any(isinstance(v, str) and _LEADING_ZERO.match(v) for v in sample)

_DATEISH = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"  # 2021-01-05, 1/5/21, 05.01.2021
    r"|\d{1,2}:\d{2}"  # 14:30
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ .,-]*\d{1,4}"
    r"|\d{1,2}[ .,-]*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.IGNORECASE,
)

# Largest float that safely round-trips through int64.
# int64's exact range. Bounds are compared in Python-int space (see
# _fits_int64): float64 cannot represent 2**63 - 1 — it rounds up to 2**63 —
# so any float-space threshold either rejects legitimate boundary values or
# admits one that overflows .astype("int64").
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _fits_int64(nonnull: pd.Series) -> bool:
    """True when every value converts to int64 without wrapping.

    Callers guarantee the series is integral (``% 1 == 0``), so ``int()`` on
    the min/max is an exact conversion; ``inf`` raises OverflowError and is
    reported as not fitting.
    """
    try:
        return int(nonnull.min()) >= _INT64_MIN and int(nonnull.max()) <= _INT64_MAX
    except (OverflowError, ValueError):
        return False


def _finalize_numeric(parsed: pd.Series) -> pd.Series:
    """Normalize a parsed numeric column to int64 / Int64 / float64."""
    nonnull = parsed.dropna()
    is_integral = (
        len(nonnull) > 0
        and bool((nonnull % 1 == 0).all())
        and _fits_int64(nonnull)
    )
    if is_integral:
        return parsed.astype("int64") if not parsed.isna().any() else parsed.astype("Int64")
    return parsed.astype("float64")


def _try_boolean(s: pd.Series, nonnull: pd.Series) -> pd.Series | None:
    """Convert true/false-vocabulary text (or raw Python bools) to boolean."""
    try:
        uniques = nonnull.unique()[:10]
    except TypeError:  # unhashable values (lists/dicts) — clearly not boolean
        return None
    if len(uniques) > 8:  # vocabulary has at most 8 spellings
        return None
    if all(isinstance(v, bool) for v in uniques):
        converted = s.astype("boolean")
    elif all(isinstance(v, str) for v in uniques) and {
        v.casefold() for v in uniques
    } <= _BOOL_WORDS:
        mapping = dict.fromkeys(_TRUE_WORDS, True)
        mapping.update(dict.fromkeys(_FALSE_WORDS, False))
        converted = s.str.casefold().map(mapping).astype("boolean")
    else:
        return None
    if not converted.isna().any():
        return converted.astype(bool)
    return converted


def _to_numeric_or_none(values: pd.Series) -> pd.Series | None:
    """``to_numeric`` that tolerates non-scalar cells (lists raise even with
    ``errors="coerce"``)."""
    try:
        return pd.to_numeric(values, errors="coerce")
    except (TypeError, ValueError):
        return None


def _try_numeric(
    s: pd.Series, nonnull: pd.Series, config: CleanConfig
) -> tuple[pd.Series | None, int]:
    """Parse as numeric. Returns ``(converted, n_coerced)`` or ``(None, 0)``."""
    threshold = config.numeric_threshold
    n = len(nonnull)
    sample = sample_series(nonnull, config.sample_size, config.random_state)

    # Zero-padded numbers ("007", "02115") are identifiers, not quantities —
    # coercing them to int silently drops the padding, so keep them as text.
    if config.preserve_leading_zeros and _has_leading_zero_ids(sample):
        return None, 0

    formatted_re, noise_re, cleanup = _number_format(config)

    # Cheap rejection on the sample before paying for a full-column parse.
    sample_parsed = _to_numeric_or_none(sample)
    if sample_parsed is None:
        return None, 0
    parsed = None
    if sample_parsed.notna().mean() >= threshold * 0.8:
        candidate = _to_numeric_or_none(s)
        if candidate is not None and candidate.notna().sum() / n >= threshold:
            parsed = candidate

    if parsed is None:
        # Second chance: values like "$1,234.56". Only worth attempting if the
        # sample actually contains separator/currency characters.
        has_noise = sample.astype("string").str.contains(noise_re, regex=True, na=False)
        if not bool(has_noise.any()):
            return None, 0
        matches = s.str.fullmatch(formatted_re).eq(True)
        if matches.dtype != bool:  # BooleanDtype (string input) keeps NA through eq()
            matches = matches.fillna(False).astype(bool)
        if not matches.any():
            return None, 0
        cleaned = cleanup(s).where(matches, s)
        candidate = _to_numeric_or_none(cleaned)
        if candidate is None or candidate.notna().sum() / n < threshold:
            return None, 0
        parsed = candidate

    n_coerced = int((s.notna() & parsed.isna()).sum())
    return _finalize_numeric(parsed), n_coerced


def _looks_dateish(sample: pd.Series) -> bool:
    """Cheap pre-screen: do most sampled values resemble dates/times at all?"""
    values = [v for v in sample.head(20) if isinstance(v, str) and len(v) <= 40]
    if not values:
        return False
    hits = sum(1 for v in values if _DATEISH.search(v))
    return hits / len(values) >= 0.6


#: Relative-date words dateutil resolves against the real wall-clock date.
#: Duplicated (not imported) from the semantic layer intentionally: the core
#: engine stays import-light and semantic-agnostic, and this check must apply
#: unconditionally, not just when semantic mode is on.
_RELATIVE_DATE_WORDS = frozenset({"today", "yesterday", "tomorrow"})


def _has_relative_date_word(nonnull: pd.Series) -> bool:
    """True if any string value is a literal relative-date phrase.

    ``pd.to_datetime`` (via dateutil) silently resolves ``"today"`` to the
    real current date. A column containing that word must never be
    auto-converted here — it's left as text so a caller who actually knows
    the intended reference date (e.g. the semantic layer's ``DatePhraseExpert``,
    gated on an explicit ``reference_date``) is the only place that resolves it.
    """
    try:
        return bool(
            nonnull.astype("string").str.strip().str.casefold().isin(_RELATIVE_DATE_WORDS).any()
        )
    except (TypeError, AttributeError):  # unhashable/exotic payloads
        return any(
            isinstance(v, str) and v.strip().casefold() in _RELATIVE_DATE_WORDS
            for v in nonnull
        )


_DATE_FIELDS = re.compile(r"^\s*(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})")


def _resolve_dayfirst(values: pd.Series, config: CleanConfig) -> bool:
    """Resolve the day/month order for ambiguous numeric dates.

    Explicit ``True``/``False`` are honored. Under ``"auto"`` we look for
    disambiguating values — a first field > 12 means the day leads, a second
    field > 12 means the month leads — and vote. With no evidence (every value
    ambiguous) we fall back to month-first, matching pandas' default.
    """
    if config.dayfirst is not True and config.dayfirst is not False:
        day_votes = month_votes = 0
        for v in values:
            if not isinstance(v, str):
                continue
            m = _DATE_FIELDS.match(v)
            if not m:
                continue
            first, second = int(m.group(1)), int(m.group(2))
            if first > 12 and first <= 31:
                day_votes += 1
            elif second > 12 and second <= 31:
                month_votes += 1
        return day_votes > month_votes
    return config.dayfirst


def _parse_datetime(
    s: pd.Series, mixed_formats: bool, dayfirst: bool = False
) -> pd.Series | None:
    kwargs: dict = {"errors": "coerce", "dayfirst": dayfirst}
    if mixed_formats:
        kwargs["format"] = "mixed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # format-inference chatter; report covers it
        try:
            return pd.to_datetime(s, **kwargs)
        except (TypeError, ValueError, OverflowError):
            return None  # non-scalar cells can raise even with errors="coerce"


def _try_datetime(
    s: pd.Series, nonnull: pd.Series, kind: str, config: CleanConfig
) -> tuple[pd.Series | None, int]:
    """Parse as datetime. Returns ``(converted, n_coerced)`` or ``(None, 0)``."""
    threshold = config.datetime_threshold
    n = len(nonnull)

    if kind in ("date", "datetime", "datetime64"):
        # Column already holds date/datetime objects — just normalize the dtype.
        parsed = _parse_datetime(s, mixed_formats=False)
    else:
        if _has_relative_date_word(nonnull):
            return None, 0
        sample = sample_series(nonnull, config.sample_size, config.random_state)
        if not _looks_dateish(sample):
            return None, 0
        dayfirst = _resolve_dayfirst(sample, config)
        parsed = None
        plain_sample = _parse_datetime(sample, mixed_formats=False, dayfirst=dayfirst)
        if plain_sample is not None and plain_sample.notna().mean() >= threshold * 0.8:
            parsed = _parse_datetime(s, mixed_formats=False, dayfirst=dayfirst)
        if parsed is None and PANDAS_MAJOR >= 2:
            mixed_sample = _parse_datetime(sample, mixed_formats=True, dayfirst=dayfirst)
            if mixed_sample is not None and mixed_sample.notna().mean() >= threshold * 0.8:
                parsed = _parse_datetime(s, mixed_formats=True, dayfirst=dayfirst)
        if parsed is None:
            return None, 0
        if parsed.notna().sum() / n < threshold and PANDAS_MAJOR >= 2:
            # Single-format inference came up short; mixed-format parsing is
            # slower but handles heterogeneous date styles in one column.
            mixed = _parse_datetime(s, mixed_formats=True, dayfirst=dayfirst)
            if mixed is not None and mixed.notna().sum() > parsed.notna().sum():
                parsed = mixed

    if parsed is None or parsed.notna().sum() / n < threshold:
        return None, 0
    n_coerced = int((s.notna() & parsed.isna()).sum())
    return parsed, n_coerced


def suggest_conversion(
    s: pd.Series, config: CleanConfig
) -> tuple[str, pd.Series | None, int]:
    """Infer the best target type for one text-capable column.

    Returns ``(target, converted, n_coerced)`` where *target* is one of
    ``"boolean"``, ``"numeric"``, ``"datetime"``, ``"none"``. Shared by both
    the cleaning pipeline and :func:`freshdata.profile` so the preview always
    matches what cleaning would actually do.
    """
    nonnull = s.dropna()
    if nonnull.empty:
        return "none", None, 0
    kind = infer_dtype(s, skipna=True)
    if kind not in ("string", "mixed", "mixed-integer", "boolean", "date", "datetime",
                    "datetime64", "mixed-integer-float"):
        return "none", None, 0

    if kind in ("string", "boolean", "mixed"):
        converted = _try_boolean(s, nonnull)
        if converted is not None:
            return "boolean", converted, 0

    if kind not in ("date", "datetime", "datetime64"):
        converted, n_coerced = _try_numeric(s, nonnull, config)
        if converted is not None:
            return "numeric", converted, n_coerced

    converted, n_coerced = _try_datetime(s, nonnull, kind, config)
    if converted is not None:
        return "datetime", converted, n_coerced

    return "none", None, 0


def fix_dtypes(df: pd.DataFrame, config: CleanConfig, report: CleanReport) -> pd.DataFrame:
    """Apply :func:`suggest_conversion` to every object/string column."""
    from ..guard import hard_protected_columns  # noqa: PLC0415 — cycle-safe lazy import

    protected = hard_protected_columns(config, df.columns)
    for col in stringlike_columns(df):
        if str(col) in protected:
            continue  # context-protected columns must stay byte-identical
        target, converted, n_coerced = suggest_conversion(df[col], config)
        if converted is None:
            continue
        description = f"converted to {converted.dtype}"
        if n_coerced:
            description += f" ({n_coerced} unparseable value(s) set to missing)"
        report.add("fix_dtypes", description, column=str(col),
                   count=int(converted.notna().sum()) + n_coerced)
        df[col] = converted
    return df
