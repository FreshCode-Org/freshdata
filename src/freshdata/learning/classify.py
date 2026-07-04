"""Stage 3 of the learning pipeline: classify diffs into transform families.

Each distinct ``(raw_value, clean_value)`` pair is tested against FreshData's
own deterministic forward transforms (the Phase-2/3 experts).  A diff is only
assigned a family when running that transform on the raw value reproduces the
clean value — learning never invents transforms, it recognizes its own.

When several families explain a pair, the most specific one (earliest in
:data:`~freshdata.learning.types.TRANSFORM_FAMILIES`) wins.  Pairs nothing
explains are classified ``unexplained`` and can only ever become examples.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Callable

import pandas as pd

from .._sentinels import DEFAULT_SENTINELS
from ..semantic.experts import (
    parse_boolean,
    parse_currency,
    parse_number_words,
    parse_unit,
)
from ..semantic.formats import normalize_phone_in, repair_email
from .types import TRANSFORM_FAMILIES, ClassifiedDiff, DiffSummary, ValueDiff

__all__ = ["classify_diffs"]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

#: Clean-side distinct-value ceiling for allowed-value / category learning.
_ALLOWED_CARDINALITY = 20


def _norm_token(value: object) -> str:
    return _NON_ALNUM.sub("", str(value).strip().casefold())


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_date(value: object, *, dayfirst: bool) -> pd.Timestamp | None:
    text = str(value).strip()
    if not text or not any(ch.isdigit() for ch in text):
        return None
    try:
        parsed = pd.to_datetime(text, dayfirst=dayfirst, errors="coerce")
    except (TypeError, ValueError, OverflowError):  # pragma: no cover - defensive
        return None
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


# ---------------------------------------------------------------------------
# Family testers.  Each returns a params dict when the family explains the
# pair, or None.  ``ctx`` carries column-level context (allowed value sets).
# ---------------------------------------------------------------------------

_Tester = Callable[[ValueDiff, Mapping[str, Any]], "dict[str, Any] | None"]


def _test_whitespace(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str) or not isinstance(d.clean_value, str):
        return None
    collapsed = re.sub(r"\s+", " ", d.raw_value.strip())
    if collapsed == d.clean_value and d.raw_value != d.clean_value:
        return {}
    return None


def _test_case_fold(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str) or not isinstance(d.clean_value, str):
        return None
    collapsed = re.sub(r"\s+", " ", d.raw_value.strip())
    if collapsed.casefold() == d.clean_value.casefold() == d.clean_value:
        return {}
    return None


def _test_sentinel(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if d.kind != "value_to_missing" or not isinstance(d.raw_value, str):
        return None
    token = d.raw_value.strip()
    if not token or len(token) > 24:
        return None
    if _as_float(token) is not None or _parse_date(token, dayfirst=False) is not None:
        return None
    return {"sentinel": token, "known": token.lower() in DEFAULT_SENTINELS}


def _test_currency(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str):
        return None
    clean_num = _as_float(d.clean_value)
    if clean_num is None:
        return None
    parsed = parse_currency(d.raw_value.strip())
    if parsed is not None and abs(parsed - clean_num) < 1e-9:
        return {}
    return None


def _test_unit_strip(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str):
        return None
    clean_num = _as_float(d.clean_value)
    if clean_num is None:
        return None
    parsed = parse_unit(d.raw_value.strip())
    if parsed is not None and abs(parsed[0] - clean_num) < 1e-9:
        return {"unit": parsed[1]}
    return None


def _test_boolean(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str):
        return None
    parsed = parse_boolean(d.raw_value.strip())
    if parsed is None:
        return None
    clean = d.clean_value
    if isinstance(clean, bool) and clean is parsed:
        return {}
    if (
        isinstance(clean, str)
        and clean.strip().casefold() in {"true", "false"}
        and (clean.strip().casefold() == "true") is parsed
    ):
        return {}
    if (
        isinstance(clean, (int, float))
        and not isinstance(clean, bool)
        and clean in (0, 1)
        and bool(clean) is parsed
    ):
        return {}
    return None


def _test_spelled_number(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str):
        return None
    clean_num = _as_float(d.clean_value)
    if clean_num is None:
        return None
    parsed = parse_number_words(d.raw_value.strip())
    if parsed is not None and float(parsed) == clean_num:
        return {}
    return None


def _clean_date_target(d: ValueDiff) -> pd.Timestamp | None:
    clean = d.clean_value
    if isinstance(clean, pd.Timestamp):
        return clean
    if isinstance(clean, str):
        # Clean side must already be canonical ISO to count as date evidence.
        if not re.match(r"^\d{4}-\d{2}-\d{2}", clean.strip()):
            return None
        return _parse_date(clean, dayfirst=False)
    if hasattr(clean, "isoformat"):
        return _parse_date(str(clean), dayfirst=False)
    return None


def _test_date_parse(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str):
        return None
    target = _clean_date_target(d)
    if target is None:
        return None
    first = _parse_date(d.raw_value, dayfirst=False)
    day_first = _parse_date(d.raw_value, dayfirst=True)
    matches_default = first is not None and first == target
    matches_dayfirst = day_first is not None and day_first == target
    if not matches_default and not matches_dayfirst:
        return None
    if matches_dayfirst and not matches_default:
        return {"dayfirst_evidence": True}
    if matches_default and not matches_dayfirst:
        return {"dayfirst_evidence": False}
    return {}  # ambiguous (day == month): no dayfirst signal either way


def _test_email(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str) or not isinstance(d.clean_value, str):
        return None
    if "@" not in d.clean_value:
        return None
    repaired = repair_email(d.raw_value)
    if repaired is not None and repaired[0] == d.clean_value:
        return {"steps": list(repaired[1])}
    return None


def _test_phone(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str) or not isinstance(d.clean_value, str):
        return None
    normalized = normalize_phone_in(d.raw_value)
    if normalized is not None and normalized == d.clean_value:
        return {"region": "IN"}
    return None


def _test_reference(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str) or not isinstance(d.clean_value, str):
        return None
    raw_norm = _norm_token(d.raw_value)
    clean_norm = _norm_token(d.clean_value)
    if not raw_norm or not clean_norm:
        return None
    if raw_norm == clean_norm and d.raw_value != d.clean_value:
        # Same canonical token: punctuation/case/spacing variant of a
        # reference value ("pend-ing" -> "pending").
        allowed = ctx.get("allowed_values")
        if isinstance(allowed, frozenset) and d.clean_value in allowed:
            return {"canonical": d.clean_value}
    return None


def _test_allowed_value(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    allowed = ctx.get("allowed_values")
    if not isinstance(allowed, frozenset) or not allowed:
        return None
    if not isinstance(d.raw_value, str) or d.clean_value not in allowed:
        return None
    if d.raw_value in allowed:
        return None  # a valid value changed into another: not a repair pattern
    return {"allowed_values": sorted(str(v) for v in allowed)}


def _test_category_map(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str) or not isinstance(d.clean_value, str):
        return None
    cardinality = ctx.get("clean_cardinality")
    if cardinality is None or cardinality > _ALLOWED_CARDINALITY:
        return None
    return {}


def _test_numeric_rounding(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_num = _as_float(d.raw_value)
    clean_num = _as_float(d.clean_value)
    if raw_num is None or clean_num is None or raw_num == clean_num:
        return None
    for digits in range(7):
        if round(raw_num, digits) == clean_num:
            return {"digits": digits}
    return None


def _test_dtype_coercion(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(d.raw_value, str):
        return None
    raw_num = _as_float(d.raw_value)
    clean_num = _as_float(d.clean_value)
    if raw_num is not None and clean_num is not None and raw_num == clean_num:
        return {}
    return None


def _test_missing_imputation(d: ValueDiff, ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    if d.kind == "missing_to_value":
        return {}
    return None


_TESTERS: dict[str, _Tester] = {
    "email_normalize": _test_email,
    "phone_normalize": _test_phone,
    "reference_normalize": _test_reference,
    "date_parse": _test_date_parse,
    "currency_parse": _test_currency,
    "unit_strip": _test_unit_strip,
    "spelled_number": _test_spelled_number,
    "boolean_synonym": _test_boolean,
    "sentinel_to_missing": _test_sentinel,
    "allowed_value_map": _test_allowed_value,
    "category_map": _test_category_map,
    "numeric_rounding": _test_numeric_rounding,
    "dtype_coercion": _test_dtype_coercion,
    "case_fold": _test_case_fold,
    "whitespace": _test_whitespace,
    "missing_imputation": _test_missing_imputation,
}


def _column_context(clean_col: pd.Series | None) -> dict[str, Any]:
    if clean_col is None:
        return {}
    non_null = clean_col.dropna()
    distinct = non_null.unique()
    ctx: dict[str, Any] = {"clean_cardinality": int(len(distinct))}
    if 0 < len(distinct) <= _ALLOWED_CARDINALITY:
        import contextlib  # noqa: PLC0415

        with contextlib.suppress(TypeError):  # unhashable categories
            ctx["allowed_values"] = frozenset(distinct.tolist())
    return ctx


def classify_diffs(
    summary: DiffSummary,
    *,
    clean_frame: pd.DataFrame | None = None,
) -> dict[str, list[ClassifiedDiff]]:
    """Assign a transform family to every distinct diff, column by column.

    ``clean_frame`` (the aligned clean side) supplies column-level context:
    the distinct clean values that anchor allowed-value and category
    learning.  Without it those families are skipped, never guessed.
    """
    classified: dict[str, list[ClassifiedDiff]] = {}
    for column, diffs in summary.column_diffs.items():
        clean_col: pd.Series | None = None
        if clean_frame is not None and column in clean_frame.columns:
            candidate = clean_frame[column]
            if isinstance(candidate, pd.Series):
                clean_col = candidate
        ctx = _column_context(clean_col)

        column_results: list[ClassifiedDiff] = []
        for diff in diffs:
            family = "unexplained"
            params: dict[str, Any] = {}
            for name in TRANSFORM_FAMILIES:
                tester = _TESTERS.get(name)
                if tester is None:
                    continue
                result = tester(diff, ctx)
                if result is not None:
                    family, params = name, result
                    break
            column_results.append(ClassifiedDiff(diff=diff, family=family, params=params))

        column_results = _apply_dayfirst_inference(column_results)
        classified[column] = column_results
    return classified


def _apply_dayfirst_inference(results: list[ClassifiedDiff]) -> list[ClassifiedDiff]:
    """Upgrade date_parse to date_dayfirst_inference on consistent evidence."""
    evidence = [
        r.params.get("dayfirst_evidence")
        for r in results
        if r.family == "date_parse" and "dayfirst_evidence" in r.params
    ]
    if not evidence or not all(e is True for e in evidence):
        return results
    upgraded: list[ClassifiedDiff] = []
    for r in results:
        if r.family == "date_parse":
            params = dict(r.params)
            params["dayfirst"] = True
            upgraded.append(
                ClassifiedDiff(diff=r.diff, family="date_dayfirst_inference", params=params)
            )
        else:
            upgraded.append(r)
    return upgraded
