"""Stage 2 of the learning pipeline: vectorized cell diffs.

Works per shared column on the aligned frames and aggregates *distinct*
``(raw_value, clean_value)`` pairs with support counts.  Full rows are never
stored — only the distinct cell pairs and their counts survive this stage.
"""

from __future__ import annotations

import warnings

import pandas as pd

from .types import AlignedPair, DiffSummary, RowDiffSummary, SchemaDiffSummary, ValueDiff

__all__ = ["compute_diff"]


def _diff_kind(raw_missing: bool, clean_missing: bool) -> str:
    if raw_missing and not clean_missing:
        return "missing_to_value"
    if clean_missing and not raw_missing:
        return "value_to_missing"
    return "value_change"


def _column_diffs(messy: pd.Series, clean: pd.Series, column: str) -> list[ValueDiff]:
    raw_na = messy.isna()
    clean_na = clean.isna()
    # A cell differs when values are unequal, except both-missing which is equal.
    # eq() between an object (string) column and a datetime64 column makes
    # pandas <2 attempt an implicit to_datetime coercion on the string side,
    # which can emit a dayfirst-ambiguity UserWarning; pandas >=2 dropped that
    # coercion. Suppress here rather than requiring every caller to know why.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        unequal = ~(messy.eq(clean) | (raw_na & clean_na))
    if not unequal.any():
        return []

    pairs = pd.DataFrame(
        {
            "raw": messy[unequal],
            "clean": clean[unequal],
            "raw_na": raw_na[unequal],
            "clean_na": clean_na[unequal],
        }
    )
    # Group on stringified values so unhashable/NaN cells aggregate safely,
    # while keeping one representative original value per group.
    pairs["raw_key"] = pairs["raw"].astype(str)
    pairs["clean_key"] = pairs["clean"].astype(str)

    diffs: list[ValueDiff] = []
    grouped = pairs.groupby(["raw_key", "clean_key"], sort=True, dropna=False)
    for _, group in grouped:
        first = group.iloc[0]
        raw_missing = bool(first["raw_na"])
        clean_missing = bool(first["clean_na"])
        diffs.append(
            ValueDiff(
                column=column,
                raw_value=None if raw_missing else first["raw"],
                clean_value=None if clean_missing else first["clean"],
                support=int(len(group)),
                kind=_diff_kind(raw_missing, clean_missing),
            )
        )
    diffs.sort(key=lambda d: (-d.support, str(d.raw_value)))
    return diffs


def compute_diff(aligned: AlignedPair) -> DiffSummary:
    """Summarize all differences between the aligned messy and clean frames."""
    messy = aligned.messy_aligned
    clean = aligned.clean_aligned
    report = aligned.alignment_report

    messy_cols = list(messy.columns)
    clean_cols = list(clean.columns)
    shared = tuple(c for c in messy_cols if c in set(clean_cols))
    added = tuple(c for c in clean_cols if c not in set(messy_cols))
    removed = tuple(c for c in messy_cols if c not in set(clean_cols))

    dtype_changes: dict[str, tuple[str, str]] = {}
    column_diffs: dict[str, list[ValueDiff]] = {}
    if report.row_level:
        for col in shared:
            m_col = messy[col]
            c_col = clean[col]
            if isinstance(m_col, pd.DataFrame) or isinstance(c_col, pd.DataFrame):
                continue  # duplicated column labels: skip rather than guess
            if str(m_col.dtype) != str(c_col.dtype):
                dtype_changes[col] = (str(m_col.dtype), str(c_col.dtype))
            diffs = _column_diffs(m_col, c_col, col)
            if diffs:
                column_diffs[col] = diffs

    row_diffs = RowDiffSummary(
        dropped_rows=report.unmatched_messy if report.mode == "key" else 0,
        added_rows=report.unmatched_clean if report.mode == "key" else 0,
        keyed=report.mode == "key",
    )
    schema_diffs = SchemaDiffSummary(
        added_columns=added,
        removed_columns=removed,
        shared_columns=shared,
        dtype_changes=dtype_changes,
    )
    return DiffSummary(column_diffs=column_diffs, row_diffs=row_diffs, schema_diffs=schema_diffs)
