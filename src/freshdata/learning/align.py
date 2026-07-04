"""Stage 1 of the learning pipeline: align the messy and clean frames.

Key-based alignment is strongly preferred; positional alignment is used only
when it is provably safe (no key given, equal row counts, and compatible
indexes).  Anything else degrades to column-level learning with an explicit
warning rather than silently pairing unrelated rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .types import AlignedPair, AlignmentReport

__all__ = ["align_pair"]


def _normalize_key(key: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if key is None:
        return None
    if isinstance(key, str):
        return (key,)
    keys = tuple(str(k) for k in key)
    if not keys:
        return None
    return keys


def _key_frame(df: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    """Index ``df`` by the (stringified) key columns, keeping first duplicates."""
    idx = pd.MultiIndex.from_frame(df[list(keys)].astype(str)) if len(keys) > 1 else None
    out = df.copy()
    out.index = idx if idx is not None else pd.Index(df[keys[0]].astype(str))
    return out


def _positional_safe(messy: pd.DataFrame, clean: pd.DataFrame) -> tuple[bool, str]:
    """Positional pairing is safe only for equal-length, compatible indexes."""
    if len(messy) != len(clean):
        return False, (
            f"row counts differ (messy={len(messy)}, clean={len(clean)}); "
            "positional alignment would pair unrelated rows"
        )
    messy_range = isinstance(messy.index, pd.RangeIndex)
    clean_range = isinstance(clean.index, pd.RangeIndex)
    if messy_range and clean_range:
        return True, ""
    if list(messy.index) == list(clean.index):
        return True, ""
    return False, "indexes differ; refusing to pair rows by position"


def align_pair(
    messy_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    *,
    key: str | Sequence[str] | None = None,
) -> AlignedPair:
    """Align the training pair row-by-row (or degrade to column-level).

    Returns an :class:`~freshdata.learning.types.AlignedPair` whose
    ``alignment_report.mode`` is one of ``"key"``, ``"positional"``, or
    ``"column_only"``.  In ``column_only`` mode the aligned frames are empty
    and only schema-level learning is possible.
    """
    keys = _normalize_key(key)
    if keys is not None:
        return _align_by_key(messy_df, clean_df, keys)
    return _align_positionally(messy_df, clean_df)


def _align_by_key(
    messy_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    keys: tuple[str, ...],
) -> AlignedPair:
    missing_messy = [k for k in keys if k not in messy_df.columns]
    missing_clean = [k for k in keys if k not in clean_df.columns]
    if missing_messy or missing_clean:
        raise KeyError(
            "alignment key column(s) missing: "
            f"messy={missing_messy or 'ok'}, clean={missing_clean or 'ok'}"
        )

    warnings: list[str] = []
    messy_dupes = int(messy_df.duplicated(subset=list(keys)).sum())
    clean_dupes = int(clean_df.duplicated(subset=list(keys)).sum())
    if messy_dupes:
        warnings.append(
            f"{messy_dupes} duplicate key row(s) in messy frame; kept first occurrence"
        )
    if clean_dupes:
        warnings.append(
            f"{clean_dupes} duplicate key row(s) in clean frame; kept first occurrence"
        )

    messy_k = _key_frame(messy_df.drop_duplicates(subset=list(keys), keep="first"), keys)
    clean_k = _key_frame(clean_df.drop_duplicates(subset=list(keys), keep="first"), keys)

    shared = messy_k.index.intersection(clean_k.index)
    unmatched_messy = len(messy_k) - len(shared)
    unmatched_clean = len(clean_k) - len(shared)
    if unmatched_messy:
        warnings.append(
            f"{unmatched_messy} messy row(s) have no clean counterpart "
            "(candidate row drops; kept as evidence only)"
        )
    if unmatched_clean:
        warnings.append(f"{unmatched_clean} clean row(s) have no messy counterpart")

    report = AlignmentReport(
        mode="key",
        key=keys,
        matched_rows=len(shared),
        unmatched_messy=unmatched_messy,
        unmatched_clean=unmatched_clean,
        duplicate_messy_keys=messy_dupes,
        duplicate_clean_keys=clean_dupes,
        warnings=warnings,
    )
    if len(shared) == 0:
        report.mode = "column_only"
        report.warnings.append(
            "no rows share a key between messy and clean frames; "
            "degrading to column-level learning"
        )
        empty_m = messy_df.iloc[0:0]
        empty_c = clean_df.iloc[0:0]
        return AlignedPair(empty_m, empty_c, report)

    return AlignedPair(messy_k.loc[shared], clean_k.loc[shared], report)


def _align_positionally(messy_df: pd.DataFrame, clean_df: pd.DataFrame) -> AlignedPair:
    safe, reason = _positional_safe(messy_df, clean_df)
    if safe:
        messy = messy_df.reset_index(drop=True)
        clean = clean_df.reset_index(drop=True)
        report = AlignmentReport(
            mode="positional",
            key=None,
            matched_rows=len(messy),
            unmatched_messy=0,
            unmatched_clean=0,
            duplicate_messy_keys=0,
            duplicate_clean_keys=0,
            warnings=[
                "aligned by position (no key= given); pass key= for reliable "
                "row-level learning and row-drop evidence"
            ],
        )
        return AlignedPair(messy, clean, report)

    report = AlignmentReport(
        mode="column_only",
        key=None,
        matched_rows=0,
        unmatched_messy=len(messy_df),
        unmatched_clean=len(clean_df),
        duplicate_messy_keys=0,
        duplicate_clean_keys=0,
        warnings=[
            f"row alignment unsafe: {reason}; degrading to column-level learning "
            "(pass key= to enable cell-level learning)"
        ],
    )
    return AlignedPair(messy_df.iloc[0:0], clean_df.iloc[0:0], report)
