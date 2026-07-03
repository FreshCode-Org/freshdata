"""Deterministic corruptors: inject known dirt into a known-clean frame.

Every corruptor takes ``(df, seed=...)`` plus optional column selectors,
returns a **new** corrupted frame (the input is never mutated), and derives
all randomness from ``random.Random(seed)`` — same inputs, same corruption,
on every platform. Corrupting only *string-representable* cells keeps the
ground truth trivially comparable cell-by-cell.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import pandas as pd


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _target_columns(
    df: pd.DataFrame, columns: Iterable[str] | None, *, textual: bool = True
) -> list[str]:
    if columns is not None:
        return [c for c in columns if c in df.columns]
    if not textual:
        return [str(c) for c in df.columns]
    return [
        str(c)
        for c in df.columns
        if df[c].map(lambda v: isinstance(v, str)).any()
    ]


def _corrupt_cells(
    df: pd.DataFrame,
    columns: Iterable[str] | None,
    seed: int,
    share: float,
    transform,
) -> pd.DataFrame:
    """Apply *transform* to a deterministic ``share`` of string cells."""
    rng = _rng(seed)
    out = df.copy(deep=True)
    for col in _target_columns(df, columns):
        for idx in out.index:
            value = out.at[idx, col]
            if isinstance(value, str) and rng.random() < share:
                mutated = transform(value, rng)
                if mutated is not None:
                    out.at[idx, col] = mutated
    return out


# -- 1. whitespace ------------------------------------------------------------

def whitespace_corruption(
    df: pd.DataFrame, columns: Iterable[str] | None = None,
    *, seed: int = 0, share: float = 0.5,
) -> pd.DataFrame:
    """Pad string cells with leading/trailing whitespace."""
    pads = ("  {}", "{} ", " {}  ", "\t{}")
    return _corrupt_cells(
        df, columns, seed, share,
        lambda v, rng: rng.choice(pads).format(v),
    )


# -- 2. casing ----------------------------------------------------------------

def casing_corruption(
    df: pd.DataFrame, columns: Iterable[str] | None = None,
    *, seed: int = 1, share: float = 0.5,
) -> pd.DataFrame:
    """Flip string cells to UPPER / lower / Title case."""
    def flip(v: str, rng: random.Random) -> str:
        return rng.choice((v.upper(), v.lower(), v.title()))

    return _corrupt_cells(df, columns, seed, share, flip)


# -- 3. sentinels ---------------------------------------------------------------

SENTINELS = ("N/A", "null", "-", "NONE", "?")


def sentinel_injection(
    df: pd.DataFrame, columns: Iterable[str] | None = None,
    *, seed: int = 2, share: float = 0.15,
) -> pd.DataFrame:
    """Replace string cells with sentinel junk that should become missing."""
    return _corrupt_cells(
        df, columns, seed, share, lambda v, rng: rng.choice(SENTINELS)
    )


# -- 4/5. emails ----------------------------------------------------------------

def email_double_at(
    df: pd.DataFrame, columns: Iterable[str] | None = None,
    *, seed: int = 3, share: float = 0.5,
) -> pd.DataFrame:
    """Duplicate the ``@`` in email cells (``a@b.com`` -> ``a@@b.com``)."""
    def double(v: str, rng: random.Random) -> str | None:
        return v.replace("@", "@@", 1) if v.count("@") == 1 else None

    return _corrupt_cells(df, columns, seed, share, double)


def email_at_spacing(
    df: pd.DataFrame, columns: Iterable[str] | None = None,
    *, seed: int = 4, share: float = 0.5,
) -> pd.DataFrame:
    """Insert spaces around the ``@`` (``a@b.com`` -> ``a @ b.com``)."""
    def space(v: str, rng: random.Random) -> str | None:
        if v.count("@") != 1:
            return None
        local, _, domain = v.partition("@")
        return f"{local}{rng.choice((' @', ' @ ', '@ '))}{domain}"

    return _corrupt_cells(df, columns, seed, share, space)


# -- 6. Indian phone formatting ----------------------------------------------

def phone_format_corruption(
    df: pd.DataFrame, columns: Iterable[str] | None = None,
    *, seed: int = 5, share: float = 0.8,
) -> pd.DataFrame:
    """De-canonicalize ``+91XXXXXXXXXX`` phones into common Indian formats."""
    def mangle(v: str, rng: random.Random) -> str | None:
        if not (v.startswith("+91") and len(v) == 13 and v[1:].isdigit()):
            return None
        ten = v[3:]
        return rng.choice((
            ten,
            "0" + ten,
            f"{ten[:5]} {ten[5:]}",
            f"+91 {ten[:5]} {ten[5:]}",
            f"91-{ten}",
        ))

    return _corrupt_cells(df, columns, seed, share, mangle)


# -- 7/8. allowed values --------------------------------------------------------

def allowed_value_casing(
    df: pd.DataFrame, columns: Iterable[str] | None = None,
    *, seed: int = 6, share: float = 0.6,
) -> pd.DataFrame:
    """Case-mangle categorical cells that must match a reference list."""
    return casing_corruption(df, columns, seed=seed, share=share)


def allowed_value_punctuation(
    df: pd.DataFrame, columns: Iterable[str] | None = None,
    *, seed: int = 7, share: float = 0.5,
) -> pd.DataFrame:
    """Inject separator noise into categorical cells (``pending``→``pend-ing``)."""
    def punct(v: str, rng: random.Random) -> str | None:
        if len(v) < 4:
            return None
        cut = rng.randrange(1, len(v) - 1)
        return v[:cut] + rng.choice("-_ ") + v[cut:]

    return _corrupt_cells(df, columns, seed, share, punct)


# -- 9. date phrases -------------------------------------------------------------

def date_phrase_corruption(
    df: pd.DataFrame, columns: Iterable[str] | None = None,
    *, seed: int = 8, share: float = 0.6,
) -> pd.DataFrame:
    """Rewrite ISO date strings into month-name / slash forms.

    Targets ``YYYY-MM-DD`` strings, producing forms the deterministic
    :class:`~freshdata.semantic.experts.DatePhraseExpert` resolves without a
    ``dayfirst`` hint.
    """
    months = ("January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December")

    def rephrase(v: str, rng: random.Random) -> str | None:
        parts = v.split("-")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            return None
        year, month, day = (int(p) for p in parts)
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        return rng.choice((
            f"{months[month - 1]} {day}, {year}",
            f"{day} {months[month - 1]} {year}",
            f"{year}/{month:02d}/{day:02d}",
        ))

    return _corrupt_cells(df, columns, seed, share, rephrase)


# -- 10. duplicate rows ----------------------------------------------------------

def duplicate_row_injection(
    df: pd.DataFrame, *, seed: int = 9, n_duplicates: int = 2
) -> pd.DataFrame:
    """Append exact copies of deterministic random rows."""
    rng = _rng(seed)
    if df.empty or n_duplicates <= 0:
        return df.copy(deep=True)
    picks = [rng.randrange(len(df)) for _ in range(n_duplicates)]
    extra = df.iloc[picks]
    return pd.concat([df, extra], ignore_index=True)


#: Registry of all corruptors, for sweep-style harnesses.
CORRUPTORS = {
    "whitespace": whitespace_corruption,
    "casing": casing_corruption,
    "sentinel": sentinel_injection,
    "email_double_at": email_double_at,
    "email_at_spacing": email_at_spacing,
    "phone_format": phone_format_corruption,
    "allowed_value_casing": allowed_value_casing,
    "allowed_value_punctuation": allowed_value_punctuation,
    "date_phrase": date_phrase_corruption,
    "duplicate_rows": duplicate_row_injection,
}
