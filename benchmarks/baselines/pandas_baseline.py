"""Hand-written pandas equivalent of FreshData's Layer-1 cleaning.

This is the "do it yourself" baseline: the representation-repair actions
FreshData performs automatically (snake_case names, whitespace strip, sentinel
normalisation, drop empty rows/cols, dtype coercion, numeric median fill,
duplicate removal) expressed as explicit pandas. It produces **no** rationale,
risk level, confidence, audit trail, or id/target protection — that gap is the
point of the authored-code-reduction and explainability comparisons.

``AUTHORED_LINES`` is measured from :func:`run`'s source, so the Metric 6 count
is always honest.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import count_authored_lines

_SENTINELS = {"", "-", "--", "---", "?", "na", "n/a", "n.a", "n.a.", "nan",
              "null", "none", "nil", "missing"}


def run(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\W+", "_", str(c).strip()).strip("_").lower() for c in out.columns]
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            lowered = out[col].map(lambda v: v.lower() if isinstance(v, str) else v)
            out[col] = out[col].mask(lowered.isin(_SENTINELS))
    out = out.dropna(axis=1, how="all").dropna(axis=0, how="all")
    for col in out.columns:
        if out[col].dtype != object:
            continue
        cleaned = out[col].map(
            lambda v: v.replace(",", "").replace("$", "").strip("()")
            if isinstance(v, str) else v
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() >= 0.9:
            out[col] = numeric
            continue
        dt = pd.to_datetime(out[col], errors="coerce")
        if dt.notna().mean() >= 0.9:
            out[col] = dt
    for col in out.select_dtypes(include="number").columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    out = out.drop_duplicates().reset_index(drop=True)
    return out


AUTHORED_LINES: int = count_authored_lines(run)
