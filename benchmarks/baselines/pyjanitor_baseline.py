"""pyjanitor equivalent for the same defect set.

pyjanitor adds chainable cleaning verbs on top of pandas. It still has no
notion of rationale, risk, id/target protection, or an audit trail, and its
coercion verbs are explicit per column. ``run`` raises ``ImportError`` with an
actionable message when pyjanitor is not installed; the harness skips it.
"""

from __future__ import annotations

import pandas as pd

from . import count_authored_lines


def _require_janitor():
    try:
        import janitor  # noqa: F401  (registers the .clean_names/.remove_empty accessors)
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "pyjanitor is not installed; install with `pip install pyjanitor` "
            "to run this baseline (the harness skips it otherwise)."
        ) from exc


def run(df: pd.DataFrame) -> pd.DataFrame:
    _require_janitor()
    import janitor  # noqa: F401

    sentinels = ["", "-", "--", "?", "na", "n/a", "n.a.", "nan", "null", "none", "missing"]
    out = (
        df.clean_names()
        .remove_empty()
    )
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
            out[col] = out[col].mask(out[col].str.lower().isin(sentinels))
    for col in out.columns:
        if out[col].dtype == object:
            coerced = pd.to_numeric(
                out[col].str.replace(",", "", regex=False).str.replace("$", "", regex=False),
                errors="coerce",
            )
            if coerced.notna().mean() >= 0.9:
                out[col] = coerced
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].fillna(out[col].median())
    out = out.drop_duplicates().reset_index(drop=True)
    return out


AUTHORED_LINES: int = count_authored_lines(run)
