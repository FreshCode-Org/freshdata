"""Column-name normalization: snake_case, valid identifiers, no collisions."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from ..report import CleanReport

# CamelCase boundaries: "CustomerID" -> "Customer_ID", "HTTPCode" -> "HTTP_Code"
_CAMEL_LOWER_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD = re.compile(r"\W+", re.UNICODE)  # keeps unicode letters/digits


def snake_case(name: str) -> str:
    """Convert one column name to snake_case.

    ``" First Name "`` -> ``"first_name"``, ``"CustomerID"`` -> ``"customer_id"``,
    ``"Salary($)"`` -> ``"salary"``. Returns ``""`` if nothing alphanumeric remains.
    """
    s = _CAMEL_ACRONYM.sub("_", _CAMEL_LOWER_UPPER.sub("_", name.strip()))
    s = _NON_WORD.sub("_", s).strip("_").lower()
    return re.sub(r"__+", "_", s)


def _deduplicate(names: list[object]) -> list[object]:
    """Suffix repeated names with ``_2``, ``_3``, … without creating new clashes."""
    seen: set = set()
    counters: dict[object, int] = {}
    out: list[object] = []
    for name in names:
        candidate = name
        if candidate in seen:
            k = counters.get(name, 1)
            while candidate in seen:
                k += 1
                candidate = f"{name}_{k}"
            counters[name] = k
        seen.add(candidate)
        out.append(candidate)
    return out


def normalized_column_labels(columns: Iterable[object]) -> list[object]:
    """Return the labels produced by the column-name normalization step.

    Keeping this transformation separate lets callers translate metadata such as
    domain ``column_map`` overrides before the frame itself is cleaned.
    """
    renamed: list[object] = []
    for i, col in enumerate(columns):
        if isinstance(col, str):
            renamed.append(snake_case(col) or f"column_{i}")
        else:
            renamed.append(col)
    return _deduplicate(renamed)


def normalize_column_names(df: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    """snake_case string column names; deduplicate collisions; keep non-str labels.

    Non-string labels (e.g. integer positions from a headerless CSV) are left
    untouched — inventing names for them would be surprising.
    """
    renamed = normalized_column_labels(df.columns)

    changes = [(old, new) for old, new in zip(df.columns, renamed) if old != new]
    if changes:
        preview = ", ".join(f"{o!r}->{n!r}" for o, n in changes[:4])
        if len(changes) > 4:
            preview += f", … (+{len(changes) - 4} more)"
        report.add("column_names", f"renamed {len(changes)} column(s): {preview}",
                   count=len(changes))
        df.columns = pd.Index(renamed)
    return df
