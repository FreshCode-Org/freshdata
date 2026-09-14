"""Input-driven reasons a pandas source must take the pandas reference path.

:meth:`~freshdata.execution.PlanGenerator.fallback_reason` looks only at the
config. Some *inputs* cannot enter a native engine faithfully under any config:
Arrow/Polars ingestion rejects duplicate column labels, and an object column
holding mixed value types is either rejected (Polars) or silently cast to text
(DuckDB). Those runs take the disclosed pandas fallback instead, which
``fallback_policy="error"`` still blocks before any pandas work.
"""

from __future__ import annotations

from typing import Any

#: ``infer_dtype`` kinds of an object column that native ingestion cannot keep as-is.
_MIXED_KINDS = frozenset({"mixed", "mixed-integer"})


def pandas_ingest_fallback_reason(source: Any) -> str | None:
    """Return why *source* must be cleaned by the pandas reference, or ``None``."""
    import pandas as pd
    from pandas.api.types import infer_dtype, is_object_dtype

    if not isinstance(source, pd.DataFrame):
        return None
    if source.columns.duplicated().any():
        return "duplicate input column labels require the pandas reference path"
    for i, dtype in enumerate(source.dtypes):
        if is_object_dtype(dtype) and infer_dtype(source.iloc[:, i], skipna=True) in _MIXED_KINDS:
            return (
                f"object column {source.columns[i]!r} mixes value types (e.g. numbers "
                "and strings), which native ingestion would reject or cast to text"
            )
    return None
