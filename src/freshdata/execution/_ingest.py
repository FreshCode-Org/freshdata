"""Input-driven reasons a pandas source must take the pandas reference path.

:meth:`~freshdata.execution.PlanGenerator.fallback_reason` looks only at the
config. Some *inputs* cannot enter a native engine faithfully under any config:
Arrow/Polars ingestion rejects duplicate column labels, an object column
holding mixed value types is either rejected (Polars) or silently cast to text
(DuckDB), and some pandas extension dtypes survive the round trip only as a
different — sometimes silently truncated — value. Those runs take the disclosed
pandas fallback instead, which ``fallback_policy="error"`` still blocks before
any pandas work.
"""

from __future__ import annotations

from typing import Any

#: ``infer_dtype`` kinds of an object column that native ingestion cannot keep as-is.
_MIXED_KINDS = frozenset({"mixed", "mixed-integer"})


def pandas_ingest_fallback_reason(source: Any, engine: str | None = None) -> str | None:
    """Return why *source* must be cleaned by the pandas reference, or ``None``.

    *engine* names the backend that would ingest *source*. Checks that belong
    to one backend's type system only apply to that backend; omit it to run the
    checks every native engine shares.
    """
    import pandas as pd
    from pandas.api.types import infer_dtype, is_object_dtype

    if not isinstance(source, pd.DataFrame):
        return None
    if source.columns.duplicated().any():
        return "duplicate input column labels require the pandas reference path"
    for i, dtype in enumerate(source.dtypes):
        label = source.columns[i]
        if is_object_dtype(dtype) and infer_dtype(source.iloc[:, i], skipna=True) in _MIXED_KINDS:
            return (
                f"object column {label!r} mixes value types (e.g. numbers "
                "and strings), which native ingestion would reject or cast to text"
            )
        lossy = _lossy_dtype_reason(dtype, label, engine)
        if lossy is not None:
            return lossy
    return None


def _lossy_dtype_reason(dtype: Any, label: Any, engine: str | None) -> str | None:
    """Why *engine* cannot carry a column of *dtype* back unchanged."""
    name = str(dtype)
    if name.startswith(("period[", "interval[")):
        # DuckDB rejects both dtypes outright ("Data type not recognized");
        # Polars ingests a period as its raw int64 ordinal (2020-01 -> 600) and
        # an interval as a {left, right} struct.
        return (
            f"column {label!r} has dtype {name}: native ingestion does not carry "
            "period and interval dtypes, so they require the pandas reference path"
        )
    if engine != "duckdb":
        return None
    if name == "timedelta64[ns]":
        return (
            f"timedelta column {label!r} is nanosecond-resolution: DuckDB stores "
            "INTERVAL in microseconds and would truncate it, so it requires the "
            "pandas reference path"
        )
    if name.startswith("datetime64[") and getattr(dtype, "tz", None) is not None:
        return (
            f"datetime column {label!r} is timezone-aware ({dtype.tz}): DuckDB "
            "returns TIMESTAMP WITH TIME ZONE in the session time zone and at "
            "microsecond resolution, so it requires the pandas reference path"
        )
    return None
