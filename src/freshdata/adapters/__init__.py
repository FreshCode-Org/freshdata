"""Optional framework adapters (Polars, etc.)."""

from .polars import from_pandas, is_polars_frame, to_pandas

__all__ = ["from_pandas", "is_polars_frame", "to_pandas"]
