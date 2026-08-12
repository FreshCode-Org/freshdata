"""Optional framework adapters (Polars, etc.).

See `ARCHITECTURE.md <https://github.com/FreshCode-Org/freshdata/blob/main/ARCHITECTURE.md>`_
for how this package fits into the overall cleaning flow.
"""

from .polars import from_pandas, is_polars_frame, to_pandas

__all__ = ["from_pandas", "is_polars_frame", "to_pandas"]
