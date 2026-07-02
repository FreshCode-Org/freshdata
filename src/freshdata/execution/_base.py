"""Abstract execution-engine interface shared by all freshdata backends.

A backend takes a *source* (a pandas/polars frame or a file path), applies the
exact same cleaning the in-memory pandas pipeline would, and returns the cleaned
data in its native representation together with a :class:`~freshdata.CleanReport`.

The pandas backend is the reference implementation: it simply delegates to the
existing :func:`freshdata.cleaner.run_pipeline`, so its output is identical to
``fd.clean(df)``. The Polars and DuckDB backends reproduce the deterministic
"representation repair" subset natively (with projection/predicate pushdown and
streaming/spill) and transparently fall back to the pandas pipeline for the
accuracy-first decision engine and other context-dependent steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from ..config import CleanConfig
    from ..report import CleanReport
    from ._config import EngineConfig


class ExecutionEngine(ABC):
    """Base class for a freshdata execution backend."""

    #: Short backend identifier, e.g. ``"pandas"`` / ``"polars"`` / ``"duckdb"``.
    name: str = "base"

    @abstractmethod
    def execute(
        self,
        source: Any,
        config: CleanConfig,
        engine_config: EngineConfig,
    ) -> tuple[Any, CleanReport]:
        """Clean *source* and return ``(cleaned_native, report)``.

        ``cleaned_native`` is in the backend's native format (a pandas frame for
        the pandas/duckdb backends, a polars frame for the polars backend); the
        caller converts it to the requested ``output_format``. The backend never
        raises on a low trust score and never makes cleaning decisions that the
        config did not ask for — it only executes.
        """

    def to_pandas(self, result: Any) -> pd.DataFrame:
        """Convert a backend-native result to a pandas DataFrame."""
        import pandas as pd

        if isinstance(result, pd.DataFrame):
            return result
        to_pandas = getattr(result, "to_pandas", None)
        if callable(to_pandas):
            return to_pandas()
        raise TypeError(f"cannot convert {type(result).__name__} to a pandas DataFrame")
