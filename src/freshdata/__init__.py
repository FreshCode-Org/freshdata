# src/freshdata/__init__.py
from __future__ import annotations

from freshdata.api import apply, clean, pipeline, plan
from freshdata.config import DEFAULT_NULL_SENTINELS, CleanConfig
from freshdata.execution import (
    DuckDBEngine,
    EngineConfig,
    PandasEngine,
    PolarsEngine,
    StreamingUnsupportedError,
)
from freshdata.report import CleanAction, CleanReport
from freshdata.schema_envelope import ColumnSchemaSpec, SchemaEnvelope

try:
    from freshdata import _freshcore as _native_core  # type: ignore[import-not-found]
    __backend__ = "rust-freshcore"
except ImportError:
    _native_core = None
    __backend__ = "python-vectorized"

__version__ = "0.4.0"

__all__ = [
    "CleanAction",
    "CleanConfig",
    "CleanReport",
    "ColumnSchemaSpec",
    "DEFAULT_NULL_SENTINELS",
    "DuckDBEngine",
    "EngineConfig",
    "PandasEngine",
    "PolarsEngine",
    "SchemaEnvelope",
    "StreamingUnsupportedError",
    "__backend__",
    "__version__",
    "apply",
    "clean",
    "pipeline",
    "plan",
]
