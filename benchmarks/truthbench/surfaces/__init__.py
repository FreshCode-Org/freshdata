"""Surface adapter protocol and registry."""

from .base import (
    ExceptionDetails,
    GenericSurfaceAdapter,
    SurfaceAdapter,
    SurfaceObservation,
    adapter_for,
    adapters,
    register_adapter,
)

__all__ = [
    "ExceptionDetails",
    "GenericSurfaceAdapter",
    "SurfaceAdapter",
    "SurfaceObservation",
    "adapter_for",
    "adapters",
    "register_adapter",
]
