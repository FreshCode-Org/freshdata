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
from .cleaning import CleaningAdapter
from .validation import ValidationAdapter
from .privacy import PrivacyAdapter
from .reporting import ReportingAdapter
from .copilot import CopilotAdapter

__all__ = [
    "ExceptionDetails",
    "GenericSurfaceAdapter",
    "SurfaceAdapter",
    "SurfaceObservation",
    "adapter_for",
    "adapters",
    "register_adapter",
    "CleaningAdapter",
    "ValidationAdapter",
    "PrivacyAdapter",
    "ReportingAdapter",
    "CopilotAdapter",
]
