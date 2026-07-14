"""Surface adapter protocol and registry."""

from .backends import (
    EXTENDED_BACKEND_CONTRACTS,
    REQUIRED_BACKENDS,
    BackendAdapter,
    BackendExecution,
    BackendParityAdapter,
    BackendParityError,
    BackendParityResult,
    BackendProvenanceError,
    BackendUnavailableError,
    ExtendedBackendContract,
    assert_backend_parity,
    common_native_config,
    evaluate_backend_parity,
    exercise_extended_backend_contract,
    preflight_required_backends,
)
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
from .copilot import CopilotAdapter
from .privacy import PrivacyAdapter
from .reporting import ReportingAdapter
from .validation import ValidationAdapter

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
    "BackendAdapter",
    "BackendExecution",
    "BackendParityAdapter",
    "BackendParityError",
    "BackendParityResult",
    "BackendProvenanceError",
    "BackendUnavailableError",
    "EXTENDED_BACKEND_CONTRACTS",
    "ExtendedBackendContract",
    "REQUIRED_BACKENDS",
    "assert_backend_parity",
    "common_native_config",
    "evaluate_backend_parity",
    "exercise_extended_backend_contract",
    "preflight_required_backends",
]
