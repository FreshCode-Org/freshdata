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
from .backends import (
    BackendAdapter,
    BackendExecution,
    BackendParityAdapter,
    BackendParityError,
    BackendParityResult,
    BackendProvenanceError,
    BackendUnavailableError,
    EXTENDED_BACKEND_CONTRACTS,
    ExtendedBackendContract,
    REQUIRED_BACKENDS,
    assert_backend_parity,
    common_native_config,
    evaluate_backend_parity,
    exercise_extended_backend_contract,
    preflight_required_backends,
)

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
