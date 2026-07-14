"""Deterministic oracle and result primitives for FreshData TruthBench."""

from .exact import TypedValue, canonical_json, encode_typed, exact_equal, stable_digest
from .models import (
    UNSET,
    CaseExpectation,
    DecisionRecord,
    Disposition,
    GateResult,
    GoldCell,
    RunResult,
)
from .privacy import Leak, PrivacySafeValue, SinkScanner, normalize_variants, redact_value

__all__ = [
    "UNSET",
    "CaseExpectation",
    "DecisionRecord",
    "Disposition",
    "GateResult",
    "GoldCell",
    "RunResult",
    "TypedValue",
    "canonical_json",
    "encode_typed",
    "exact_equal",
    "stable_digest",
    "Leak",
    "PrivacySafeValue",
    "SinkScanner",
    "normalize_variants",
    "redact_value",
]
