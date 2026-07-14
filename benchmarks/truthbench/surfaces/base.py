"""Contracts for observing FreshData's public surfaces.

Adapters are deliberately small: they execute a public API against a fixture and
return an evidence envelope.  The envelope is independent of FreshData's report
types so that TruthBench can compare all backends and sinks uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class ExceptionDetails:
    """Structured unexpected-exception information (never an exception object)."""

    type_name: str
    message: str
    traceback: str | None = None


@dataclass(frozen=True)
class SurfaceObservation:
    """Evidence emitted by one adapter invocation.

    ``output_frame`` and the other payloads intentionally remain ``Any``: pandas,
    Polars, DuckDB, report objects, and scalar outputs all occur on public
    FreshData surfaces.  Adapters must capture failures into
    ``unexpected_exception`` rather than raising them from ``observe``.
    """

    output_frame: Any = None
    raw_decisions: Any = None
    audit_sinks: Any = None
    trust: Any = None
    backend_disclosure: Mapping[str, Any] | None = None
    generated_code: str | None = None
    captured_stdout: str = ""
    captured_stderr: str = ""
    unexpected_exception: ExceptionDetails | None = None

    @property
    def stdout(self) -> str:
        """Compatibility alias used by runner/report code."""

        return self.captured_stdout

    @property
    def stderr(self) -> str:
        """Compatibility alias used by runner/report code."""

        return self.captured_stderr

    @classmethod
    def from_exception(cls, exc: BaseException, **kwargs: Any) -> SurfaceObservation:
        """Build a safe observation for an unexpected exception.

        Traceback text is accepted from callers only when explicitly supplied;
        adapters should sanitize it with the TruthBench privacy scanner before
        persisting a result.
        """

        details = ExceptionDetails(type(exc).__name__, str(exc))
        return cls(unexpected_exception=details, **kwargs)


class SurfaceAdapter(ABC):
    """Abstract protocol implemented by one public-surface adapter."""

    name: ClassVar[str]

    @abstractmethod
    def observe(self, fixture: Any, context: Any) -> SurfaceObservation:
        """Observe *fixture* under *context* and return an evidence envelope."""


class GenericSurfaceAdapter(SurfaceAdapter):
    """Placeholder adapter used until a concrete surface adapter is installed."""

    name = "generic"

    def observe(self, fixture: Any, context: Any) -> SurfaceObservation:
        return SurfaceObservation(output_frame=fixture)


_ADAPTER_TYPES: dict[str, type[SurfaceAdapter]] = {
    GenericSurfaceAdapter.name: GenericSurfaceAdapter
}


def register_adapter(adapter: type[SurfaceAdapter]) -> type[SurfaceAdapter]:
    """Register an adapter class by its stable ``name`` and return the class."""

    name = getattr(adapter, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("surface adapter must define a non-empty name")
    if not isinstance(adapter, type) or not issubclass(adapter, SurfaceAdapter):
        raise TypeError("adapter must subclass SurfaceAdapter")
    _ADAPTER_TYPES[name] = adapter
    return adapter


def adapter_for(name: str) -> SurfaceAdapter:
    """Instantiate the registered adapter named *name*."""

    try:
        adapter_type = _ADAPTER_TYPES[name]
    except KeyError as exc:
        raise KeyError(f"unknown TruthBench surface adapter: {name!r}") from exc
    return adapter_type()


def adapters() -> Mapping[str, type[SurfaceAdapter]]:
    """Return a read-only snapshot of registered adapter classes."""

    return dict(_ADAPTER_TYPES)
