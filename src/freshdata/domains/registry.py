"""Plugin registry that maps a ``domain`` string to a :class:`DomainValidator`.

Built-in packs are registered lazily (imported only when first requested, so
``import freshdata`` stays cheap). Third-party packs register themselves through
the ``freshdata.domains`` entry-point group — see ``CONTRIBUTING_DOMAINS.md``.
Built-in names always take precedence over external ones.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Any

from .base import DomainError, DomainValidator

#: Built-in packs, as ``"name" -> "module:attribute"`` for lazy import.
_BUILTINS: dict[str, str] = {
    "finance": "freshdata.domains.finance:FinanceValidator",
    "retail": "freshdata.domains.retail:RetailValidator",
    "transport": "freshdata.domains.transport:TransportValidator",
}

#: Validators registered at runtime via :func:`register`.
_REGISTERED: dict[str, type] = {}

_ENTRY_POINT_GROUP = "freshdata.domains"


class UnknownDomainError(DomainError):
    """Raised when a ``domain`` string matches no registered pack."""

    def __init__(self, name: str, available: list[str]) -> None:
        listed = ", ".join(available) if available else "(none registered)"
        super().__init__(f"unknown domain {name!r}; available domains: {listed}")
        self.name = name
        self.available = available


def register(name: str, validator_cls: type) -> None:
    """Register *validator_cls* under *name* (overrides any prior registration)."""
    if not (isinstance(validator_cls, type) and issubclass(validator_cls, DomainValidator)):
        raise TypeError("validator_cls must be a DomainValidator subclass")
    _REGISTERED[name] = validator_cls


def _entry_point_classes() -> dict[str, type]:
    """Discover third-party packs registered via the entry-point group."""
    found: dict[str, type] = {}
    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # Python 3.9: entry_points() returns a dict keyed by group.
        eps = entry_points().get(_ENTRY_POINT_GROUP, [])  # type: ignore
    for ep in eps:
        try:
            validator_cls = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not break the registry
            continue
        if isinstance(validator_cls, type) and issubclass(validator_cls, DomainValidator):
            found[ep.name] = validator_cls
    return found


def _resolve_builtin(name: str) -> type:
    module_path, _, attr = _BUILTINS[name].partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def available() -> list[str]:
    """Return all registered domain names (built-in, runtime, and entry-point)."""
    names = set(_BUILTINS) | set(_REGISTERED) | set(_entry_point_classes())
    return sorted(names)


def validator_class(name: str) -> type:
    """Resolve the validator class for *name* without instantiating it.

    Resolution order: built-in packs, then runtime registrations, then
    entry-point plugins. Raises :class:`UnknownDomainError` if nothing matches.
    """
    if name in _BUILTINS:
        return _resolve_builtin(name)
    if name in _REGISTERED:
        return _REGISTERED[name]
    cls = _entry_point_classes().get(name)
    if cls is None:
        raise UnknownDomainError(name, available())
    return cls


def get_validator(
    name: str, *, column_map: Mapping[str, str] | None = None, **extra: Any
) -> DomainValidator:
    """Instantiate the validator registered under *name*.

    Extra keyword arguments (e.g. ``gtfs_file``, ``feed``) are forwarded to the
    validator constructor, so multi-frame packs can receive their feed context.
    """
    return validator_class(name)(column_map=column_map, **extra)
