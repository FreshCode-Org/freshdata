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

from .base import DomainError, DomainValidator

#: Built-in packs, as ``"name" -> "module:attribute"`` for lazy import.
_BUILTINS: dict[str, str] = {
    "finance": "freshdata.domains.finance:FinanceValidator",
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
            found[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not break the registry
            continue
    return found


def _resolve_builtin(name: str) -> type:
    module_path, _, attr = _BUILTINS[name].partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def available() -> list[str]:
    """Return all registered domain names (built-in, runtime, and entry-point)."""
    names = set(_BUILTINS) | set(_REGISTERED) | set(_entry_point_classes())
    return sorted(names)


def get_validator(
    name: str, *, column_map: Mapping[str, str] | None = None
) -> DomainValidator:
    """Instantiate the validator registered under *name*.

    Resolution order: built-in packs, then runtime registrations, then
    entry-point plugins. Raises :class:`UnknownDomainError` if nothing matches.
    """
    cls: type | None = None
    if name in _BUILTINS:
        cls = _resolve_builtin(name)
    elif name in _REGISTERED:
        cls = _REGISTERED[name]
    else:
        cls = _entry_point_classes().get(name)
    if cls is None:
        raise UnknownDomainError(name, available())
    return cls(column_map=column_map)
