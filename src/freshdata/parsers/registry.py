"""Plugin registry mapping a ``format`` string to a :class:`Parser`.

Mirrors the domain registry: built-in parsers are imported lazily, third-party
parsers register through the ``freshdata.parsers`` entry-point group, and built-in
names take precedence. ``import freshdata`` stays cheap because nothing is imported
until a parser is first requested.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from .base import Parser

#: Built-in parsers, as ``"format" -> "module:attribute"`` for lazy import.
_BUILTINS: dict[str, str] = {
    "hl7v2": "freshdata.parsers.hl7v2:HL7v2Parser",
    "gpx": "freshdata.parsers.gpx:GPXParser",
    "sdmx": "freshdata.parsers.sdmx:SDMXParser",
    "edifact": "freshdata.parsers.edifact:EDIFACTParser",
}

_REGISTERED: dict[str, type] = {}
_ENTRY_POINT_GROUP = "freshdata.parsers"


class UnknownParserError(ValueError):
    """Raised when a ``format`` string matches no registered parser."""

    def __init__(self, fmt: str, formats: list[str]) -> None:
        listed = ", ".join(formats) if formats else "(none registered)"
        super().__init__(f"unknown parser format {fmt!r}; available formats: {listed}")
        self.format = fmt
        self.formats = formats


def register(fmt: str, parser_cls: type) -> None:
    """Register *parser_cls* under format *fmt* (overrides any prior registration)."""
    if not (isinstance(parser_cls, type) and issubclass(parser_cls, Parser)):
        raise TypeError("parser_cls must be a Parser subclass")
    _REGISTERED[fmt] = parser_cls


def _entry_point_classes() -> dict[str, type]:
    found: dict[str, type] = {}
    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # Python 3.9: entry_points() returns a dict keyed by group.
        eps = entry_points().get(_ENTRY_POINT_GROUP, [])  # type: ignore
    for ep in eps:
        try:
            parser_cls = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not break the registry
            continue
        if isinstance(parser_cls, type) and issubclass(parser_cls, Parser):
            found[ep.name] = parser_cls
    return found


def _resolve_builtin(fmt: str) -> type:
    module_path, _, attr = _BUILTINS[fmt].partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def available() -> list[str]:
    """Return all registered format names (built-in, runtime, and entry-point)."""
    return sorted(set(_BUILTINS) | set(_REGISTERED) | set(_entry_point_classes()))


def parser_class(fmt: str) -> type:
    """Resolve the parser class for *fmt* (built-in, then runtime, then entry-point)."""
    if fmt in _BUILTINS:
        return _resolve_builtin(fmt)
    if fmt in _REGISTERED:
        return _REGISTERED[fmt]
    cls = _entry_point_classes().get(fmt)
    if cls is None:
        raise UnknownParserError(fmt, available())
    return cls


def get_parser(fmt: str) -> Parser:
    """Instantiate the parser registered under format *fmt*."""
    return parser_class(fmt)()
