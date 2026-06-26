"""freshdata format parsers: structural readers for HL7 v2, GPX, SDMX, and EDIFACT.

A parser turns a raw message/file into one or more pandas DataFrames plus an auditable
:class:`ParseResult`; the frames can then be cleaned and domain-validated with
:func:`freshdata.clean`. See :func:`freshdata.parse_domain` and
:func:`freshdata.clean_domain_file` for the high-level entry points.
"""

from __future__ import annotations

from .base import Parser, ParseResult
from .registry import (
    UnknownParserError,
    available,
    get_parser,
    parser_class,
    register,
)

__all__ = [
    "ParseResult",
    "Parser",
    "UnknownParserError",
    "available",
    "get_parser",
    "parser_class",
    "register",
]
