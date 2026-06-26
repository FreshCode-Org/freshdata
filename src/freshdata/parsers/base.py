"""Parser interface and result type for turning raw messages into DataFrames.

A :class:`Parser` performs **structural** parsing only — it reads a wire/file format
(HL7 v2, GPX, SDMX-ML, EDIFACT) and returns one or more pandas DataFrames plus an
auditable :class:`ParseResult`. It does *not* clean or domain-validate; that is the job
of :func:`freshdata.clean` once the frames exist. Malformed input is recorded in
:attr:`ParseResult.warnings` rather than raising, so a partial message is still usable.
"""

from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ParseResult:
    """The structural output of a :class:`Parser`.

    Attributes
    ----------
    format:
        The parser's format name (e.g. ``"hl7v2"``).
    frames:
        Named DataFrames (e.g. ``{"patient": ..., "observation": ...}``).
    suggested_domain:
        The freshdata domain whose validator best fits these frames, if any
        (e.g. HL7 -> ``"healthcare"``). Advisory only.
    metadata:
        Format-level metadata (interchange headers, message type, ...).
    warnings:
        Human-readable notes about anything skipped or not understood — the
        audit trail for partial/invalid input.
    """

    format: str
    frames: dict[str, pd.DataFrame]
    suggested_domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def frame(self) -> pd.DataFrame:
        """The single frame, for formats that yield exactly one."""
        if len(self.frames) == 1:
            return next(iter(self.frames.values()))
        raise ValueError(
            f"{self.format} produced {len(self.frames)} frames {list(self.frames)}; "
            "use .frames instead of .frame"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (row counts per frame, not the data itself)."""
        return {
            "format": self.format,
            "frames": {name: len(df) for name, df in self.frames.items()},
            "suggested_domain": self.suggested_domain,
            "metadata": self.metadata,
            "warnings": list(self.warnings),
        }


class Parser(ABC):
    """Base class for structural format parsers."""

    #: Short format name used by the registry and ``fd.parse_domain(format=...)``.
    format: str = ""
    #: Domain whose validator best fits this parser's output, if any.
    suggested_domain: str | None = None

    @abstractmethod
    def parse(self, source: Any) -> ParseResult:
        """Parse *source* (path, text, bytes, or file-like) into a :class:`ParseResult`."""

    def read_text(self, source: Any, *, encoding: str = "utf-8") -> str:
        """Read *source* into text, accepting a path, str content, bytes, or file-like."""
        if isinstance(source, (bytes, bytearray)):
            return bytes(source).decode(encoding)
        if hasattr(source, "read"):
            data = source.read()
            return data.decode(encoding) if isinstance(data, (bytes, bytearray)) else data
        if isinstance(source, Path):
            return source.read_text(encoding=encoding)
        if isinstance(source, str):
            # A short string that names an existing file is treated as a path;
            # otherwise it is treated as the content itself.
            if (len(source) < 4096 and "\n" not in source and "\r" not in source
                    and os.path.exists(source)):
                return Path(source).read_text(encoding=encoding)
            return source
        raise TypeError(f"cannot read a {type(source).__name__} source")

    def open_binary(self, source: Any) -> io.BufferedIOBase | io.BytesIO:
        """Return a binary stream for XML parsers, accepting path/bytes/text/file-like."""
        if isinstance(source, (bytes, bytearray)):
            return io.BytesIO(bytes(source))
        if hasattr(source, "read"):
            data = source.read()
            return io.BytesIO(data if isinstance(data, (bytes, bytearray))
                              else str(data).encode("utf-8"))
        if isinstance(source, (str, Path)) and os.path.exists(str(source)):
            return open(source, "rb")  # noqa: SIM115 - caller consumes immediately
        if isinstance(source, (str, Path)):
            return io.BytesIO(str(source).encode("utf-8"))
        raise TypeError(f"cannot open a {type(source).__name__} source")
