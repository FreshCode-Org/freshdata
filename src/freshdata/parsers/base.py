"""Parser interface and result type for turning raw messages into DataFrames.

A :class:`Parser` performs **structural** parsing only — it reads a wire/file format
(HL7 v2, GPX, SDMX-ML, EDIFACT) and returns one or more pandas DataFrames plus an
auditable :class:`ParseResult`. It does *not* clean or domain-validate; that is the job
of :func:`freshdata.clean` once the frames exist. Malformed input is recorded in
:attr:`ParseResult.warnings` rather than raising, so a partial message is still usable.
"""

from __future__ import annotations

import contextlib
import io
import re
import xml.parsers.expat
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..render.mixins import HtmlReprMixin

_MAX_XML_BYTES = 10 * 1024 * 1024

_DTD_NOT_ALLOWED = "XML DTD/entity declarations are not allowed"

#: Byte-order marks (XML 1.0 Appendix F). Order matters: the UTF-32LE mark
#: starts with the UTF-16LE mark, so it is tested first.
_XML_BOMS = (
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\xfe\xff", "utf-16-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xef\xbb\xbf", "utf-8"),
)
#: BOM-less prefixes of a document that starts with ``<`` (Appendix F).
_XML_PREFIXES = (
    (b"\x00\x00\x00\x3c", "utf-32-be"),
    (b"\x3c\x00\x00\x00", "utf-32-le"),
    (b"\x00\x3c\x00\x3f", "utf-16-be"),
    (b"\x3c\x00\x3f\x00", "utf-16-le"),
)
_EBCDIC_XML_PREFIX = b"\x4c\x6f\xa7\x94"  # "<?xm" in EBCDIC
_XML_DECL_ENCODING = re.compile(
    rb"<\?xml\s[^>]*?\bencoding\s*=\s*([\"'])([A-Za-z][A-Za-z0-9._\-]*)\1"
)


def _detect_xml_encoding(data: bytes) -> tuple[str, int]:
    """Return ``(codec, bom_length)`` for XML *data* (XML 1.0 Appendix F).

    Raises ``ValueError`` for EBCDIC, which the XML readers do not support.
    """
    for bom, codec in _XML_BOMS:
        if data.startswith(bom):
            return codec, len(bom)
    for prefix, codec in _XML_PREFIXES:
        if data.startswith(prefix):
            return codec, 0
    if data.startswith(_EBCDIC_XML_PREFIX):
        raise ValueError("unsupported XML encoding (EBCDIC)")
    # A document entity starts with ASCII, so a NUL in either of the first two
    # bytes means UTF-16 (expat applies the same rule, e.g. to "<!DOCTYPE" or
    # leading whitespace with no declaration and no BOM).
    if data[:1] == b"\x00":
        return "utf-16-be", 0
    if data[1:2] == b"\x00":
        return "utf-16-le", 0
    match = _XML_DECL_ENCODING.match(data)
    return (match.group(2).decode("ascii") if match else "utf-8"), 0


def _declares_dtd(data: bytes) -> bool:
    """Whether *data* contains a DTD/entity marker, in its raw bytes or decoded text."""
    lowered = data.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        return True
    codec, bom_length = _detect_xml_encoding(data)
    try:
        # Undecodable bytes become U+FFFD, which cannot form a marker, so a
        # stray invalid sequence cannot hide a DTD from this scan.
        text = data[bom_length:].decode(codec, errors="replace")
    except (LookupError, UnicodeError):
        # An unknown or non-text codec is not unsafe by itself: the expat pass
        # decides, and the XML reader reports the document as invalid.
        return False
    text = text.casefold()
    return "<!doctype" in text or "<!entity" in text


def _reject_dtd_decl(*_args: Any) -> None:
    raise ValueError(_DTD_NOT_ALLOWED)


def _expat_rejects_dtd(data: bytes) -> None:
    """Run expat over *data* and raise ``ValueError`` on any DTD/entity declaration.

    This is the authoritative check: it sees the document exactly as
    ``xml.etree.ElementTree`` will (same parser, same encoding detection), so no
    encoding can hide a declaration from it. Parse errors are left for the real
    XML reader to report, so this never raises ``ExpatError``.
    """
    parser = xml.parsers.expat.ParserCreate(None, "}")  # ElementTree's configuration
    parser.StartDoctypeDeclHandler = _reject_dtd_decl
    parser.EntityDeclHandler = _reject_dtd_decl
    with contextlib.suppress(xml.parsers.expat.ExpatError, LookupError, UnicodeError):
        parser.Parse(data, True)


@dataclass
class ParseResult(HtmlReprMixin):
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

    _render_kind = "parse"

    def summary(self) -> str:
        """Plain-text Peel summary — parsing is structural only, so this makes
        the parsed/validated/cleaned distinction explicit (spec §9)."""
        # Deferred: render.normalize imports back into freshdata internals; a
        # module-level import here would be circular at package-init time.
        from ..render import normalize, plain  # noqa: PLC0415
        from ..render.options import get_display  # noqa: PLC0415

        return plain.render_plain(normalize.normalize(self), get_display(mode="standard"))

    def __str__(self) -> str:
        return self.summary()

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
        """Parse *source* (Path, text, bytes, or file-like) into a :class:`ParseResult`."""

    def read_text(self, source: Any, *, encoding: str = "utf-8-sig") -> str:
        """Read *source* into text, accepting a Path, str content, bytes, or file-like.

        A leading UTF-8 byte-order mark (common in files exported by Windows tools) is
        dropped: ``utf-8-sig`` strips it when decoding bytes, and already-decoded text
        has it removed explicitly.
        """
        if isinstance(source, (bytes, bytearray)):
            return bytes(source).decode(encoding)
        if hasattr(source, "read"):
            data = source.read()
            if isinstance(data, (bytes, bytearray)):
                return bytes(data).decode(encoding)
            return data.lstrip("\ufeff") if isinstance(data, str) else data
        if isinstance(source, Path):
            return source.read_text(encoding=encoding)
        if isinstance(source, str):
            return source.lstrip("\ufeff")
        raise TypeError(f"cannot read a {type(source).__name__} source")

    def open_binary(self, source: Any) -> io.BufferedIOBase | io.BytesIO:
        """Return a binary stream for XML parsers, accepting path/bytes/text/file-like."""
        if isinstance(source, (bytes, bytearray)):
            return io.BytesIO(bytes(source))
        if hasattr(source, "read"):
            data = source.read()
            return io.BytesIO(data if isinstance(data, (bytes, bytearray))
                              else str(data).encode("utf-8"))
        if isinstance(source, Path):
            return open(source, "rb")  # noqa: SIM115 - caller consumes immediately
        if isinstance(source, str):
            return io.BytesIO(str(source).encode("utf-8"))
        raise TypeError(f"cannot open a {type(source).__name__} source")

    def open_safe_xml_binary(
        self,
        source: Any,
        *,
        max_bytes: int = _MAX_XML_BYTES,
    ) -> io.BytesIO:
        """Return bounded XML bytes with DTD/entity declarations rejected.

        The document encoding is detected first (byte-order mark, UTF-16/UTF-32
        prefix, or XML declaration) so the marker scan also covers non-UTF-8
        documents, and an expat pass then rejects any DOCTYPE or entity
        declaration the XML reader would process. Raises ``ValueError`` for an
        oversized, DTD-bearing, or EBCDIC document.
        """
        stream = self.open_binary(source)
        try:
            data = stream.read(max_bytes + 1)
        finally:
            if hasattr(stream, "close"):
                stream.close()

        if len(data) > max_bytes:
            raise ValueError(f"XML input exceeds {max_bytes} bytes")
        data = bytes(data)
        if _declares_dtd(data):
            raise ValueError(_DTD_NOT_ALLOWED)
        _expat_rejects_dtd(data)
        return io.BytesIO(data)
