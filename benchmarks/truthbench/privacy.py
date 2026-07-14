"""Privacy-safe TruthBench sink values and exhaustive canary scanning.

The scanner deliberately stores only canary identifiers, transform names, paths,
and run-scoped HMAC digests.  Matched input is never included in a ``Leak`` or
in a redaction marker.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import html
import json
import re
import unicodedata
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from .exact import canonical_json, encode_typed


@dataclass(frozen=True)
class Leak:
    """A privacy leak without retaining the matched value."""

    canary_id: str
    variant: str
    path: str


@dataclass(frozen=True)
class PrivacySafeValue:
    """A leaf value represented by a redacted display and keyed digest."""

    value: str = "[REDACTED]"
    digest: str = ""

    def __repr__(self) -> str:
        return f"PrivacySafeValue(value={self.value!r}, digest={self.digest!r})"


_ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"


def _remove_zero_width(value: str) -> str:
    return "".join(
        char for char in value if char not in _ZERO_WIDTH and unicodedata.category(char) != "Cf"
    )


def _remove_punctuation(value: str) -> str:
    return "".join(char for char in value if not unicodedata.category(char).startswith("P"))


def _json_escaped(value: str) -> str:
    encoded = json.dumps(value, ensure_ascii=True)
    return encoded[1:-1]


def _digit_only(value: str) -> str:
    return "".join(char for char in value if char.isdigit())


Transform = Callable[[str], str]

# Public names make the normalization contract inspectable and testable.
NORMALIZERS: tuple[tuple[str, Transform], ...] = (
    ("literal", lambda value: value),
    ("casefold", str.casefold),
    ("whitespace-stripped", lambda value: re.sub(r"\s+", "", value)),
    ("punctuation-stripped", _remove_punctuation),
    ("digit-only", _digit_only),
    ("url-decoded", urllib.parse.unquote),
    ("html-unescaped", html.unescape),
    ("nfkc", lambda value: unicodedata.normalize("NFKC", value)),
    ("nfc", lambda value: unicodedata.normalize("NFC", value)),
    ("nfd", lambda value: unicodedata.normalize("NFD", value)),
    ("zero-width-removed", _remove_zero_width),
    ("json-escaped", _json_escaped),
)


def normalize_variants(value: str) -> dict[str, str]:
    """Return named normalized forms of text, omitting empty forms."""

    if not isinstance(value, str):
        raise TypeError("normalize_variants expects text")
    result: dict[str, str] = {}
    for name, transform in NORMALIZERS:
        try:
            normalized = transform(value)
        except (UnicodeError, ValueError):
            continue
        if normalized:
            result.setdefault(name, normalized)
    return result


def _text_forms(value: Any) -> dict[str, str]:
    """Convert text/bytes to forms which can be safely searched."""

    if isinstance(value, str):
        return normalize_variants(value)
    if isinstance(value, bytes):
        forms: dict[str, str] = {}
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError:
            decoded = ""
        if decoded:
            for name, text in normalize_variants(decoded).items():
                forms.setdefault(name, text)
        forms["bytes-hex"] = value.hex()
        forms["bytes-escape"] = value.decode("latin1")
        forms["bytes-repr"] = repr(value)
        return forms
    return {}


def _path_component(path: str, key: Any) -> str:
    key_text = str(key)
    if key_text.isidentifier():
        return f"{path}.{key_text}"
    return f"{path}[{json.dumps(key_text, ensure_ascii=False)}]"


def _is_redacted_typed_mapping(value: Mapping[Any, Any]) -> bool:
    """Recognize the exact redacted ``TypedValue.to_dict`` shape."""

    return (
        value.get("redacted") is True
        and value.get("value") is None
        and value.get("display") == "[REDACTED]"
        and bool(value.get("digest"))
    )


_DIGEST_MARKER = re.compile(r"^\[REDACTED:([0-9a-f]{64})\]$")


class SinkScanner:
    """Scan arbitrary nested sink values for normalized synthetic canaries."""

    def __init__(self, canaries: Mapping[str, Any], *, key: bytes, run_id: str = "run") -> None:
        if not isinstance(canaries, Mapping) or not canaries:
            raise ValueError("at least one canary is required")
        if not isinstance(key, bytes) or not key:
            raise TypeError("key must be non-empty bytes")
        self.run_id = run_id
        self._key = key
        self._canaries = {str(identifier): value for identifier, value in canaries.items()}
        self._digests = {
            identifier: self._digest(value) for identifier, value in self._canaries.items()
        }
        self._patterns: dict[str, tuple[tuple[str, str], ...]] = {}
        for identifier, value in self._canaries.items():
            forms: list[tuple[str, str]] = []
            if isinstance(value, bytes):
                forms.extend(_text_forms(value).items())
            elif isinstance(value, str):
                forms.extend(normalize_variants(value).items())
                forms.append(("utf8", value.encode("utf-8").decode("latin1")))
                forms.append(("bytes-hex", value.encode("utf-8").hex()))
                forms.append(
                    ("bytes-escape", "".join(f"\\x{byte:02x}" for byte in value.encode("utf-8")))
                )
            else:
                forms.extend(normalize_variants(str(value)).items())
            # Keep longest forms first; digit-only canaries otherwise produce
            # surprising matches before the literal form.
            priority = {name: index for index, (name, _) in enumerate(NORMALIZERS)}
            priority.update(
                {"utf8": 100, "bytes-hex": 101, "bytes-escape": 102, "bytes-repr": 103}
            )
            self._patterns[identifier] = tuple(
                sorted(
                    set(forms),
                    key=lambda pair: (-len(pair[1]), priority.get(pair[0], 200)),
                )
            )

    @classmethod
    def from_canaries(
        cls,
        canaries: Mapping[str, Any],
        *,
        run_id: str = "run",
        key: bytes | None = None,
        run_key: bytes | None = None,
    ) -> SinkScanner:
        """Build a scanner with a run-scoped HMAC-SHA256 key."""

        if key is not None and run_key is not None:
            raise ValueError("pass only one of key or run_key")
        if run_key is not None:
            key = run_key
        if key is None:
            key = hmac.new(
                b"truthbench-privacy-run-v1",
                str(run_id).encode("utf-8"),
                hashlib.sha256,
            ).digest()
        return cls(canaries, key=key, run_id=run_id)

    @classmethod
    def from_fixture(
        cls,
        fixture: Any,
        *,
        run_id: str = "run",
        key: bytes | None = None,
    ) -> SinkScanner:
        """Build directly from a ``TruthFixture`` canary mapping."""

        canaries = getattr(fixture, "pii_canaries", None)
        if not isinstance(canaries, Mapping):
            raise TypeError("fixture must expose a pii_canaries mapping")
        return cls.from_canaries(canaries, run_id=run_id, key=key)

    def _digest(self, value: Any) -> str:
        encoded = canonical_json(encode_typed(value)).encode("utf-8")
        return hmac.new(self._key, encoded, hashlib.sha256).hexdigest()

    @property
    def digests(self) -> Mapping[str, str]:
        return dict(self._digests)

    def _scan_text(self, value: Any, path: str) -> Leak | None:
        if (
            isinstance(value, str)
            and (match := _DIGEST_MARKER.fullmatch(value)) is not None
            and match.group(1) in self._digests.values()
        ):
            return None
        forms = _text_forms(value)
        for identifier in self._canaries:
            if isinstance(value, bytes):
                byte_patterns = {
                    variant: pattern
                    for variant, pattern in self._patterns[identifier]
                    if variant in {"utf8", "bytes-escape", "bytes-hex", "bytes-repr"}
                }
                for variant in ("utf8", "bytes-escape", "bytes-hex", "bytes-repr"):
                    pattern = byte_patterns.get(variant)
                    if pattern is not None and any(pattern in text for text in forms.values()):
                        return Leak(identifier, variant, path)
            for variant, pattern in self._patterns[identifier]:
                if not pattern:
                    continue
                same_form = forms.get(variant)
                if same_form is not None and pattern in same_form:
                    return Leak(identifier, variant, path)
                # Byte and escape forms do not have corresponding normalizers
                # on ordinary text, so retain a conservative cross-form check.
                if variant.startswith("bytes-") or variant == "utf8":
                    for text in forms.values():
                        if pattern in text:
                            return Leak(identifier, variant, path)
        return None

    def scan(self, sink: Any, *, path: str = "$") -> list[Leak]:
        """Recursively scan a sink and return one leak per canary/path."""

        leaks: list[Leak] = []
        seen: set[tuple[str, str]] = set()

        def visit(value: Any, current_path: str) -> None:
            if isinstance(value, (str, bytes)):
                leak = self._scan_text(value, current_path)
                if leak is not None and (leak.canary_id, leak.path) not in seen:
                    seen.add((leak.canary_id, leak.path))
                    leaks.append(leak)
                return
            if value is None or isinstance(value, (bool, int, float, complex)):
                return
            if isinstance(value, Mapping) and _is_redacted_typed_mapping(value):
                return
            if (
                getattr(value, "redacted", False) is True
                and getattr(value, "value", object()) is None
                and getattr(value, "display", None) == "[REDACTED]"
            ):
                return
            if isinstance(value, pd.DataFrame):
                for column in value.columns:
                    visit(column, f"{current_path}[<column>]")
                for index in value.index:
                    visit(index, f"{current_path}[<index>]")
                for column in value.columns:
                    column_path = (
                        f"{current_path}[<column>]"
                        if self._scan_text(column, current_path)
                        else _path_component(current_path, column)
                    )
                    for index, cell in value[column].items():
                        visit(
                            cell,
                            f"{column_path}[<index>]"
                            if self._scan_text(index, current_path)
                            else (
                                f"{column_path}[{index!r}]"
                                if isinstance(index, str)
                                else f"{column_path}[{index}]"
                            ),
                        )
                return
            if isinstance(value, pd.Series):
                if value.name is not None:
                    visit(value.name, f"{current_path}[<name>]")
                for index in value.index:
                    visit(index, f"{current_path}[<index>]")
                for index, cell in value.items():
                    visit(
                        cell,
                        f"{current_path}[<index>]"
                        if self._scan_text(index, current_path)
                        else (
                            f"{current_path}[{index!r}]"
                            if isinstance(index, str)
                            else f"{current_path}[{index}]"
                        ),
                    )
                return
            if isinstance(value, pd.Index):
                if value.name is not None:
                    visit(value.name, f"{current_path}[<name>]")
                for index, cell in enumerate(value):
                    visit(cell, f"{current_path}[{index}]")
                return
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                for field in dataclasses.fields(value):
                    visit(getattr(value, field.name), _path_component(current_path, field.name))
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    # Keys can also be serialized sinks, but never include a
                    # raw key in a reported path.
                    try:
                        key_text = key if isinstance(key, (str, bytes)) else str(key)
                    except Exception:  # pragma: no cover - hostile key object
                        key_text = f"<{type(key).__name__}>"
                    visit(key_text, f"{current_path}[<key>]")
                    key_leak = self._scan_text(key_text, current_path)
                    item_path = (
                        f"{current_path}[<key>]"
                        if key_leak is not None
                        else _path_component(current_path, key)
                    )
                    visit(item, item_path)
                return
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for index, item in enumerate(value):
                    visit(item, f"{current_path}[{index}]")
                return
            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                try:
                    visit(to_dict(), current_path)
                    return
                except Exception:  # pragma: no cover - defensive sink handling
                    pass
            attrs = getattr(value, "__dict__", None)
            if isinstance(attrs, Mapping):
                visit(attrs, current_path)

        visit(sink, path)
        return leaks

    def redact(self, sink: Any) -> Any:
        """Return a recursively redacted copy of a sink value."""

        def redact_value(value: Any) -> Any:
            if isinstance(value, (str, bytes)):
                leak = self._scan_text(value, "$")
                if leak is not None:
                    # A digest marker contains no input-derived text.
                    return f"[REDACTED:{self._digests[leak.canary_id]}]"
                return value
            if isinstance(value, pd.DataFrame):
                copied = value.copy(deep=True)
                copied.columns = [redact_value(column) for column in copied.columns]
                copied.index = pd.Index([redact_value(index) for index in copied.index])
                for column in copied.columns:
                    copied[column] = copied[column].map(redact_value)
                return copied
            if isinstance(value, pd.Series):
                copied = value.map(redact_value)
                copied.index = pd.Index([redact_value(index) for index in copied.index])
                copied.name = redact_value(value.name) if value.name is not None else None
                return copied
            if isinstance(value, pd.Index):
                return pd.Index(
                    [redact_value(item) for item in value],
                    name=redact_value(value.name) if value.name is not None else None,
                )
            if isinstance(value, Mapping) and _is_redacted_typed_mapping(value):
                return dict(value)
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return {
                    field.name: redact_value(getattr(value, field.name))
                    for field in dataclasses.fields(value)
                }
            if isinstance(value, Mapping):
                result: dict[Any, Any] = {}
                for key, item in value.items():
                    try:
                        key_text = key if isinstance(key, (str, bytes)) else str(key)
                    except Exception:  # pragma: no cover - hostile key object
                        key_text = f"<{type(key).__name__}>"
                    key_leak = self._scan_text(key_text, "$")
                    safe_key = (
                        f"[REDACTED:{self._digests[key_leak.canary_id]}]"
                        if key_leak is not None
                        else key
                    )
                    result[safe_key] = redact_value(item)
                return result
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return [redact_value(item) for item in value]
            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                try:
                    return redact_value(to_dict())
                except Exception:  # pragma: no cover
                    pass
            return value

        return redact_value(sink)

    def self_test(self, sink: Any) -> bool:
        leaks = self.scan(sink)
        if leaks:
            raise AssertionError("privacy self-test found one or more canary leaks")
        return True

    def scan_sinks(self, sinks: Mapping[str, Any]) -> list[Leak]:
        return self.scan(sinks)

    def scan_all(self, **sinks: Any) -> list[Leak]:
        """Scan all supplied named output sinks in one pass."""

        return self.scan(sinks)

    def digest_for(self, value: Any) -> str:
        """Return a run-scoped digest without retaining the value."""

        return self._digest(value)

    def digest(self, value: Any) -> str:
        """Compatibility alias for :meth:`digest_for`."""

        return self.digest_for(value)

    def redact_with_digests(self, sink: Any) -> Any:
        """Alias emphasizing that redaction markers contain keyed digests."""

        return self.redact(sink)

    def redact_value(self, value: Any) -> Any:
        """Method alias for callers that redact one sink value."""

        return self.redact(value)

    def _scan_named(self, name: str, value: Any) -> list[Leak]:
        return self.scan(value, path=f"$.{name}")

    # Named entry points keep sink coverage explicit for callers and audits.
    def scan_exception(self, value: Any) -> list[Leak]:
        return self.scan(value, path="$.exception_text")

    def scan_clean_report(self, value: Any) -> list[Leak]:
        return self.scan(value, path="$.clean_report")

    def scan_action_metadata(self, value: Any) -> list[Leak]:
        return self.scan(value, path="$.actions")

    def scan_generated_code(self, value: Any) -> list[Leak]:
        return self.scan(value, path="$.generated_code")

    def scan_output(self, value: Any) -> list[Leak]:
        return self.scan(value, path="$.output")

    def scan_failure_artifacts(self, value: Any) -> list[Leak]:
        return self.scan(value, path="$.failure_artifacts")

    def scan_coerced_cells(self, value: Any) -> list[Leak]:
        return self._scan_named("coerced_cells", value)

    def scan_findings(self, value: Any) -> list[Leak]:
        return self._scan_named("findings", value)

    def scan_plan(self, value: Any) -> list[Leak]:
        return self._scan_named("plan", value)

    def scan_validation_report(self, value: Any) -> list[Leak]:
        return self._scan_named("validation", value)

    def scan_domain_report(self, value: Any) -> list[Leak]:
        return self._scan_named("domain", value)

    def scan_privacy_report(self, value: Any) -> list[Leak]:
        return self._scan_named("privacy", value)

    def scan_copilot_report(self, value: Any) -> list[Leak]:
        return self._scan_named("copilot", value)

    def scan_stdout(self, value: Any) -> list[Leak]:
        return self._scan_named("stdout", value)

    def scan_stderr(self, value: Any) -> list[Leak]:
        return self._scan_named("stderr", value)

    def scan_markdown(self, value: Any) -> list[Leak]:
        return self._scan_named("markdown", value)

    def scan_html(self, value: Any) -> list[Leak]:
        return self._scan_named("html", value)

    def scan_json(self, value: Any) -> list[Leak]:
        return self._scan_named("json", value)


def redact_value(value: Any, scanner: SinkScanner) -> Any:
    """Functional convenience wrapper around :meth:`SinkScanner.redact`."""

    return scanner.redact(value)


__all__ = [
    "Leak",
    "NORMALIZERS",
    "PrivacySafeValue",
    "SinkScanner",
    "normalize_variants",
    "redact_value",
]
