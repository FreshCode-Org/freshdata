"""Stronger PII detection plus reversible / format-preserving anonymization.

This module approximates Microsoft Presidio-style behaviour while keeping every
heavy dependency optional:

* **Detection** — regex + context-aware scoring over free-text columns
  (:func:`detect_pii`). When the optional ``freshdata-cleaner[privacy]`` extra
  (Presidio) is installed and ``use_ner=True``, an NER pass is layered on;
  otherwise the dependency-free regex/context detector is used.
* **Anonymization** — the legacy ``hash``/``redact``/``partial``/``regex_scrub``/
  ``drop`` strategies plus three new ones: reversible ``tokenize`` (vault-backed),
  ``fpe`` and ``surrogate`` format-preserving anonymization (:func:`anonymize`).
* **k-anonymity** — :func:`check_k_anonymity` flags quasi-identifier groups
  smaller than *k*.

Every masking event carries HIPAA / GDPR tags and a risk level. Reports redact
raw previews by default; pass ``audit_include_pii=True`` to include them.

Security notes:

* No raw PII is logged and no network calls are made.
* Raw keys never appear in reports.
* Reversible anonymization is opt-in (``reversible=True`` with a key).
* The ``surrogate`` / fallback ``fpe`` mode is *format-preserving but not
  cryptographic FPE*; reports mark this as
  ``fpe_mode="surrogate_format_preserving_not_crypto_fpe"``.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ..adapters.polars import from_pandas, to_pandas
from .cleaner import _hash_value, _partial_value, _resolve_columns, _scrub_patterns
from .config import (
    AnonymizationConfig,  # noqa: F401  (re-exported for discoverability)
    KAnonymityConfig,  # noqa: F401
    MaskingRule,
    PIIDetectionConfig,
)

_MAX_EVENTS = 1000
_PREVIEW_LEN = 24


# =====================================================================
# Entity patterns, context keywords, and HIPAA/GDPR maps
# =====================================================================

#: ``entity_type -> (regex, base_score)``. Patterns are lookaround-free so they
#: also work under Polars' Rust regex engine.
ENTITY_PATTERNS: dict[str, tuple[str, float]] = {
    "EMAIL": (r"[\w.+-]+@[\w-]+\.[\w.-]+", 0.90),
    "CREDIT_CARD": (r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b", 0.80),
    "SSN": (r"\b\d{3}-\d{2}-\d{4}\b", 0.85),
    "IBAN": (r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b", 0.80),
    "IP_ADDRESS": (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", 0.70),
    "PHONE": (r"\+?\d[\d ()\-.]{7,}\d", 0.55),
    "DATE_OF_BIRTH": (r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", 0.45),
    "ZIP_CODE": (r"\b\d{5}(?:-\d{4})?\b", 0.35),
    "ICD_CODE": (r"\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b", 0.50),
    "MRN": (r"\bMRN[:#]?\s*\d{4,10}\b", 0.70),
    "PATIENT_ID": (r"\b(?:PT|PID|PAT)[:#-]?\s*\d{4,10}\b", 0.65),
    "INSURANCE_ID": (r"\b[A-Z]{3}\d{6,12}\b", 0.50),
    "PASSPORT": (r"\b[A-Z]{1,2}\d{6,9}\b", 0.50),
    "DRIVER_LICENSE": (r"\b[A-Z]\d{6,12}\b", 0.45),
    "GEO_LOCATION": (r"[-+]?\d{1,3}\.\d{3,},\s*[-+]?\d{1,3}\.\d{3,}", 0.45),
}

HIPAA_CONTEXT = (
    "patient", "mrn", "dob", "diagnosis", "medication", "lab", "encounter", "provider",
)
GDPR_CONTEXT = (
    "name", "email", "phone", "address", "passport", "national id", "location",
)

#: Entities that are HIPAA identifiers, mapped to their identifier category.
HIPAA_TAGS: dict[str, str] = {
    "EMAIL": "HIPAA:email",
    "PHONE": "HIPAA:phone",
    "SSN": "HIPAA:ssn",
    "DATE_OF_BIRTH": "HIPAA:dates",
    "MRN": "HIPAA:medical_record_number",
    "PATIENT_ID": "HIPAA:account_number",
    "ICD_CODE": "HIPAA:health_data",
    "INSURANCE_ID": "HIPAA:health_plan_beneficiary",
    "IP_ADDRESS": "HIPAA:ip_address",
    "ZIP_CODE": "HIPAA:geographic",
    "GEO_LOCATION": "HIPAA:geographic",
    "NAME": "HIPAA:names",
    "PERSON": "HIPAA:names",
    "ADDRESS": "HIPAA:geographic",
    "PASSPORT": "HIPAA:certificate_license_number",
    "DRIVER_LICENSE": "HIPAA:certificate_license_number",
}

#: Entities that are GDPR personal data, mapped to a category label.
GDPR_TAGS: dict[str, str] = {
    "EMAIL": "GDPR:contact_data",
    "PHONE": "GDPR:contact_data",
    "NAME": "GDPR:identity_data",
    "PERSON": "GDPR:identity_data",
    "ADDRESS": "GDPR:contact_data",
    "SSN": "GDPR:national_identifier",
    "PASSPORT": "GDPR:national_identifier",
    "DRIVER_LICENSE": "GDPR:national_identifier",
    "IP_ADDRESS": "GDPR:online_identifier",
    "GEO_LOCATION": "GDPR:location_data",
    "ZIP_CODE": "GDPR:location_data",
    "IBAN": "GDPR:financial_data",
    "CREDIT_CARD": "GDPR:financial_data",
    "DATE_OF_BIRTH": "GDPR:identity_data",
    "INSURANCE_ID": "GDPR:identity_data",
}

_RISK_LEVEL: dict[str, str] = {
    "SSN": "high",
    "CREDIT_CARD": "high",
    "PASSPORT": "high",
    "DRIVER_LICENSE": "high",
    "MRN": "high",
    "PATIENT_ID": "high",
    "INSURANCE_ID": "high",
    "IBAN": "high",
    "EMAIL": "medium",
    "PHONE": "medium",
    "DATE_OF_BIRTH": "medium",
    "IP_ADDRESS": "medium",
    "ADDRESS": "medium",
    "PERSON": "medium",
    "NAME": "medium",
    "GEO_LOCATION": "low",
    "ZIP_CODE": "low",
    "ICD_CODE": "low",
}

_COMPILED: dict[str, re.Pattern[str]] = {
    name: re.compile(pat) for name, (pat, _score) in ENTITY_PATTERNS.items()
}


def hipaa_tag_for(entity_type: str) -> str | None:
    return HIPAA_TAGS.get(entity_type)


def gdpr_tag_for(entity_type: str) -> str | None:
    return GDPR_TAGS.get(entity_type)


def risk_level_for(entity_type: str) -> str:
    return _RISK_LEVEL.get(entity_type, "medium")


# =====================================================================
# Detection dataclasses
# =====================================================================


@dataclass
class PIIEntity:
    """One detected PII span."""

    entity_type: str
    start: int
    end: int
    text: str
    score: float
    source: Literal["regex", "context", "ner", "checksum"]
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass
class PIIScanReport:
    """Result of :func:`detect_pii`."""

    entities: list[PIIEntity] = field(default_factory=list)
    columns_scanned: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def entity_types(self) -> set[str]:
        return {e.entity_type for e in self.entities}

    def by_column(self) -> dict[str, list[PIIEntity]]:
        out: dict[str, list[PIIEntity]] = {}
        for e in self.entities:
            out.setdefault(str(e.metadata.get("column", "")), []).append(e)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_entities": len(self.entities),
            "entity_types": sorted(self.entity_types),
            "columns_scanned": list(self.columns_scanned),
            "entities": [e.to_dict() for e in self.entities],
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        from collections import Counter

        counts = Counter(e.entity_type for e in self.entities)
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return f"detected {len(self.entities)} PII entit(y/ies): {parts or 'none'}"


# =====================================================================
# Detection
# =====================================================================


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 13:
        return False
    checksum = 0
    parity = len(nums) % 2
    for i, num in enumerate(nums):
        d = num
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _context_boost(
    text: str, start: int, end: int, column: str, window: int
) -> tuple[float, list[str]]:
    """Return ``(extra_score, matched_keyword_tags)`` from nearby context."""
    lo = max(0, start - window)
    around = (column + " " + text[lo:end + window]).lower()
    boost = 0.0
    tags: list[str] = []
    for kw in HIPAA_CONTEXT:
        if kw in around:
            boost += 0.15
            tags.append(f"context:{kw}")
            break
    for kw in GDPR_CONTEXT:
        if kw in around:
            boost += 0.10
            tags.append(f"context:{kw}")
            break
    return min(boost, 0.4), tags


def _entity_tags(entity_type: str, context_tags: list[str]) -> tuple[str, ...]:
    tags = list(context_tags)
    if entity_type in HIPAA_TAGS:
        tags.append(HIPAA_TAGS[entity_type])
    if entity_type in GDPR_TAGS:
        tags.append(GDPR_TAGS[entity_type])
    return tuple(dict.fromkeys(tags))  # dedupe, keep order


def detect_in_text(
    text: str, *, column: str = "", config: PIIDetectionConfig | None = None
) -> list[PIIEntity]:
    """Detect PII entities in a single free-text string (regex + context)."""
    cfg = config or PIIDetectionConfig()
    if not cfg.enabled or not isinstance(text, str) or not text:
        return []
    wanted = set(cfg.entities)
    patterns: dict[str, tuple[re.Pattern[str], float]] = {}
    if cfg.use_regex:
        for name, pat in _COMPILED.items():
            if name in wanted:
                patterns[name] = (pat, ENTITY_PATTERNS[name][1])
    for custom in cfg.custom_patterns:
        name = str(custom.get("name", "CUSTOM"))
        patterns[name] = (re.compile(str(custom["regex"])), float(custom.get("score", 0.6)))

    found: list[PIIEntity] = []
    occupied: list[tuple[int, int]] = []
    # Higher base score first so stronger entities win overlapping spans.
    for name in sorted(patterns, key=lambda n: -patterns[n][1]):
        pattern, base = patterns[name]
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if any(s < end and start < e for s, e in occupied):
                continue
            score = base
            source: Literal["regex", "context", "ner", "checksum"] = "regex"
            if name == "CREDIT_CARD" and _luhn_ok(m.group()):
                score = max(score, 0.95)
                source = "checksum"
            context_tags: list[str] = []
            if cfg.use_context:
                boost, context_tags = _context_boost(
                    text, start, end, column, cfg.context_window
                )
                if boost > 0:
                    score = min(1.0, score + boost)
                    if source == "regex":
                        source = "context"
            if score < cfg.min_score:
                continue
            raw = m.group()
            occupied.append((start, end))
            found.append(
                PIIEntity(
                    entity_type=name,
                    start=start,
                    end=end,
                    text="<redacted>" if cfg.redact_samples else raw,
                    score=score,
                    source=source,
                    tags=_entity_tags(name, context_tags),
                    metadata={"column": column},
                )
            )
    found.sort(key=lambda e: e.start)
    return found


def _ner_entities(
    text: str, column: str, cfg: PIIDetectionConfig
) -> list[PIIEntity]:  # pragma: no cover - requires optional Presidio
    """Optional Presidio NER pass; silently returns ``[]`` if unavailable."""
    analyzer = _get_presidio_analyzer()
    if analyzer is None:
        return []
    out: list[PIIEntity] = []
    for r in analyzer.analyze(text=text, language=cfg.language):
        out.append(
            PIIEntity(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                text="<redacted>" if cfg.redact_samples else text[r.start : r.end],
                score=float(r.score),
                source="ner",
                tags=_entity_tags(r.entity_type, []),
                metadata={"column": column},
            )
        )
    return out


_PRESIDIO_ANALYZER: Any = None


def _get_presidio_analyzer() -> Any:  # pragma: no cover - requires optional Presidio
    global _PRESIDIO_ANALYZER
    if _PRESIDIO_ANALYZER is None:
        try:
            from presidio_analyzer import AnalyzerEngine

            _PRESIDIO_ANALYZER = AnalyzerEngine()
        except Exception:
            _PRESIDIO_ANALYZER = None
    return _PRESIDIO_ANALYZER


def detect_pii(df: Any, *, config: PIIDetectionConfig | None = None) -> PIIScanReport:
    """Scan the text columns of *df* for PII; return a :class:`PIIScanReport`.

    Read-only. Only object/string columns are scanned. Raw matched substrings
    are redacted in the report unless ``config.redact_samples=False``.
    """
    cfg = config or PIIDetectionConfig()
    frame = to_pandas(df)
    entities: list[PIIEntity] = []
    scanned: list[str] = []
    for col in frame.columns:
        series = frame[col]
        if series.dtype != object and not pd.api.types.is_string_dtype(series):
            continue
        scanned.append(str(col))
        for row, value in series.items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value)
            cell_entities = detect_in_text(text, column=str(col), config=cfg)
            if cfg.use_ner:
                cell_entities = _merge_ner(cell_entities, _ner_entities(text, str(col), cfg))
            for e in cell_entities:
                e.metadata["row"] = int(row) if isinstance(row, (int, float)) else row
            entities.extend(cell_entities)
    return PIIScanReport(
        entities=entities,
        columns_scanned=tuple(scanned),
        metadata={"ner": bool(cfg.use_ner)},
    )


def _merge_ner(
    regex_hits: list[PIIEntity], ner_hits: list[PIIEntity]
) -> list[PIIEntity]:  # pragma: no cover - requires optional Presidio
    out = list(regex_hits)
    for n in ner_hits:
        if not any(h.start < n.end and n.start < h.end for h in regex_hits):
            out.append(n)
    return out


# =====================================================================
# Token vault + reversible tokenization
# =====================================================================


class TokenVault(ABC):
    """Maps tokens back to their original values for reversible tokenization."""

    @abstractmethod
    def get(self, token: str) -> str | None: ...

    @abstractmethod
    def put(self, token: str, value: str) -> None: ...


class InMemoryTokenVault(TokenVault):
    """A process-local token vault (lost on exit)."""

    def __init__(self) -> None:
        self._map: dict[str, str] = {}

    def get(self, token: str) -> str | None:
        return self._map.get(token)

    def put(self, token: str, value: str) -> None:
        self._map[token] = value

    def __len__(self) -> int:
        return len(self._map)


class JsonTokenVault(TokenVault):
    """A token vault persisted to an explicit JSON file.

    The file holds the sensitive token→value mapping, so protect it like any
    secret. Nothing is written until :meth:`put` (or :meth:`save`) is called.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._map: dict[str, str] = {}
        if self.path.exists():
            self._map = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, token: str) -> str | None:
        return self._map.get(token)

    def put(self, token: str, value: str) -> None:
        self._map[token] = value
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._map, indent=2), encoding="utf-8")


class SqliteTokenVault(TokenVault):
    """A token vault persisted to a SQLite database file.

    Suited to larger reversible-tokenization runs where holding the whole map in
    memory (or rewriting a JSON file on every ``put``) is undesirable. The table
    stores only the ``token -> value`` mapping; protect the file like any secret.
    """

    def __init__(self, path: str | Path) -> None:
        import sqlite3

        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tokens (token TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, token: str) -> str | None:
        cur = self._conn.execute("SELECT value FROM tokens WHERE token = ?", (token,))
        row = cur.fetchone()
        return None if row is None else str(row[0])

    def put(self, token: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO tokens (token, value) VALUES (?, ?)", (token, value)
        )
        self._conn.commit()

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM tokens").fetchone()[0])

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()

    def __del__(self) -> None:  # best-effort: avoid leaking the DB handle
        self.close()


def make_vault(
    backend: str = "memory", *, path: str | Path | None = None
) -> TokenVault:
    """Construct a :class:`TokenVault` for ``backend`` (``memory``/``json``/``sqlite``).

    The ``json`` and ``sqlite`` backends require ``path``. This is the single
    factory used by the policy engine so vault selection stays declarative.
    """
    backend = (backend or "memory").lower()
    if backend == "memory":
        return InMemoryTokenVault()
    if backend == "json":
        if not path:
            raise ValueError("json vault backend requires path=")
        return JsonTokenVault(path)
    if backend == "sqlite":
        if not path:
            raise ValueError("sqlite vault backend requires path=")
        return SqliteTokenVault(path)
    raise ValueError(f"unknown vault backend: {backend!r} (use memory/json/sqlite)")


def vault_metadata(vault: TokenVault | None, backend: str | None = None) -> dict[str, Any]:
    """Describe a vault for an audit report **without ever exposing its secrets**.

    Returns the backend kind, entry count, and (file) location only — never the
    token→value mapping or any key material.
    """
    if vault is None:
        return {"backend": backend or "none", "entries": 0}
    kind = backend or {
        "InMemoryTokenVault": "memory",
        "JsonTokenVault": "json",
        "SqliteTokenVault": "sqlite",
    }.get(type(vault).__name__, type(vault).__name__)
    info: dict[str, Any] = {"backend": kind}
    with contextlib.suppress(TypeError):
        info["entries"] = len(vault)  # type: ignore[arg-type]
    location = getattr(vault, "path", None)
    if location is not None:
        info["location"] = str(location)
    return info


def _hmac_hex(key: str, value: str, length: int = 16) -> str:
    return hmac.new(key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()[
        :length
    ]


def tokenize_value(
    value: Any, vault: TokenVault, key: str, *, prefix: str = "tok"
) -> str:
    """Deterministically tokenize *value* and record it in *vault* for reversal.

    The token is ``f"{prefix}_{hmac_sha256(key, value)}"`` so equal inputs map to
    equal tokens under the same key. The raw key is never stored or logged.
    """
    if not key:
        raise ValueError("tokenize_value requires a non-empty key")
    token = f"{prefix}_{_hmac_hex(key, str(value))}"
    vault.put(token, str(value))
    return token


def detokenize_value(token: str, vault: TokenVault, key: str | None = None) -> str:
    """Recover the original value for *token*; raises ``KeyError`` if unknown.

    Reversal requires the *vault* that holds the mapping — without it (or for an
    unknown token) the original cannot be recovered.
    """
    original = vault.get(token)
    if original is None:
        raise KeyError(f"token not present in vault: {token!r}")
    return original


# =====================================================================
# Format-preserving anonymization (surrogate / FPE)
# =====================================================================


def _surrogate_value(
    value: Any,
    key: str | None,
    *,
    visible: int = 0,
    preserve_domain: bool = False,
) -> str:
    """Deterministic, format-preserving surrogate (NOT cryptographic FPE).

    Preserves digit count, alpha case pattern, and separators; optionally keeps
    the last ``visible`` characters and an email domain. Deterministic per
    ``(key, value)`` so equal inputs map to equal surrogates.
    """
    s = str(value)
    if preserve_domain and "@" in s:
        local, _, domain = s.partition("@")
        return _surrogate_value(local, key, visible=0) + "@" + domain
    seed = (key or "freshdata-surrogate").encode("utf-8")
    digest = hmac.new(seed, s.encode("utf-8"), hashlib.sha256).digest()
    n = len(s)
    keep_from = n - visible if 0 < visible < n else n
    out: list[str] = []
    for i, ch in enumerate(s):
        if i >= keep_from:
            out.append(ch)
            continue
        b = digest[i % len(digest)]
        if ch.isdigit():
            out.append(str(b % 10))
        elif ch.isascii() and ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            out.append(chr(base + b % 26))
        else:
            out.append(ch)
    return "".join(out)


def _fpe_value(value: Any, key: str, *, visible: int = 0) -> tuple[str, str]:
    """Format-preserving encryption when ``pyffx`` is available, else surrogate.

    Returns ``(masked, mode)`` where mode flags whether real FPE was used.
    """
    try:  # pragma: no cover - optional crypto dependency
        import pyffx

        s = str(value)
        digits = "".join(c for c in s if c.isdigit())
        if digits and key:
            cipher = pyffx.Integer(key.encode("utf-8"), length=len(digits))
            enc = str(cipher.encrypt(int(digits))).zfill(len(digits))
            it = iter(enc)
            rebuilt = "".join(next(it) if c.isdigit() else c for c in s)
            return rebuilt, "crypto_fpe"
    except Exception:
        pass
    return _surrogate_value(value, key, visible=visible), (
        "surrogate_format_preserving_not_crypto_fpe"
    )


# =====================================================================
# Anonymization events & report
# =====================================================================


@dataclass
class MaskingEvent:
    """An audit record for one anonymized cell (or detected span)."""

    column: str
    row: Any
    entity_type: str
    strategy: str
    reversible: bool
    format_preserving: bool
    hipaa_tag: str | None
    gdpr_tag: str | None
    risk_level: str
    source: str
    score: float | None = None
    original_preview: str = "<redacted>"
    masked_preview: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # --- policy-engine audit fields (optional; empty for legacy anonymize()) ---
    rule_id: str | None = None
    action: str | None = None
    legal_basis_or_reason: str | None = None
    jurisdiction: str | None = None
    compliance_pack: str | None = None
    classification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "row": self.row,
            "entity_type": self.entity_type,
            "strategy": self.strategy,
            "reversible": self.reversible,
            "format_preserving": self.format_preserving,
            "hipaa_tag": self.hipaa_tag,
            "gdpr_tag": self.gdpr_tag,
            "risk_level": self.risk_level,
            "source": self.source,
            "score": self.score,
            "original_preview": self.original_preview,
            "masked_preview": self.masked_preview,
            "metadata": self.metadata,
            "rule_id": self.rule_id,
            "action": self.action,
            "legal_basis_or_reason": self.legal_basis_or_reason,
            "jurisdiction": self.jurisdiction,
            "compliance_pack": self.compliance_pack,
            "classification": self.classification,
        }


@dataclass
class PrivacyReport:
    """Summary of an :func:`anonymize` run."""

    entities_found: int = 0
    cells_changed: int = 0
    columns_changed: tuple[str, ...] = ()
    events: list[MaskingEvent] = field(default_factory=list)
    k_anonymity: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # --- policy-engine fields (optional; unset for legacy anonymize()) ---
    policy_name: str | None = None
    jurisdiction: str | None = None
    compliance_pack: tuple[str, ...] = ()
    classifications: dict[str, dict[str, Any]] = field(default_factory=dict)
    trust_dimension: dict[str, Any] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)
    vault_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities_found": self.entities_found,
            "cells_changed": self.cells_changed,
            "columns_changed": list(self.columns_changed),
            "n_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
            "k_anonymity": self.k_anonymity,
            "metadata": self.metadata,
            "policy_name": self.policy_name,
            "jurisdiction": self.jurisdiction,
            "compliance_pack": list(self.compliance_pack),
            "classifications": self.classifications,
            "trust_dimension": self.trust_dimension,
            "violations": self.violations,
            "vault_info": self.vault_info,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_frame(self) -> pd.DataFrame:
        """Return the per-event audit trail as a :class:`pandas.DataFrame`.

        One row per :class:`MaskingEvent`. Previews are redacted unless the run
        was created with ``audit_include_pii=True``; no key material is included.
        """
        return pd.DataFrame([e.to_dict() for e in self.events])

    def summary(self) -> str:
        cols = ", ".join(self.columns_changed)
        verb = "applied policy" if self.policy_name else "anonymized"
        head = self.policy_name or ""
        line = (
            f"{verb} {head}: {self.cells_changed} cell(s) across "
            f"{len(self.columns_changed)} column(s): {cols}"
        ).replace(": :", ":")
        if self.jurisdiction:
            line += f"\n  jurisdiction: {self.jurisdiction}"
        if self.compliance_pack:
            line += f"\n  compliance_pack(s): {', '.join(self.compliance_pack)}"
        if self.violations:
            line += f"\n  policy_violations: {len(self.violations)}"
        if "fpe_mode" in self.metadata:
            line += f"\n  fpe_mode: {self.metadata['fpe_mode']}"
        return line


def _preview(value: str, masked: str, *, include_pii: bool) -> tuple[str, str]:
    masked_preview = masked[:_PREVIEW_LEN]
    if include_pii:
        return value[:_PREVIEW_LEN], masked_preview
    return "<redacted>", masked_preview


def _resolve_key(rule: MaskingRule) -> str | None:
    if rule.key_env:
        return os.environ.get(rule.key_env)
    return rule.key or None


# =====================================================================
# anonymize()
# =====================================================================


def anonymize(
    df: Any,
    *,
    rules: tuple[MaskingRule, ...] = (),
    detection_config: PIIDetectionConfig | None = None,
    return_report: bool = True,
    audit_include_pii: bool = False,
) -> Any:
    """Anonymize *df* by column ``rules`` and/or PII ``detection_config``.

    Returns ``(df_out, PrivacyReport)`` when ``return_report`` is true (the
    default), else just ``df_out`` (always the same frame type as the input).
    The input is never mutated. Previews in the report are redacted unless
    ``audit_include_pii=True``.
    """
    frame = to_pandas(df).copy()
    events: list[MaskingEvent] = []
    changed_cols: list[str] = []
    cells_changed = 0
    metadata: dict[str, Any] = {}

    for rule in rules:
        key = _resolve_key(rule)
        vault = _vault_for(rule)
        for column in _resolve_columns(rule, list(frame.columns)):
            if column not in frame.columns:
                continue
            n, fpe_mode = _apply_rule_column(
                frame, column, rule, key, vault, events, audit_include_pii
            )
            if n:
                cells_changed += n
                changed_cols.append(str(column))
            if fpe_mode:
                metadata["fpe_mode"] = fpe_mode

    entities_found = 0
    if detection_config is not None and detection_config.enabled:
        entities_found = _anonymize_detected(
            frame, detection_config, events, changed_cols, audit_include_pii
        )
        cells_changed += entities_found

    report = PrivacyReport(
        entities_found=entities_found,
        cells_changed=cells_changed,
        columns_changed=tuple(dict.fromkeys(changed_cols)),
        events=events[:_MAX_EVENTS],
        metadata=metadata,
    )
    out = from_pandas(frame, df)
    return (out, report) if return_report else out


def _vault_for(rule: MaskingRule) -> TokenVault:
    if rule.token_vault_path:
        return JsonTokenVault(rule.token_vault_path)
    return InMemoryTokenVault()


def _entity_for_rule(rule: MaskingRule, column: str) -> str:
    if rule.entity_types:
        return rule.entity_types[0]
    return "UNKNOWN"


def _record_event(
    events: list[MaskingEvent],
    *,
    column: str,
    row: Any,
    entity_type: str,
    rule: MaskingRule | None,
    strategy: str,
    reversible: bool,
    format_preserving: bool,
    source: str,
    original: str,
    masked: str,
    include_pii: bool,
    score: float | None = None,
) -> None:
    if len(events) >= _MAX_EVENTS:
        return
    orig_prev, masked_prev = _preview(original, masked, include_pii=include_pii)
    hipaa = hipaa_tag_for(entity_type)
    gdpr = gdpr_tag_for(entity_type)
    extra_hipaa = list(rule.hipaa_tags) if rule else []
    extra_gdpr = list(rule.gdpr_tags) if rule else []
    events.append(
        MaskingEvent(
            column=column,
            row=row,
            entity_type=entity_type,
            strategy=strategy,
            reversible=reversible,
            format_preserving=format_preserving,
            hipaa_tag=hipaa or (extra_hipaa[0] if extra_hipaa else None),
            gdpr_tag=gdpr or (extra_gdpr[0] if extra_gdpr else None),
            risk_level=risk_level_for(entity_type),
            source=source,
            score=score,
            original_preview=orig_prev,
            masked_preview=masked_prev,
            metadata={
                "extra_hipaa_tags": extra_hipaa,
                "extra_gdpr_tags": extra_gdpr,
            },
        )
    )


def _apply_rule_column(
    frame: pd.DataFrame,
    column: str,
    rule: MaskingRule,
    key: str | None,
    vault: TokenVault,
    events: list[MaskingEvent],
    include_pii: bool,
) -> tuple[int, str | None]:
    if rule.strategy == "drop":
        n = int(frame[column].notna().sum())
        _record_event(
            events, column=str(column), row=None, entity_type=_entity_for_rule(rule, column),
            rule=rule, strategy="drop", reversible=False, format_preserving=False,
            source="column", original="", masked="<dropped>", include_pii=include_pii,
        )
        frame.drop(columns=[column], inplace=True)
        return n, None

    reversible = rule.strategy in ("tokenize", "fpe") and rule.reversible
    format_preserving = rule.strategy in ("fpe", "surrogate") or rule.preserve_format
    if rule.strategy in ("tokenize", "fpe") and rule.reversible and not key:
        raise ValueError(
            f"masking rule {rule.name!r}: reversible {rule.strategy} requires key= or key_env="
        )

    fpe_mode: str | None = None
    series = frame[column]
    entity_type = _entity_for_rule(rule, str(column))
    n_changed = 0
    new_values: list[Any] = []
    for row, value in series.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            new_values.append(value)
            continue
        original = str(value)
        masked, mode = _mask_one(original, rule, key, vault)
        if mode:
            fpe_mode = mode
        new_values.append(masked)
        if masked != original:
            n_changed += 1
            _record_event(
                events, column=str(column), row=row, entity_type=entity_type, rule=rule,
                strategy=rule.strategy, reversible=reversible,
                format_preserving=format_preserving, source="column",
                original=original, masked=masked, include_pii=include_pii,
            )
    frame[column] = pd.Series(new_values, index=series.index)
    return n_changed, fpe_mode


def _mask_one(
    original: str, rule: MaskingRule, key: str | None, vault: TokenVault
) -> tuple[str, str | None]:
    strategy = rule.strategy
    if strategy == "hash":
        return _hash_value(original, rule.salt, rule.hash_length), None
    if strategy == "redact":
        return rule.placeholder, None
    if strategy == "partial":
        return _partial_value(original, rule.visible, rule.placeholder), None
    if strategy == "regex_scrub":
        scrubbed = original
        for pattern in _scrub_patterns(rule):
            scrubbed = re.sub(pattern, rule.placeholder, scrubbed)
        return scrubbed, None
    if strategy == "tokenize":
        tok_key = key or _hmac_hex("freshdata-default-token-salt", rule.name, 32)
        return tokenize_value(original, vault, tok_key, prefix="tok"), None
    if strategy == "surrogate":
        visible = rule.visible if rule.preserve_format else 0
        preserve_domain = rule.preserve_format and "@" in original
        return _surrogate_value(original, key, visible=visible, preserve_domain=preserve_domain), (
            "surrogate_format_preserving_not_crypto_fpe"
        )
    # fpe
    visible = rule.visible if rule.preserve_format else 0
    if key:
        return _fpe_value(original, key, visible=visible)
    return _surrogate_value(original, key, visible=visible), (
        "surrogate_format_preserving_not_crypto_fpe"
    )


def _anonymize_detected(
    frame: pd.DataFrame,
    cfg: PIIDetectionConfig,
    events: list[MaskingEvent],
    changed_cols: list[str],
    include_pii: bool,
) -> int:
    """Replace detected PII spans in object columns with ``<ENTITY_TYPE>``."""
    n_entities = 0
    for col in list(frame.columns):
        series = frame[col]
        if series.dtype != object and not pd.api.types.is_string_dtype(series):
            continue
        touched = False
        new_values: list[Any] = []
        for row, value in series.items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                new_values.append(value)
                continue
            text = str(value)
            hits = detect_in_text(text, column=str(col), config=cfg)
            if not hits:
                new_values.append(value)
                continue
            touched = True
            masked_text = _scrub_spans(text, hits)
            new_values.append(masked_text)
            for e in hits:
                n_entities += 1
                _record_event(
                    events, column=str(col), row=row, entity_type=e.entity_type, rule=None,
                    strategy="redact", reversible=False, format_preserving=False,
                    source=e.source, original=text, masked=masked_text,
                    include_pii=include_pii, score=e.score,
                )
        if touched:
            frame[col] = pd.Series(new_values, index=series.index)
            changed_cols.append(str(col))
    return n_entities


def _scrub_spans(text: str, hits: list[PIIEntity]) -> str:
    out = text
    for e in sorted(hits, key=lambda h: h.start, reverse=True):
        out = out[: e.start] + f"<{e.entity_type}>" + out[e.end :]
    return out


# =====================================================================
# k-anonymity
# =====================================================================


@dataclass
class KAnonymityReport:
    """Quasi-identifier group analysis for k-anonymity."""

    k: int
    quasi_identifiers: tuple[str, ...]
    n_rows: int
    n_equivalence_classes: int
    smallest_class_size: int
    rows_violating_k: int
    violation_ratio: float
    high_risk_groups: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.smallest_class_size >= self.k if self.n_rows else True

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "quasi_identifiers": list(self.quasi_identifiers),
            "n_rows": self.n_rows,
            "n_equivalence_classes": self.n_equivalence_classes,
            "smallest_class_size": self.smallest_class_size,
            "rows_violating_k": self.rows_violating_k,
            "violation_ratio": round(self.violation_ratio, 6),
            "ok": self.ok,
            "high_risk_groups": self.high_risk_groups,
        }

    def summary(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        return (
            f"k-anonymity (k={self.k}): {verdict} — {self.n_equivalence_classes} class(es), "
            f"smallest={self.smallest_class_size}, {self.rows_violating_k} row(s) violate k"
        )


def check_k_anonymity(
    df: Any,
    quasi_identifiers: list[str],
    *,
    k: int = 5,
    max_report_groups: int = 20,
) -> KAnonymityReport:
    """Flag quasi-identifier groups smaller than *k* (re-identification risk).

    Read-only. Groups every row by *quasi_identifiers* and reports the size
    distribution, the rows in too-small classes, and a capped sample of the
    highest-risk (smallest) groups.
    """
    frame = to_pandas(df)
    missing = [c for c in quasi_identifiers if c not in frame.columns]
    if missing:
        raise KeyError(f"quasi-identifier columns not found: {missing}")
    if not quasi_identifiers:
        raise ValueError("check_k_anonymity requires at least one quasi-identifier")

    n_rows = len(frame)
    sizes = frame.groupby(list(quasi_identifiers), dropna=False).size()
    n_classes = int(len(sizes))
    smallest = int(sizes.min()) if n_classes else 0
    violating_mask = sizes < k
    rows_violating = int(sizes[violating_mask].sum())
    ratio = rows_violating / n_rows if n_rows else 0.0

    high_risk: list[dict[str, Any]] = []
    for key, size in sizes[violating_mask].sort_values().head(max_report_groups).items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        high_risk.append(
            {
                "group": {
                    qi: (None if pd.isna(v) else v)
                    for qi, v in zip(quasi_identifiers, key_tuple)
                },
                "size": int(size),
            }
        )

    return KAnonymityReport(
        k=k,
        quasi_identifiers=tuple(quasi_identifiers),
        n_rows=n_rows,
        n_equivalence_classes=n_classes,
        smallest_class_size=smallest,
        rows_violating_k=rows_violating,
        violation_ratio=ratio,
        high_risk_groups=high_risk,
    )
