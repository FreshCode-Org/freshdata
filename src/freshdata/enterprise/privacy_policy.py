"""Compliance-grade privacy *policy engine* layered on :mod:`freshdata.enterprise.privacy`.

The base ``privacy`` module gives detection and masking *primitives* (regex/context
PII detection, reversible tokenisation, format-preserving surrogates, k-anonymity).
This module turns those into a declarative, jurisdiction-aware **policy engine**:

* :class:`PrivacyPolicy` / :class:`PrivacyRule` / :class:`CompliancePack` describe
  *what is sensitive, where, and what must happen to it*.
* :class:`Jurisdiction` (US / EU / UK / India / Global) scopes rules so the same
  column can be handled differently depending on the governing law.
* Built-in :class:`CompliancePack` rule packs for **HIPAA, FERPA, PCI and GDPR**
  ship as YAML under ``freshdata/compliance/packs`` and are loaded by name.
* :func:`apply_privacy_policy` classifies every column, applies the resolved
  action (``classify``/``tokenize``/``pseudonymize``/``redact``/``drop``/
  ``minimize``/``quarantine``/``preserve_with_reason``) and returns an auditable
  :class:`~freshdata.enterprise.privacy.PrivacyReport`.

Four classifier kinds compose inside a rule: **column-name** (``columns`` /
``column_patterns``), **regex** (``value_regexes``), **context** (``context``
keywords) and the **entity / domain-pack** classifier (``entity_types`` resolved
through the shared detector, contributed by a named compliance pack).

Security posture (inherited and reinforced):

* No raw PII is logged; report previews are redacted unless ``audit_include_pii=True``.
* Vault *metadata* (backend, entry count, location) is reported — **never** the
  token→value map or any key material.
* Reversible pseudonymisation (tokenisation) requires an explicit vault **and**
  key; it is refused otherwise.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from ..adapters.polars import from_pandas, to_pandas
from .config import PIIDetectionConfig
from .privacy import (
    MaskingEvent,
    PrivacyReport,
    TokenVault,
    _luhn_ok,
    detect_in_text,
    detokenize_value,
    make_vault,
    risk_level_for,
    tokenize_value,
    vault_metadata,
)
from .privacy import (
    _fpe_value as _fpe,
)
from .privacy import (
    _surrogate_value as _surrogate,
)

_PREVIEW_LEN = 24
_SAMPLE_ROWS = 200  # cells sampled per column for value/entity classification


# =====================================================================
# Enumerations
# =====================================================================


class Jurisdiction(str, Enum):
    """Governing legal regime used to scope :class:`PrivacyRule` application."""

    US = "US"
    EU = "EU"
    UK = "UK"
    INDIA = "India"
    GLOBAL = "Global"

    @classmethod
    def coerce(cls, value: Any) -> Jurisdiction:
        if isinstance(value, cls):
            return value
        s = str(value).strip()
        for member in cls:
            if member.value.lower() == s.lower():
                return member
        raise ValueError(
            f"unknown jurisdiction {value!r} (use one of {[m.value for m in cls]})"
        )


class Action(str, Enum):
    """What a matched :class:`PrivacyRule` does to a column."""

    CLASSIFY = "classify"
    TOKENIZE = "tokenize"
    PSEUDONYMIZE = "pseudonymize"
    REDACT = "redact"
    DROP = "drop"
    MINIMIZE = "minimize"
    QUARANTINE = "quarantine"
    PRESERVE = "preserve_with_reason"

    @classmethod
    def coerce(cls, value: Any) -> Action:
        if isinstance(value, cls):
            return value
        s = str(value).strip().lower()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(
            f"unknown action {value!r} (use one of {[m.value for m in cls]})"
        )


_REDACT_PLACEHOLDER = "<REDACTED>"
_QUARANTINE_PLACEHOLDER = "<QUARANTINED>"


# =====================================================================
# Rule / pack / policy dataclasses
# =====================================================================


@dataclass
class PrivacyRule:
    """One declarative privacy rule: *match these columns, do this action*.

    Matching combines four classifier kinds (any hit selects the column):
    column-name (:attr:`columns`, :attr:`column_patterns`), value-:attr:`regexes`
    (``value_regexes``), :attr:`context` keywords, and the entity/domain-pack
    classifier (:attr:`entity_types`). :attr:`jurisdictions` scopes the rule; an
    empty tuple means "any jurisdiction".
    """

    id: str
    action: Action = Action.CLASSIFY
    # --- matchers ---
    columns: tuple[str, ...] = ()
    column_patterns: tuple[str, ...] = ()
    value_regexes: tuple[str, ...] = ()
    context: tuple[str, ...] = ()
    entity_types: tuple[str, ...] = ()
    requires_luhn: bool = False
    # --- scope / classification ---
    jurisdictions: tuple[str, ...] = ()
    classification: str | None = None
    risk: str = "medium"
    legal_basis: str | None = None
    minimization: str | None = None
    audit_tags: tuple[str, ...] = ()
    # --- crypto / reversibility ---
    reversible: bool = False
    key: str | None = None
    key_env: str | None = None
    vault_backend: str | None = None
    vault_path: str | None = None
    # --- provenance (filled when contributed by a pack) ---
    pack: str | None = None

    def __post_init__(self) -> None:
        self.action = Action.coerce(self.action)
        self._compiled_cols = [re.compile(p) for p in self.column_patterns]
        self._compiled_vals = [re.compile(p) for p in self.value_regexes]

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, pack: str | None = None) -> PrivacyRule:
        d = dict(data)
        # accept singular spellings for ergonomics
        if "entity" in d:
            d.setdefault("entity_types", (d.pop("entity"),))
        if "jurisdiction" in d:
            d.setdefault("jurisdictions", (d.pop("jurisdiction"),))

        def _tup(key: str) -> tuple[str, ...]:
            v = d.get(key, ())
            if isinstance(v, str):
                return (v,)
            return tuple(v or ())

        return cls(
            id=str(d.get("id") or d.get("name") or "rule"),
            action=d.get("action", "classify"),
            columns=_tup("columns"),
            column_patterns=_tup("column_patterns"),
            value_regexes=_tup("value_regexes"),
            context=_tup("context"),
            entity_types=_tup("entity_types"),
            requires_luhn=bool(d.get("requires_luhn", False)),
            jurisdictions=_tup("jurisdictions"),
            classification=d.get("classification"),
            risk=str(d.get("risk", "medium")),
            legal_basis=d.get("legal_basis"),
            minimization=d.get("minimization"),
            audit_tags=_tup("audit_tags"),
            reversible=bool(d.get("reversible", False)),
            key=d.get("key"),
            key_env=d.get("key_env"),
            vault_backend=d.get("vault_backend"),
            vault_path=d.get("vault_path"),
            pack=pack,
        )


@dataclass
class CompliancePack:
    """A named bundle of jurisdiction-scoped :class:`PrivacyRule` objects.

    Built-in packs (``hipaa``, ``ferpa``, ``pci``, ``gdpr``) ship as YAML and are
    loaded via :func:`load_compliance_pack`. A pack records its default action,
    the actions it permits, audit tags, whether reversible tokenisation is allowed,
    and any required vault settings.
    """

    name: str
    jurisdiction: str = Jurisdiction.GLOBAL.value
    description: str = ""
    reversible_allowed: bool = True
    default_action: str = Action.REDACT.value
    allowed_actions: tuple[str, ...] = tuple(a.value for a in Action)
    audit_tags: tuple[str, ...] = ()
    vault_required: bool = False
    vault_backend: str | None = None
    rules: tuple[PrivacyRule, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompliancePack:
        name = str(data.get("name", "pack"))
        vault = data.get("vault", {}) or {}
        rules = tuple(
            PrivacyRule.from_dict(r, pack=name) for r in data.get("rules", [])
        )
        allowed = data.get("allowed_actions") or tuple(a.value for a in Action)
        return cls(
            name=name,
            jurisdiction=str(data.get("jurisdiction", Jurisdiction.GLOBAL.value)),
            description=str(data.get("description", "")),
            reversible_allowed=bool(data.get("reversible_allowed", True)),
            default_action=str(data.get("default_action", Action.REDACT.value)),
            allowed_actions=tuple(allowed),
            audit_tags=tuple(data.get("audit_tags", ())),
            vault_required=bool(vault.get("required", False)),
            vault_backend=vault.get("backend"),
            rules=rules,
        )


@dataclass
class PrivacyPolicy:
    """A complete privacy policy: jurisdiction, packs, and inline rule overrides.

    Inline :attr:`rules` take priority over pack rules; within either group the
    first rule that matches a column (and is in scope for :attr:`jurisdiction`)
    wins. Set :attr:`minimize` to actually drop columns whose action is
    ``minimize`` (off by default so minimisation is an explicit opt-in).
    """

    name: str = "privacy-policy"
    jurisdiction: str = Jurisdiction.GLOBAL.value
    rules: tuple[PrivacyRule, ...] = ()
    packs: tuple[CompliancePack, ...] = ()
    default_action: str | None = None
    minimize: bool = False
    key: str | None = None
    key_env: str | None = None
    vault_backend: str = "memory"
    vault_path: str | None = None
    detection_config: PIIDetectionConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrivacyPolicy:
        vault = data.get("vault", {}) or {}
        packs: list[CompliancePack] = []
        for p in data.get("packs", ()):  # name string -> builtin; dict -> inline
            packs.append(load_compliance_pack(p) if isinstance(p, str)
                         else CompliancePack.from_dict(p))
        rules = tuple(PrivacyRule.from_dict(r) for r in data.get("rules", ()))
        det = data.get("detection")
        return cls(
            name=str(data.get("name", "privacy-policy")),
            jurisdiction=str(data.get("jurisdiction", Jurisdiction.GLOBAL.value)),
            rules=rules,
            packs=tuple(packs),
            default_action=data.get("default_action"),
            minimize=bool(data.get("minimize", False)),
            key=data.get("key"),
            key_env=data.get("key_env"),
            vault_backend=str(vault.get("backend", data.get("vault_backend", "memory"))),
            vault_path=vault.get("path", data.get("vault_path")),
            detection_config=PIIDetectionConfig(**det) if isinstance(det, dict) else None,
        )

    def effective_rules(self, jurisdiction: str | None = None) -> list[PrivacyRule]:
        """Inline rules then pack rules, filtered to the active jurisdiction."""
        juris = Jurisdiction.coerce(jurisdiction or self.jurisdiction)
        out: list[PrivacyRule] = []
        for rule in self.rules:
            if _rule_in_scope(rule, juris, default=Jurisdiction.GLOBAL.value):
                out.append(rule)
        for pack in self.packs:
            for rule in pack.rules:
                if _rule_in_scope(rule, juris, default=pack.jurisdiction):
                    out.append(rule)
        return out


# =====================================================================
# Built-in pack loading
# =====================================================================

_PACKS_DIR = Path(__file__).resolve().parent.parent / "compliance" / "packs"
_BUILTIN_PACKS = ("hipaa", "ferpa", "pci", "gdpr")
_PACK_CACHE: dict[str, CompliancePack] = {}


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ModuleNotFoundError(
                "PyYAML is required to load YAML policies/packs; install "
                "'freshdata-cleaner[enterprise]' or pyyaml, or use a .json file."
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text or "{}")


def load_compliance_pack(name_or_path: str | Path) -> CompliancePack:
    """Load a :class:`CompliancePack` by built-in name or from a YAML/JSON file.

    ``load_compliance_pack("hipaa")`` returns the bundled HIPAA pack;
    ``load_compliance_pack("/path/to/custom.yaml")`` loads an external pack.
    """
    key = str(name_or_path)
    candidate = Path(name_or_path)
    if candidate.exists() and candidate.is_file():
        return CompliancePack.from_dict(_load_yaml_or_json(candidate))
    name = key.lower()
    if name not in _BUILTIN_PACKS:
        raise ValueError(
            f"unknown compliance pack {name_or_path!r}; built-ins: {_BUILTIN_PACKS}"
        )
    if name not in _PACK_CACHE:
        _PACK_CACHE[name] = CompliancePack.from_dict(
            _load_yaml_or_json(_PACKS_DIR / f"{name}.yaml")
        )
    return _PACK_CACHE[name]


def available_packs() -> tuple[str, ...]:
    """Names of the built-in compliance packs."""
    return _BUILTIN_PACKS


def load_privacy_policy(path: str | Path) -> PrivacyPolicy:
    """Load a :class:`PrivacyPolicy` from a YAML or JSON file.

    >>> import freshdata as fd
    >>> policy = fd.load_privacy_policy("policy.yaml")   # doctest: +SKIP
    """
    return PrivacyPolicy.from_dict(_load_yaml_or_json(Path(path)))


# =====================================================================
# Classification
# =====================================================================


def _rule_in_scope(rule: PrivacyRule, juris: Jurisdiction, *, default: str) -> bool:
    scope = {s.lower() for s in (rule.jurisdictions or (default,))}
    if "global" in scope or juris.value.lower() in scope:
        return True
    # UK mirrors EU (UK GDPR), so EU-scoped rules also apply under UK.
    return juris is Jurisdiction.UK and "eu" in scope


@dataclass
class _ColumnClassification:
    column: str
    rule: PrivacyRule | None
    classification: str | None
    risk: str
    matched_by: str  # which classifier kind selected the column


def _column_sample(series: pd.Series) -> tuple[list[str], str]:
    values = [str(v) for v in series.dropna().tolist()[:_SAMPLE_ROWS]]
    return values, "\n".join(values)


def _luhn_candidate(text: str) -> bool:
    return any(_luhn_ok(m) for m in re.findall(r"(?:\d[ -]?){13,19}", text))


#: Classifier specificity — a more specific signal wins when several rules match
#: the same column (column-name beats entity beats value-regex beats context).
_CLASSIFIER_SPECIFICITY = {"column-name": 4, "entity": 3, "regex": 2, "context": 1}


def _rule_matches_column(
    rule: PrivacyRule,
    column: str,
    sample_values: list[str],
    sample_text: str,
    detected: set[str],
) -> str | None:
    """Return the *strongest* classifier kind that matched, or ``None``.

    A rule may match through several classifiers; we report the most specific so
    that, e.g., an exact column-name hit outranks a loose ``context`` keyword that
    merely appears in another column's name.
    """
    col_l = column.lower()
    if rule.requires_luhn and not _luhn_candidate(sample_text):
        return None
    # column-name classifier
    if any(col_l == c.lower() for c in rule.columns) or any(
        p.search(column) for p in rule._compiled_cols
    ):
        return "column-name"
    # entity / domain-pack classifier
    if rule.entity_types and detected & {e.upper() for e in rule.entity_types}:
        return "entity"
    # regex (value) classifier
    if rule._compiled_vals and any(
        p.search(v) for p in rule._compiled_vals for v in sample_values
    ):
        return "regex"
    # context classifier — keywords in the *surrounding data values*. Column-name
    # signals are the column-name classifier's job; matching context against the
    # name too would let a generic word (e.g. "card") over-claim a column.
    if rule.context:
        haystack = sample_text.lower()
        if any(k.lower() in haystack for k in rule.context):
            return "context"
    return None


def classify_columns(
    df: Any, policy: PrivacyPolicy, *, jurisdiction: str | None = None
) -> dict[str, _ColumnClassification]:
    """Classify each column under *policy* without mutating the data."""
    frame = to_pandas(df)
    rules = policy.effective_rules(jurisdiction)
    cfg = policy.detection_config or PIIDetectionConfig()
    result: dict[str, _ColumnClassification] = {}
    for col in frame.columns:
        sample_values, sample_text = _column_sample(frame[col])
        detected: set[str] = set()
        if sample_text:
            detected = {e.entity_type for e in detect_in_text(
                sample_text, column=str(col), config=cfg)}
        chosen: PrivacyRule | None = None
        matched_by = ""
        best_rank = -1
        for rule in rules:  # inline rules precede pack rules
            kind = _rule_matches_column(
                rule, str(col), sample_values, sample_text, detected)
            if not kind:
                continue
            rank = _CLASSIFIER_SPECIFICITY[kind]
            # most specific classifier wins; ties broken by rule order (earlier wins)
            if rank > best_rank:
                best_rank, chosen, matched_by = rank, rule, kind
        if chosen is not None:
            result[str(col)] = _ColumnClassification(
                column=str(col), rule=chosen, classification=chosen.classification,
                risk=chosen.risk, matched_by=matched_by,
            )
        elif detected:
            # detected as sensitive but no rule claimed it
            top = sorted(detected)[0]
            result[str(col)] = _ColumnClassification(
                column=str(col), rule=None,
                classification=f"detected PII: {', '.join(sorted(detected))}",
                risk=risk_level_for(top), matched_by="entity",
            )
    return result


# =====================================================================
# Application
# =====================================================================


def _resolve_key(rule: PrivacyRule | None, policy: PrivacyPolicy) -> str | None:
    for env in ((rule.key_env if rule else None), policy.key_env):
        if env and os.environ.get(env):
            return os.environ[env]
    if rule and rule.key:
        return rule.key
    return policy.key


def _resolve_vault(
    rule: PrivacyRule | None,
    policy: PrivacyPolicy,
    cache: dict[tuple[str, str | None], TokenVault],
) -> tuple[TokenVault, str]:
    backend = (
        rule.vault_backend if rule and rule.vault_backend else policy.vault_backend
    ) or "memory"
    path = rule.vault_path if rule and rule.vault_path else policy.vault_path
    key = (backend, str(path) if path else None)
    if key not in cache:
        cache[key] = make_vault(backend, path=path)
    return cache[key], backend


def _redact_cell(value: str, rule: PrivacyRule | None, cfg: PIIDetectionConfig) -> str:
    """Scrub detected spans for free text; else redact the whole cell."""
    if rule and rule.entity_types:
        hits = detect_in_text(value, config=cfg)
        if hits:
            out = value
            for e in sorted(hits, key=lambda h: h.start, reverse=True):
                out = out[: e.start] + f"<{e.entity_type}>" + out[e.end:]
            if out != value:
                return out
    return _REDACT_PLACEHOLDER


def _make_event(
    *,
    column: str,
    rule: PrivacyRule | None,
    action: Action,
    classification: str | None,
    risk: str,
    reversible: bool,
    format_preserving: bool,
    jurisdiction: str,
    n_cells: int,
    sample_masked: str,
    include_pii: bool,
    sample_original: str = "",
) -> MaskingEvent:
    pack = rule.pack if rule else None
    reason = rule.legal_basis if rule else None
    return MaskingEvent(
        column=column,
        row=None,
        entity_type=(rule.entity_types[0] if rule and rule.entity_types else "COLUMN"),
        strategy=action.value,
        reversible=reversible,
        format_preserving=format_preserving,
        hipaa_tag=None,
        gdpr_tag=None,
        risk_level=risk,
        source="policy",
        original_preview=(sample_original[:_PREVIEW_LEN] if include_pii else "<redacted>"),
        masked_preview=sample_masked[:_PREVIEW_LEN],
        metadata={"n_cells": n_cells, "matched_by": getattr(rule, "_matched_by", None)},
        rule_id=(rule.id if rule else None),
        action=action.value,
        legal_basis_or_reason=reason,
        jurisdiction=jurisdiction,
        compliance_pack=pack,
        classification=classification,
    )


def apply_privacy_policy(
    df: Any,
    policy: PrivacyPolicy,
    *,
    jurisdiction: str | None = None,
    return_report: bool = True,
    audit_include_pii: bool = False,
    vault: TokenVault | None = None,
) -> Any:
    """Apply *policy* to *df*, returning ``(df_out, PrivacyReport)`` by default.

    The input frame is never mutated. ``jurisdiction`` overrides the policy's own
    when given (handy for "what changes under EU vs US?" comparisons). Reversible
    tokenisation requires a vault and key — supply ``vault=`` to share one, or let
    the policy/rule vault settings build it; a key must come from ``key``/``key_env``.
    Report previews are redacted unless ``audit_include_pii=True``.
    """
    frame = to_pandas(df).copy()
    juris = Jurisdiction.coerce(jurisdiction or policy.jurisdiction)
    cfg = policy.detection_config or PIIDetectionConfig()
    classifications = classify_columns(frame, policy, jurisdiction=juris.value)

    events: list[MaskingEvent] = []
    violations: list[dict[str, Any]] = []
    changed_cols: list[str] = []
    cells_changed = 0
    touched: list[str] = []
    documented_raw: list[str] = []
    unprotected: list[str] = []
    quarantined: list[str] = []
    packs_used: set[str] = set()
    vault_cache: dict[tuple[str, str | None], TokenVault] = {}
    used_vault: TokenVault | None = vault
    used_backend: str | None = None  # None => infer from the vault object's type
    drop_cols: list[str] = []

    pack_by_name = {p.name: p for p in policy.packs}

    for col, cls in classifications.items():
        rule = cls.rule
        action = rule.action if rule is not None else (
            Action.coerce(policy.default_action) if policy.default_action else Action.CLASSIFY
        )
        pack = pack_by_name.get(rule.pack) if rule and rule.pack else None
        if pack is not None:
            packs_used.add(pack.name)

        # --- policy-conformance checks -------------------------------------
        if pack is not None and action.value not in pack.allowed_actions:
            violations.append({"column": col, "rule_id": rule.id if rule else None,
                               "type": "action_not_allowed",
                               "detail": f"{action.value} not allowed by pack {pack.name}"})
        if rule and rule.reversible and pack is not None and not pack.reversible_allowed:
            violations.append({"column": col, "rule_id": rule.id,
                               "type": "reversible_not_allowed",
                               "detail": f"pack {pack.name} forbids reversible tokenisation"})

        series = frame[col]
        n_cells = int(series.notna().sum())
        reversible = False
        format_preserving = False
        sample_masked = ""
        sample_original = ""

        if action is Action.CLASSIFY:
            unprotected.append(col)

        elif action is Action.PRESERVE:
            reason = (rule.legal_basis if rule else None)
            if not reason:
                violations.append({"column": col, "rule_id": rule.id if rule else None,
                                   "type": "preserve_without_reason",
                                   "detail": "preserve_with_reason requires legal_basis"})
                unprotected.append(col)
            else:
                documented_raw.append(col)

        elif action is Action.DROP:
            drop_cols.append(col)
            touched.append(col)
            cells_changed += n_cells

        elif action is Action.MINIMIZE:
            if policy.minimize:
                drop_cols.append(col)
                touched.append(col)
                cells_changed += n_cells
            else:
                # minimisation not enabled: treat as a flagged-but-kept classification
                unprotected.append(col)

        elif action is Action.QUARANTINE:
            new = series.where(series.isna(), _QUARANTINE_PLACEHOLDER)
            frame[col] = new
            quarantined.append(col)
            touched.append(col)
            changed_cols.append(col)
            cells_changed += n_cells
            sample_masked = _QUARANTINE_PLACEHOLDER

        elif action in (Action.TOKENIZE, Action.PSEUDONYMIZE, Action.REDACT):
            key = _resolve_key(rule, policy)
            new_values: list[Any] = []
            changed = 0
            tok_vault: TokenVault | None = None
            if action is Action.TOKENIZE:
                # tokenisation always needs a key (deterministic HMAC) and a vault
                # to record the mapping; reversibility just governs what we advertise.
                if not key:
                    raise ValueError(
                        f"rule {rule.id if rule else col!r}: tokenize requires a key "
                        "(rule.key / key_env or policy key)"
                    )
                if vault is not None:
                    tok_vault = vault  # caller-supplied vault wins
                else:
                    tok_vault, used_backend = _resolve_vault(rule, policy, vault_cache)
                used_vault = tok_vault
                reversible = rule is None or rule.reversible
            for value in series:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    new_values.append(value)
                    continue
                original = str(value)
                if action is Action.TOKENIZE:
                    assert tok_vault is not None and key is not None  # set just above
                    masked = tokenize_value(original, tok_vault, key, prefix="tok")
                elif action is Action.PSEUDONYMIZE:
                    if key:
                        masked, _mode = _fpe(original, key)
                    else:
                        masked = _surrogate(original, None)
                    format_preserving = True
                else:  # REDACT
                    masked = _redact_cell(original, rule, cfg)
                new_values.append(masked)
                if masked != original:
                    changed += 1
                    if not sample_masked:
                        sample_masked, sample_original = masked, original
            frame[col] = pd.Series(new_values, index=series.index)
            if changed:
                touched.append(col)
                changed_cols.append(col)
                cells_changed += changed

        events.append(_make_event(
            column=col, rule=rule, action=action, classification=cls.classification,
            risk=cls.risk, reversible=reversible, format_preserving=format_preserving,
            jurisdiction=juris.value, n_cells=n_cells, sample_masked=sample_masked,
            include_pii=audit_include_pii, sample_original=sample_original,
        ))

    if drop_cols:
        frame.drop(columns=[c for c in drop_cols if c in frame.columns], inplace=True)

    detected = list(classifications.keys())
    trust_dimension = {
        "sensitive_fields_detected": len(detected),
        "sensitive_fields_touched": len(dict.fromkeys(touched)),
        "unprotected_sensitive_fields": list(dict.fromkeys(unprotected)),
        "policy_violations": len(violations),
        "documented_raw_fields": list(dict.fromkeys(documented_raw)),
        "score": _privacy_score(len(detected), touched, documented_raw, violations),
        "fields": {
            c: {
                "classification": cl.classification,
                "action": (cl.rule.action.value if cl.rule else
                           (policy.default_action or "classify")),
                "reversible": bool(cl.rule and cl.rule.reversible
                                   and cl.rule.action is Action.TOKENIZE),
                "legal_basis": cl.rule.legal_basis if cl.rule else None,
                "compliance_pack": cl.rule.pack if cl.rule else None,
                "provable": True,  # an audit event was emitted for this column
            }
            for c, cl in classifications.items()
        },
    }

    report = PrivacyReport(
        entities_found=len(detected),
        cells_changed=cells_changed,
        columns_changed=tuple(dict.fromkeys(changed_cols + drop_cols)),
        events=events,
        metadata={"quarantined_columns": quarantined, "dropped_columns": drop_cols},
        policy_name=policy.name,
        jurisdiction=juris.value,
        compliance_pack=tuple(sorted(packs_used)),
        classifications={
            c: {"classification": cl.classification, "risk": cl.risk,
                "rule_id": cl.rule.id if cl.rule else None,
                "matched_by": cl.matched_by}
            for c, cl in classifications.items()
        },
        trust_dimension=trust_dimension,
        violations=violations,
        vault_info=vault_metadata(used_vault, used_backend) if used_vault else {},
    )
    out = from_pandas(frame, df)
    return (out, report) if return_report else out


def _privacy_score(
    detected: int,
    touched: list[str],
    documented_raw: list[str],
    violations: list[dict[str, Any]],
) -> float:
    """0–100 privacy posture: protected fraction, penalised for violations."""
    if detected == 0:
        return 100.0
    protected = len(dict.fromkeys(touched)) + len(dict.fromkeys(documented_raw))
    base = 100.0 * min(protected, detected) / detected
    base -= 10.0 * len(violations)
    return round(max(0.0, min(100.0, base)), 1)


# =====================================================================
# Reversible detokenisation API (explicit vault + key required)
# =====================================================================


def detokenize_series(
    series: Any, vault: TokenVault, key: str, *, prefix: str = "tok"
) -> pd.Series:
    """Reverse tokenised values in *series* using *vault* and *key*.

    Reversal is only possible with **both** the vault that holds the mapping and
    a non-empty key (the same key that produced the tokens). Non-token values pass
    through unchanged. Tokens absent from the vault are left as-is.
    """
    if not key:
        raise ValueError("detokenize_series requires a non-empty key")
    s = pd.Series(series)

    def _rev(value: Any) -> Any:
        if isinstance(value, str) and value.startswith(f"{prefix}_"):
            try:
                return detokenize_value(value, vault, key)
            except KeyError:
                return value
        return value

    return s.map(_rev)
