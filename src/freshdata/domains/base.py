"""Domain validator interface and the shared config-driven rule engine.

A *domain pack* teaches :func:`freshdata.clean` how to validate and repair a
specific kind of tabular data (a finance ledger, a GS1 product catalog, a GTFS
feed, …). Every pack implements :class:`DomainValidator`; most subclass
:class:`ConfigDrivenValidator`, which runs rules declared in a ``rules.yaml``
file through five ordered layers (schema → format → reference → business →
semantic) and reports findings without ever touching the data silently.

This module depends only on the standard library, pandas, and numpy — loading a
pack's YAML rules is the only step that needs an extra dependency (PyYAML), and
that import is deferred until a pack is actually used.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

#: Validation layers, executed in this strict order.
LAYERS: tuple[str, ...] = ("schema", "format", "reference", "business", "semantic")
#: Finding severities, in increasing order of seriousness.
SEVERITIES: tuple[str, ...] = ("info", "warning", "error")
#: Repair strategies a rule may declare.
REPAIR_STRATEGIES: tuple[str, ...] = ("fill_default", "coerce", "flag_only", "reject", "none")

#: Stable finding code emitted when a required canonical field is absent.
MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"

#: Maps a finding severity onto the core :class:`~freshdata.report.Action` risk levels.
SEVERITY_TO_RISK: dict[str, str] = {"error": "high", "warning": "medium", "info": "low"}
#: Trust-score penalty weight per severity (per fully-violated rule).
SEVERITY_WEIGHT: dict[str, float] = {"error": 1.0, "warning": 0.25, "info": 0.05}


class DomainError(ValueError):
    """Base class for domain-pack errors."""


class SkipCheck(Exception):  # noqa: N818 - control-flow signal, not an error condition
    """Raised by a custom check to mark its rule as skipped (not a false pass).

    Used when a rule cannot apply in the current context — e.g. a GTFS cross-file
    reference check when the referenced file is not part of a single-file run.
    """


@dataclass
class ColumnMapping:
    """Resolved mapping from a pack's canonical field names to real columns.

    ``mapped`` holds ``canonical -> actual_column``. ``unmapped_required`` lists
    canonical fields the pack marked required that no column matched.
    ``log`` records, per canonical field, how the match was made (exact,
    case-insensitive, regex, override) or that it was missing — nothing is ever
    guessed silently.
    """

    mapped: dict[str, str] = field(default_factory=dict)
    unmapped_required: list[str] = field(default_factory=list)
    log: list[dict[str, str]] = field(default_factory=list)

    def actual(self, canonical: str) -> str | None:
        """Real column name for *canonical*, or ``None`` if unmapped."""
        return self.mapped.get(canonical)

    def is_mapped(self, canonical: str) -> bool:
        return canonical in self.mapped

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapped": dict(self.mapped),
            "unmapped_required": list(self.unmapped_required),
            "log": list(self.log),
        }


@dataclass
class RuleResult:
    """Outcome of evaluating one rule against a frame."""

    rule_id: str
    name: str
    layer: str
    severity: str
    fields: tuple[str, ...]
    check: str
    status: str  # "passed" | "violated" | "skipped"
    n_violations: int = 0
    violation_rows: list[Any] = field(default_factory=list)
    message: str = ""
    repair: str = "none"

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def violated(self) -> bool:
        return self.status == "violated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "layer": self.layer,
            "severity": self.severity,
            "fields": list(self.fields),
            "check": self.check,
            "status": self.status,
            "n_violations": self.n_violations,
            "violation_rows": [_json_safe(r) for r in self.violation_rows],
            "message": self.message,
            "repair": self.repair,
        }


@dataclass
class ValidationReport:
    """Per-rule results plus a 0–1 ``domain_trust_score`` for one validation run.

    This is distinct from (and unrelated to) the optional enterprise layer's
    ``ValidationReport``; the two never interact.
    """

    domain: str
    version: str
    schema_version: str
    results: list[RuleResult] = field(default_factory=list)
    mapping: ColumnMapping = field(default_factory=ColumnMapping)
    domain_trust_score: float = 1.0

    @property
    def errors(self) -> list[RuleResult]:
        return [r for r in self.results if r.violated and r.severity == "error"]

    @property
    def warnings(self) -> list[RuleResult]:
        return [r for r in self.results if r.violated and r.severity == "warning"]

    @property
    def severity_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(SEVERITIES, 0)
        for r in self.results:
            if r.violated:
                counts[r.severity] += 1
        return counts

    @property
    def violation_index(self) -> dict[str, list[Any]]:
        """Map each violated rule id to the row labels it flagged."""
        return {r.rule_id: list(r.violation_rows) for r in self.results if r.violated}

    @property
    def passed(self) -> bool:
        """True when no error-severity rule was violated."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "version": self.version,
            "schema_version": self.schema_version,
            "domain_trust_score": self.domain_trust_score,
            "severity_counts": self.severity_counts,
            "mapping": self.mapping.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        counts = self.severity_counts
        return (
            f"freshdata domain={self.domain} v{self.version} "
            f"(schema {self.schema_version}): trust={self.domain_trust_score:.2f}, "
            f"{counts['error']} error(s), {counts['warning']} warning(s), "
            f"{counts['info']} info"
        )


@dataclass
class RepairAction:
    """One repair attempt, recorded whether or not it changed a value."""

    rule_id: str
    strategy: str
    column: str | None
    row: Any
    from_value: Any
    to_value: Any
    status: str  # "applied" | "flagged" | "unresolvable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "strategy": self.strategy,
            "column": self.column,
            "row": _json_safe(self.row),
            "from": _json_safe(self.from_value),
            "to": _json_safe(self.to_value),
            "status": self.status,
        }


@dataclass
class RepairLog:
    """Ordered log of every repair attempt for one ``repair`` call."""

    actions: list[RepairAction] = field(default_factory=list)

    def add(self, action: RepairAction) -> None:
        self.actions.append(action)

    @property
    def applied(self) -> list[RepairAction]:
        return [a for a in self.actions if a.status == "applied"]

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {"actions": [a.to_dict() for a in self.actions]}


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of a scalar to a JSON-friendly Python value."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError, TypeError):
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


class DomainValidator(ABC):
    """Common interface every domain pack implements.

    A validator is created per cleaning run and is *not* thread-safe: it caches
    the column mapping from :meth:`detect_columns` so :meth:`repair` can reuse
    what :meth:`validate` resolved.
    """

    domain_name: str = ""
    version: str = "0.0.0"
    schema_version: str = ""
    #: True for packs that validate a feed of related frames (e.g. GTFS), which
    #: accept dict input / a ``gtfs_file`` selector via :func:`freshdata.clean`.
    multi_frame: bool = False

    @abstractmethod
    def detect_columns(self, df: pd.DataFrame) -> ColumnMapping:
        """Resolve canonical field names to real columns in *df*."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run all rules against *df* and return findings (never mutates *df*)."""

    @abstractmethod
    def repair(self, df: pd.DataFrame, report: ValidationReport) -> tuple[pd.DataFrame, RepairLog]:
        """Apply each rule's repair strategy, returning a new frame and an audit log."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the pack (for the audit trail)."""


# Signatures for pack-supplied extension functions.
CheckFn = Callable[[pd.DataFrame, ColumnMapping, "Rule"], "list[Any]"]
RepairFn = Callable[[pd.DataFrame, ColumnMapping, "Rule", RuleResult], "dict[Any, Any]"]


@dataclass(frozen=True)
class Rule:
    """One parsed rule from a pack's ``rules.yaml``."""

    id: str
    name: str
    layer: str
    severity: str
    fields: tuple[str, ...]
    check: str
    params: Mapping[str, Any] = field(default_factory=dict)
    repair: str = "none"
    repair_params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Rule:
        missing_keys = [key for key in ("id", "layer", "severity", "check") if key not in raw]
        if missing_keys:
            raise DomainError(f"rule is missing required key(s): {', '.join(missing_keys)}")
        if "field" in raw and "fields" in raw:
            raise DomainError(f"rule {raw.get('id')!r}: use either 'field' or 'fields', not both")
        fields = raw.get("fields") or ([raw["field"]] if raw.get("field") else [])
        if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
            raise DomainError(f"rule {raw.get('id')!r}: 'fields' must be a list")
        rule = cls(
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            layer=str(raw["layer"]),
            severity=str(raw["severity"]),
            fields=tuple(fields),
            check=str(raw["check"]),
            params=dict(raw.get("params") or {}),
            repair=str(raw.get("repair", "none")),
            repair_params=dict(raw.get("repair_params") or {}),
        )
        if rule.layer not in LAYERS:
            raise DomainError(
                f"rule {rule.id!r}: layer must be one of {LAYERS}, got {rule.layer!r}"
            )
        if rule.severity not in SEVERITIES:
            raise DomainError(
                f"rule {rule.id!r}: severity must be one of {SEVERITIES}, got {rule.severity!r}"
            )
        if rule.repair not in REPAIR_STRATEGIES:
            raise DomainError(
                f"rule {rule.id!r}: repair must be one of {REPAIR_STRATEGIES}, got {rule.repair!r}"
            )
        return rule


def _load_yaml_rules(path: str) -> list[dict[str, Any]]:
    """Load a ``rules.yaml`` file, deferring the PyYAML import until needed."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via test monkeypatch
        raise ImportError(
            "freshdata domain packs need PyYAML to read their rule files. "
            "Install it with: pip install 'freshdata-cleaner[domains]'"
        ) from exc
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if isinstance(data, list):
        rules = data
    elif isinstance(data, Mapping):
        rules = data.get("rules", [])
    else:
        raise DomainError(f"{path}: YAML root must be a mapping or a list of rules")
    if not isinstance(rules, list):
        raise DomainError(f"{path}: 'rules' must be a list")
    if not all(isinstance(rule, Mapping) for rule in rules):
        raise DomainError(f"{path}: every rule must be a mapping")
    return [dict(rule) for rule in rules]


class ConfigDrivenValidator(DomainValidator):
    """A :class:`DomainValidator` whose rules live in a ``rules.yaml`` file.

    Subclasses declare the pack's canonical fields, which are required, which are
    identifiers (never repaired), regex aliases for column detection, the path to
    ``rules.yaml``, and any custom check/repair functions. The layered execution,
    standard checks, trust scoring, and ID-safe repair are all handled here.
    """

    #: Canonical field names this pack knows about.
    canonical_fields: tuple[str, ...] = ()
    #: Subset of :attr:`canonical_fields` that must be present.
    required_fields: tuple[str, ...] = ()
    #: Subset that identifies entities; these columns are never repaired.
    id_fields: tuple[str, ...] = ()
    #: Regex aliases per canonical field, matched against real column names.
    aliases: Mapping[str, Sequence[str]] = {}
    #: Absolute path to the pack's ``rules.yaml``.
    rules_path: str = ""

    def __init__(self, *, column_map: Mapping[str, str] | None = None) -> None:
        self._column_map = dict(column_map or {})
        self._rules: list[Rule] | None = None
        self._mapping: ColumnMapping | None = None
        self._checks: dict[str, CheckFn] = {}
        self._repairs: dict[str, RepairFn] = {}
        self.register_extensions()

    # -- pack extension points ---------------------------------------------

    def register_extensions(self) -> None:
        """Hook for subclasses to register custom check/repair functions."""

    def register_check(self, name: str, fn: CheckFn) -> None:
        self._checks[name] = fn

    def register_repair(self, name: str, fn: RepairFn) -> None:
        self._repairs[name] = fn

    def reference_sources(self) -> list[dict[str, Any]]:
        """Optional ``_meta`` blocks for bundled reference data (audit trail)."""
        return []

    # -- rules --------------------------------------------------------------

    @property
    def rules(self) -> list[Rule]:
        if self._rules is None:
            if not self.rules_path:
                raise DomainError(f"{type(self).__name__} does not set rules_path")
            self._rules = [Rule.from_dict(r) for r in _load_yaml_rules(self.rules_path)]
            self._validate_rules(self._rules)
        return self._rules

    def _validate_rules(self, rules: Sequence[Rule]) -> None:
        """Reject ambiguous rule sets before validation starts."""
        if not rules:
            raise DomainError(f"{type(self).__name__} defines no validation rules")
        seen: set[str] = set()
        for rule in rules:
            if rule.id in seen:
                raise DomainError(f"duplicate rule id {rule.id!r}")
            seen.add(rule.id)

    # -- column detection ---------------------------------------------------

    def detect_columns(self, df: pd.DataFrame) -> ColumnMapping:
        mapping = ColumnMapping()
        columns = [str(c) for c in df.columns]
        lower_index: dict[str, str] = {}
        for col in columns:
            lower_index.setdefault(col.casefold(), col)
        inverted_override = {v: k for k, v in self._column_map.items()}
        for canonical in self.canonical_fields:
            actual, method = self._match_one(canonical, columns, lower_index, inverted_override)
            if actual is not None:
                mapping.mapped[canonical] = actual
                mapping.log.append({"canonical": canonical, "actual": actual, "method": method})
            else:
                mapping.log.append({"canonical": canonical, "actual": "", "method": "missing"})
                if canonical in self.required_fields:
                    mapping.unmapped_required.append(canonical)
        self._mapping = mapping
        return mapping

    def _match_one(
        self,
        canonical: str,
        columns: list[str],
        lower_index: dict[str, str],
        inverted_override: dict[str, str],
    ) -> tuple[str | None, str]:
        # 1. explicit user override (column_map maps actual -> canonical).
        override = inverted_override.get(canonical)
        if override is not None and override in columns:
            return override, "override"
        # 2. exact match on the canonical name.
        if canonical in columns:
            return canonical, "exact"
        # 3. case-insensitive match.
        ci = lower_index.get(canonical.casefold())
        if ci is not None:
            return ci, "case_insensitive"
        # 4. regex aliases declared by the pack.
        for pattern in self.aliases.get(canonical, ()):  # noqa: SIM110 - need the matched value
            for col in columns:
                if re.fullmatch(pattern, col, flags=re.IGNORECASE):
                    return col, "regex"
        return None, "missing"

    # -- validation ---------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        mapping = self.detect_columns(df)
        report = ValidationReport(
            domain=self.domain_name,
            version=self.version,
            schema_version=self.schema_version,
            mapping=mapping,
        )
        ordered = sorted(self.rules, key=lambda r: LAYERS.index(r.layer))
        for rule in ordered:
            report.results.append(self._run_rule(df, mapping, rule))
        report.domain_trust_score = self._trust_score(df, report.results)
        return report

    def _run_rule(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> RuleResult:
        result = RuleResult(
            rule_id=rule.id, name=rule.name, layer=rule.layer, severity=rule.severity,
            fields=rule.fields, check=rule.check, status="passed", repair=rule.repair,
        )
        # Presence checks operate on the mapping itself.
        if rule.check in ("required", "not_null"):
            rows = self._check_presence(df, mapping, rule, result)
        else:
            # Custom checks may span several fields; builtin checks target the
            # first. Either way, skip the rule if a needed column is absent —
            # the schema layer already reported it as MISSING_REQUIRED_FIELD.
            needed = rule.fields if rule.check == "custom" else rule.fields[:1]
            absent = [f for f in needed if not mapping.is_mapped(f)]
            if absent:
                result.status = "skipped"
                result.message = f"skipped: {', '.join(absent)} not present"
                return result
            try:
                rows = self._dispatch_check(df, mapping, rule)
            except SkipCheck as skip:
                result.status = "skipped"
                result.message = f"skipped: {skip}" if str(skip) else "skipped"
                return result
        if rows:
            result.status = "violated"
            result.violation_rows = list(rows)
            result.n_violations = len(rows)
            if not result.message:
                result.message = f"{len(rows)} row(s) violate {rule.id}"
        return result

    def _check_presence(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule, result: RuleResult
    ) -> list[Any]:
        missing = [f for f in rule.fields if not mapping.is_mapped(f)]
        if missing:
            result.message = f"{MISSING_REQUIRED_FIELD}: {', '.join(missing)}"
            # A whole-column miss is represented as a single table-level finding.
            return [MISSING_REQUIRED_FIELD]
        if rule.check == "not_null":
            columns = [mapping.actual(field_name) for field_name in rule.fields]
            rows = df.index[df[columns].isna().any(axis=1)].tolist()
            if rows:
                result.message = f"{len(rows)} null value(s) in {', '.join(rule.fields)}"
            return rows
        return []

    def _dispatch_check(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        builtin = {
            "regex": self._check_regex,
            "enum": self._check_enum,
            "reference": self._check_enum,
            "range": self._check_range,
            "unique": self._check_unique,
        }
        if rule.check in builtin:
            return builtin[rule.check](df, mapping, rule)
        if rule.check == "custom":
            func = str(rule.params.get("func", ""))
            fn = self._checks.get(func)
            if fn is None:
                raise DomainError(f"rule {rule.id!r}: unknown custom check {func!r}")
            return list(fn(df, mapping, rule))
        raise DomainError(f"rule {rule.id!r}: unknown check {rule.check!r}")

    def _check_regex(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        col = mapping.actual(rule.fields[0])
        pattern = str(rule.params["pattern"])
        series = df[col]
        present = series.notna()
        as_text = series.astype("string")
        matches = as_text.str.fullmatch(pattern)
        bad = present & ~matches.fillna(False)
        return df.index[bad].tolist()

    def _check_enum(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        col = mapping.actual(rule.fields[0])
        allowed = self._allowed_values(rule)
        series = df[col]
        present = series.notna()
        if rule.params.get("case_insensitive"):
            allowed = {str(v).casefold() for v in allowed}
            normalized = series.astype("string").str.casefold()
            bad = present & ~normalized.isin(allowed)
        else:
            bad = present & ~series.isin(allowed)
        return df.index[bad].tolist()

    def _check_range(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        col = mapping.actual(rule.fields[0])
        numeric = pd.to_numeric(df[col], errors="coerce")
        present = df[col].notna()
        low = rule.params.get("min")
        high = rule.params.get("max")
        bad = present & numeric.isna()  # non-numeric where a value exists
        if low is not None:
            bad = bad | (present & (numeric < float(low)))
        if high is not None:
            bad = bad | (present & (numeric > float(high)))
        return df.index[bad].tolist()

    def _check_unique(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        col = mapping.actual(rule.fields[0])
        series = df[col]
        present = series.notna()
        dup = present & series.duplicated(keep=False)
        return df.index[dup].tolist()

    def _allowed_values(self, rule: Rule) -> set[Any]:
        if "values" in rule.params:
            return set(rule.params["values"])
        ref = rule.params.get("reference")
        if ref:
            return set(self.load_reference_values(str(ref)))
        raise DomainError(f"rule {rule.id!r}: enum/reference needs 'values' or 'reference'")

    def load_reference_values(self, name: str) -> Iterable[Any]:
        """Override to supply reference code sets bundled with the pack."""
        raise DomainError(f"no reference data named {name!r}")

    # -- trust score --------------------------------------------------------

    def _trust_score(self, df: pd.DataFrame, results: list[RuleResult]) -> float:
        n_rows = max(len(df), 1)
        penalty = 0.0
        for r in results:
            if not r.violated:
                continue
            # Table-level findings (e.g. a missing column) count as a full fraction.
            row_like = [x for x in r.violation_rows if x != MISSING_REQUIRED_FIELD]
            frac = 1.0 if len(row_like) != len(r.violation_rows) else len(row_like) / n_rows
            penalty += SEVERITY_WEIGHT.get(r.severity, 0.0) * min(frac, 1.0)
        return round(max(0.0, min(1.0, 1.0 - penalty)), 4)

    # -- repair -------------------------------------------------------------

    def _protected_columns(self, mapping: ColumnMapping) -> set[str]:
        protected = {mapping.actual(f) for f in self.id_fields if mapping.is_mapped(f)}
        return {c for c in protected if c is not None}

    def repair(self, df: pd.DataFrame, report: ValidationReport) -> tuple[pd.DataFrame, RepairLog]:
        mapping = report.mapping
        out = df.copy(deep=True)
        log = RepairLog()
        protected = self._protected_columns(mapping)
        for result in report.results:
            if not result.violated or result.repair in ("none", "flag_only"):
                if result.violated:
                    self._log_flag(log, mapping, result)
                continue
            self._apply_repair(out, mapping, report, result, protected, log)
        return out, log

    def _log_flag(self, log: RepairLog, mapping: ColumnMapping, result: RuleResult) -> None:
        col = mapping.actual(result.fields[0]) if result.fields else None
        for row in result.violation_rows:
            log.add(RepairAction(result.rule_id, "flag_only", col, row, None, None, "flagged"))

    def _apply_repair(
        self,
        out: pd.DataFrame,
        mapping: ColumnMapping,
        report: ValidationReport,
        result: RuleResult,
        protected: set[str],
        log: RepairLog,
    ) -> None:
        rule = self._rule_by_id(result.rule_id)
        col = mapping.actual(result.fields[0]) if result.fields else None
        # Identifier columns are never imputed or dropped (fill_default/reject);
        # an explicit, audited ``coerce`` may still normalize their representation
        # (e.g. stripping formatting from a GTIN). Absent columns are always flagged.
        id_protected = col in protected and rule.repair in ("fill_default", "reject")
        if col is None or id_protected:
            for row in result.violation_rows:
                log.add(RepairAction(result.rule_id, rule.repair, col, row, None, None, "flagged"))
            return
        if rule.repair == "reject":
            self._repair_reject(out, result, col, log)
        elif rule.repair == "fill_default":
            self._repair_fill(out, rule, result, col, log)
        elif rule.repair == "coerce":
            self._repair_coerce(out, mapping, rule, result, col, log)

    def _repair_reject(
        self, out: pd.DataFrame, result: RuleResult, col: str, log: RepairLog
    ) -> None:
        for row in result.violation_rows:
            if row in out.index:
                log.add(RepairAction(result.rule_id, "reject", col, row,
                                     _json_safe(out.at[row, col]), None, "applied"))
                out.drop(index=row, inplace=True)

    def _repair_fill(
        self, out: pd.DataFrame, rule: Rule, result: RuleResult, col: str, log: RepairLog
    ) -> None:
        default = rule.repair_params.get("value")
        for row in result.violation_rows:
            if row not in out.index:
                continue
            old = out.at[row, col]
            if pd.isna(old):
                out.at[row, col] = default
                log.add(RepairAction(result.rule_id, "fill_default", col, row,
                                     _json_safe(old), _json_safe(default), "applied"))
            else:
                log.add(RepairAction(result.rule_id, "fill_default", col, row,
                                     _json_safe(old), None, "unresolvable"))

    def _repair_coerce(
        self, out: pd.DataFrame, mapping: ColumnMapping, rule: Rule,
        result: RuleResult, col: str, log: RepairLog,
    ) -> None:
        func = str(rule.repair_params.get("func", ""))
        fn = self._repairs.get(func)
        new_values = fn(out, mapping, rule, result) if fn is not None else {}
        for row in result.violation_rows:
            old = out.at[row, col] if row in out.index else None
            if row in new_values and new_values[row] is not None:
                out.at[row, col] = new_values[row]
                log.add(RepairAction(result.rule_id, "coerce", col, row,
                                     _json_safe(old), _json_safe(new_values[row]), "applied"))
            else:
                log.add(RepairAction(result.rule_id, "coerce", col, row,
                                     _json_safe(old), None, "unresolvable"))

    def _rule_by_id(self, rule_id: str) -> Rule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise DomainError(f"unknown rule id {rule_id!r}")

    # -- description --------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "domain": self.domain_name,
            "version": self.version,
            "schema_version": self.schema_version,
            "canonical_fields": list(self.canonical_fields),
            "required_fields": list(self.required_fields),
            "id_fields": list(self.id_fields),
            "rules": [{"id": r.id, "name": r.name, "layer": r.layer, "severity": r.severity}
                      for r in self.rules],
            "reference_sources": self.reference_sources(),
        }
