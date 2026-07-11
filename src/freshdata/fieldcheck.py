"""Context-aware, per-cell field validation with configurable remediation.

``fd.validate_fields(df, schema=..., policy=...)`` decides — cell by cell —
whether a value is valid *for its field*, not merely parseable. The same
string can be fine in one column and wrong in another (``"Apple"`` is a
plausible ``company_name``, an invalid ``transaction_amount`` and a malformed
``stock_ticker``), so every check combines:

* the declared :class:`FieldSpec` (expected semantic type, range, pattern,
  vocabulary, reference lookup, null markers);
* what the value itself looks like (number / date / email / URL / free text);
* the *column consensus* — the dominant shape of the other values — so a lone
  ``"apple"`` inside a numeric column is caught even without a schema.

Distinct failure classes are kept apart (a parse failure is not a statistical
outlier is not a rare category), and a :class:`RemediationPolicy` maps each
class to an action. The default policy is non-destructive: nothing is deleted,
nulled or "corrected" silently; :func:`apply_field_policy` splits the frame
into accepted / quarantined / rejected copies plus a full audit trail, leaving
the input untouched.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .findings import QualityFinding
from .semantic.experts import is_plain_number, looks_like_date_value, parse_currency
from .textclean import TextCleanConfig, clean_text_value, config_for_field

__all__ = [
    "FieldSpec",
    "RemediationPolicy",
    "CellIssue",
    "FieldValidationReport",
    "PolicyResult",
    "validate_fields",
    "apply_field_policy",
    "CLASSIFICATIONS",
    "ACTIONS",
]

#: Distinct failure classes — never collapsed into one generic "outlier".
CLASSIFICATIONS = (
    "parse_failure",          # value cannot be parsed as the expected type
    "semantic_mismatch",      # parseable text, but the wrong kind of thing
    "domain_mismatch",        # right shape, wrong for this domain field
    "schema_violation",       # nullability / required / allowed-values breach
    "statistical_outlier",    # numeric value far outside the column's spread
    "categorical_rare",       # unseen or rare category — possibly valid
    "cross_field_inconsistency",  # row fails a relationship between columns
)

#: Configurable remediation actions, least to most severe.
ACTIONS = (
    "accept",
    "accept_with_warning",
    "normalize",
    "replace_with_null",
    "quarantine",
    "reject",
    "manual_review",
)

_DEFAULT_NULL_MARKERS = frozenset({"", "n/a", "na", "null", "none", "nan", "-", "--"})

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_URL_RE = re.compile(r"^https?://\S+\.\S+", re.I)
_TICKER_RE = re.compile(r"^[A-Z]{1,6}([.\-][A-Z0-9]{1,4})?$")
_PHONE_RE = re.compile(r"^\+?[\d\s\-().]{7,17}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/]*$")


def _safe_fullmatch(pattern: str, value: str) -> bool:
    """``re.fullmatch`` that never raises.

    ``FieldSpec.pattern`` is caller-supplied; an invalid regex degrades to
    "nothing matches" (surfaced as a normal domain_mismatch) instead of
    crashing validation mid-run.
    """
    try:
        return re.fullmatch(pattern, value) is not None
    except re.error:
        return False


#: Semantic types validate_fields understands. ``numeric``-family types parse
#: through the same path; anything else falls back to pattern/vocabulary rules.
_NUMERIC_TYPES = frozenset(
    {"numeric", "integer", "float", "currency_amount", "rate", "percentage"})
_DATE_TYPES = frozenset({"date", "datetime", "date_like"})


def detect_value_type(value: Any) -> str:
    """Cheap, deterministic shape probe for one scalar."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "null"
    if isinstance(value, bool):
        return "boolean_like"
    if isinstance(value, (int, float)):
        return "numeric"
    if not isinstance(value, str):
        return "other"
    s = value.strip()
    if not s:
        return "null"
    if is_plain_number(s) or parse_currency(s) is not None:
        return "numeric"
    if looks_like_date_value(s):
        return "date_like"
    if _EMAIL_RE.match(s):
        return "email"
    if _URL_RE.match(s):
        return "url"
    if _PHONE_RE.match(s) and sum(c.isdigit() for c in s) >= 7:
        return "phone"
    return "text"


@dataclass(frozen=True)
class FieldSpec:
    """Declared expectations for one column.

    Only ``semantic_type`` is usually needed; everything else refines it.
    ``reference`` and ``suggest`` are injection points for external knowledge
    (e.g. a ticker universe) — nothing domain-specific is hard-coded here.
    """

    semantic_type: str | None = None
    required: bool = False           #: column must be present and non-null
    nullable: bool = True
    allowed_values: frozenset | None = None
    pattern: str | None = None       #: full-match regex on the (cleaned) string
    min_value: float | None = None
    max_value: float | None = None
    max_length: int | None = None
    null_markers: frozenset = _DEFAULT_NULL_MARKERS  #: field-specific missing codes
    reference: Callable[[str], bool] | Collection[str] | None = None
    suggest: Callable[[str], str | None] | Mapping[str, str] | None = None

    def is_null_marker(self, s: str) -> bool:
        return s.strip().casefold() in self.null_markers

    def in_reference(self, s: str) -> bool | None:
        """True/False when a reference source is configured, else ``None``."""
        if self.reference is None:
            return None
        if callable(self.reference):
            return bool(self.reference(s))
        return s in self.reference

    def suggestion_for(self, s: str) -> str | None:
        """A trusted correction for ``s``, or ``None``. Never invented."""
        if self.suggest is None:
            return None
        if callable(self.suggest):
            return self.suggest(s)
        return self.suggest.get(s) or self.suggest.get(s.strip().casefold())


@dataclass(frozen=True)
class RemediationPolicy:
    """Maps each failure classification to an action from :data:`ACTIONS`.

    Defaults are deliberately conservative: nothing destructive, statistical
    outliers and rare categories are *warned about*, never auto-rejected.
    """

    parse_failure: str = "quarantine"
    semantic_mismatch: str = "quarantine"
    domain_mismatch: str = "manual_review"
    schema_violation: str = "reject"
    statistical_outlier: str = "accept_with_warning"
    categorical_rare: str = "accept_with_warning"
    cross_field_inconsistency: str = "manual_review"
    normalize_text: bool = True  #: run safe text cleaning before checks

    def __post_init__(self) -> None:
        for cls in CLASSIFICATIONS:
            action = getattr(self, cls)
            if action not in ACTIONS:
                raise ValueError(
                    f"invalid action {action!r} for {cls!r}; expected one of {ACTIONS}"
                )

    def action_for(self, classification: str) -> str:
        return getattr(self, classification)


_SEVERITY_BY_CLASS = {
    "parse_failure": "error",
    "semantic_mismatch": "error",
    "domain_mismatch": "error",
    "schema_violation": "error",
    "statistical_outlier": "warning",
    "categorical_rare": "warning",
    "cross_field_inconsistency": "warning",
}


@dataclass(frozen=True)
class CellIssue:
    """One per-cell validation problem, with everything needed to act on it."""

    row: Any
    column: str
    original: Any
    cleaned: Any
    classification: str
    severity: str
    reason: str
    expected: str
    detected: str
    confidence: float
    action: str
    rule: str
    suggestion: str | None = None
    transforms: tuple = ()

    def to_dict(self) -> dict:
        return {
            "row": self.row, "column": self.column,
            "original": self.original, "cleaned": self.cleaned,
            "classification": self.classification, "severity": self.severity,
            "reason": self.reason, "expected": self.expected,
            "detected": self.detected, "confidence": round(self.confidence, 4),
            "action": self.action, "rule": self.rule,
            "suggestion": self.suggestion, "transforms": list(self.transforms),
        }

    def to_finding(self) -> QualityFinding:
        return QualityFinding.create(
            severity=self.severity,
            step="fieldcheck",
            rule_name=f"{self.classification}:{self.rule}",
            message=self.reason,
            column=self.column,
            row_index=self.row,
            observed_value=self.original,
            expected_condition=self.expected,
            action_taken=self.action,
            extra={
                "detected": self.detected,
                "confidence": round(self.confidence, 4),
                **({"suggestion": self.suggestion} if self.suggestion else {}),
            },
        )


@dataclass
class FieldValidationReport:
    """All issues plus per-row decisions from one :func:`validate_fields` run."""

    issues: list = field(default_factory=list)
    n_rows: int = 0
    columns_checked: list = field(default_factory=list)
    inferred_types: dict = field(default_factory=dict)
    normalized_cells: list = field(default_factory=list)  #: audit of text normalizations

    def __len__(self) -> int:
        return len(self.issues)

    def __bool__(self) -> bool:
        return bool(self.issues)

    def by_classification(self) -> dict:
        out: dict = {}
        for i in self.issues:
            out.setdefault(i.classification, []).append(i)
        return out

    def to_frame(self) -> pd.DataFrame:
        cols = ["row", "column", "original", "cleaned", "classification", "severity",
                "reason", "expected", "detected", "confidence", "action", "rule",
                "suggestion", "transforms"]
        return pd.DataFrame([i.to_dict() for i in self.issues], columns=cols)

    def to_findings(self) -> list:
        return [i.to_finding() for i in self.issues]

    def row_actions(self) -> dict:
        """Most severe action per row (severity = position in :data:`ACTIONS`)."""
        order = {a: n for n, a in enumerate(ACTIONS)}
        out: dict = {}
        for i in self.issues:
            if i.row is None:
                continue
            cur = out.get(i.row)
            if cur is None or order[i.action] > order[cur]:
                out[i.row] = i.action
        return out

    def summary(self) -> str:
        lines = [
            f"freshdata field check — {len(self.issues)} issue(s) over {self.n_rows} row(s), "
            f"{len(self.columns_checked)} column(s)"
        ]
        for cls, items in sorted(self.by_classification().items()):
            lines.append(f"  {cls}: {len(items)}")
            for i in items[:3]:
                lines.append(
                    f"    row {i.row} {i.column}={i.original!r}: {i.reason} -> {i.action}"
                )
        if not self.issues:
            lines.append("  all values consistent with their fields")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _parse_numeric(s: str) -> float | None:
    if is_plain_number(s):
        return float(str(s).strip().replace(",", ""))
    return parse_currency(s)


def _check_value(
    col: str,
    row: Any,
    raw: Any,
    cleaned: Any,
    transforms: tuple,
    spec: FieldSpec,
    policy: RemediationPolicy,
) -> CellIssue | None:
    """Validate one cell against its spec. Returns an issue or ``None``."""
    expected = spec.semantic_type or "any"
    s = str(cleaned).strip() if cleaned is not None else ""
    detected = detect_value_type(cleaned)

    def issue(classification: str, reason: str, rule: str, *,
              confidence: float = 1.0, suggestion: str | None = None,
              action: str | None = None) -> CellIssue:
        return CellIssue(
            row=row, column=col, original=raw, cleaned=cleaned,
            classification=classification,
            severity=_SEVERITY_BY_CLASS[classification],
            reason=reason, expected=expected, detected=detected,
            confidence=confidence,
            action=action or policy.action_for(classification),
            rule=rule, suggestion=suggestion, transforms=transforms,
        )

    # -- missing handling -----------------------------------------------------
    is_missing = raw is None or (not isinstance(raw, str) and pd.isna(raw)) or (
        isinstance(cleaned, str) and spec.is_null_marker(s))
    if is_missing:
        if not spec.nullable or spec.required:
            return issue(
                "schema_violation",
                f"{col} is required but the value is missing "
                f"({raw!r} is a configured null marker)" if raw is not None
                else f"{col} is required but the value is missing",
                "not_null",
            )
        return None

    # -- expected-type parse & range ------------------------------------------
    if spec.semantic_type in _NUMERIC_TYPES:
        num = cleaned if isinstance(cleaned, (int, float)) and not isinstance(cleaned, bool) \
            else _parse_numeric(s)
        if num is None:
            textish = ("text", "email", "url", "phone", "date_like")
            cls = "semantic_mismatch" if detected in textish else "parse_failure"
            return issue(
                cls,
                f"expected {expected} in {col!r} but got {detected} value {s!r}; "
                "not silently converted",
                "numeric_parse",
            )
        if spec.min_value is not None and num < spec.min_value:
            return issue(
                "domain_mismatch",
                f"{col}={num} below configured minimum {spec.min_value}", "min_value")
        if spec.max_value is not None and num > spec.max_value:
            return issue(
                "domain_mismatch",
                f"{col}={num} above configured maximum {spec.max_value}", "max_value")
        return None

    if spec.semantic_type in _DATE_TYPES:
        ts = pd.to_datetime(s if isinstance(cleaned, str) else cleaned, errors="coerce")
        if pd.isna(ts):
            if looks_like_date_value(s):
                return issue(
                    "parse_failure",
                    f"{s!r} is shaped like a date but is not a real calendar date",
                    "impossible_date",
                )
            return issue(
                "semantic_mismatch",
                f"expected {expected} in {col!r} but got {detected} value {s!r}",
                "date_parse",
            )
        return None

    # -- pattern / vocabulary / reference fields --------------------------------
    if spec.allowed_values is not None and s not in spec.allowed_values \
            and s.casefold() not in spec.allowed_values:
        return issue(
            "domain_mismatch",
            f"{s!r} is not in the allowed vocabulary for {col!r} "
            f"({len(spec.allowed_values)} known values)",
            "allowed_values",
            suggestion=spec.suggestion_for(s),
        )

    if spec.pattern is not None and not _safe_fullmatch(spec.pattern, s):
        return issue(
            "domain_mismatch",
            f"{s!r} does not match the {col!r} format {spec.pattern!r}",
            "pattern",
            suggestion=spec.suggestion_for(s),
        )

    ref = spec.in_reference(s)
    if ref is False:
        return issue(
            "domain_mismatch",
            f"{s!r} not found in the configured reference source for {col!r}",
            "reference_lookup",
            suggestion=spec.suggestion_for(s),
        )

    # -- semantic-type sanity for non-parsing text types ------------------------
    if spec.semantic_type in ("company_name", "entity_name", "person_name", "city",
                              "country", "free_text", "text"):
        if spec.semantic_type != "free_text" and detected == "numeric":
            return issue(
                "semantic_mismatch",
                f"{col!r} expects a {expected} but {s!r} is purely numeric",
                "entity_numeric", confidence=0.8,
            )
        if spec.max_length is not None and len(s) > spec.max_length:
            return issue(
                "schema_violation",
                f"{col!r} value length {len(s)} exceeds max_length={spec.max_length}",
                "max_length",
            )
        return None

    if spec.semantic_type in ("identifier", "account_number") and not _ID_RE.match(s):
        return issue(
            "domain_mismatch",
            f"{s!r} contains characters not allowed in a {expected}",
            "identifier_charset",
        )
    if spec.semantic_type in ("ticker", "stock_ticker") and not _TICKER_RE.match(s):
        return issue(
            "domain_mismatch",
            f"{s!r} is not a structurally valid ticker symbol "
            "(expected 1-6 uppercase letters, optional suffix)",
            "ticker_format",
            suggestion=spec.suggestion_for(s),
        )
    if spec.semantic_type == "email" and not _EMAIL_RE.match(s):
        return issue("semantic_mismatch", f"{s!r} is not a valid email address", "email_format")
    if spec.semantic_type == "url" and not _URL_RE.match(s):
        return issue("semantic_mismatch", f"{s!r} is not a valid URL", "url_format")
    if spec.semantic_type == "phone" and not (
            _PHONE_RE.match(s) and sum(c.isdigit() for c in s) >= 7):
        return issue("semantic_mismatch", f"{s!r} is not a plausible phone number", "phone_format")

    if spec.max_length is not None and len(s) > spec.max_length:
        return issue(
            "schema_violation",
            f"{col!r} value length {len(s)} exceeds max_length={spec.max_length}",
            "max_length",
        )
    return None


#: Matches values the default text-cleaning config could possibly change:
#: anything non-ASCII-printable, leading/trailing space, or doubled spaces.
_DIRTY_PROBE = re.compile(r"[^\x20-\x7e]|^\s|\s$|\s{2,}")
_DEFAULT_CLEAN_FIELDS = (
    "unicode_form", "strip_control_chars", "strip_zero_width",
    "normalize_punctuation", "collapse_whitespace", "strip",
)


def _probe_safe(cfg: TextCleanConfig) -> bool:
    """True when :data:`_DIRTY_PROBE` is a superset of what ``cfg`` can change."""
    default = TextCleanConfig()
    for name in ("strip_html", "strip_urls", "case", "remove_punctuation",
                 "max_char_repeat", "max_length", "custom"):
        if getattr(cfg, name) != getattr(default, name):
            return False
    return True


_PURE_NUMBER_RE = r"-?[\d.,]+([eE][+-]?\d+)?"


def _suspect_rows(series: pd.Series, spec: FieldSpec) -> pd.Index:
    """Cells that need the (authoritative) per-cell check.

    Vectorized pre-screen: every cell the slow path could flag **must** land
    here; cells proven fine are skipped. When in doubt (currency formatting,
    callable references, exotic dtypes) a cell simply stays a suspect — the
    slow path re-decides, so over-selection costs time, never correctness.
    """
    nonnull = series.notna()
    strs = series.astype("string").str.strip()

    missing = ~nonnull | strs.str.casefold().isin(spec.null_markers).fillna(False)
    if spec.required or not spec.nullable:
        must_flag = missing  # every missing cell is a violation → slow path
    else:
        must_flag = pd.Series(False, index=series.index)
    checkable = ~missing

    if spec.semantic_type in _NUMERIC_TYPES:
        parsed = pd.to_numeric(strs.str.replace(",", "", regex=False), errors="coerce")
        fine = parsed.notna()
        if spec.min_value is not None:
            fine &= parsed >= spec.min_value
        if spec.max_value is not None:
            fine &= parsed <= spec.max_value
        return series.index[must_flag | (checkable & ~fine.fillna(False))]

    if spec.semantic_type in _DATE_TYPES:
        try:
            import warnings  # noqa: PLC0415

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed_dt = pd.to_datetime(strs, errors="coerce")
            fine = parsed_dt.notna()
        except (ValueError, TypeError):  # pragma: no cover - exotic payloads
            fine = pd.Series(False, index=series.index)
        return series.index[must_flag | (checkable & ~fine.fillna(False))]

    fine = pd.Series(True, index=series.index)
    if spec.allowed_values is not None:
        fine &= (strs.isin(spec.allowed_values)
                 | strs.str.casefold().isin(spec.allowed_values)).fillna(False)
    if spec.pattern is not None:
        fine &= strs.map(
            lambda v: _safe_fullmatch(spec.pattern, v) if isinstance(v, str) else False
        ).fillna(False)
    if spec.reference is not None:
        if callable(spec.reference):
            fine &= False  # cannot vectorize a callable — everything is a suspect
        else:
            fine &= strs.isin([str(v) for v in spec.reference]).fillna(False)
    if spec.max_length is not None:
        fine &= (strs.str.len() <= spec.max_length).fillna(False)

    type_res = {
        "identifier": _ID_RE, "account_number": _ID_RE,
        "ticker": _TICKER_RE, "stock_ticker": _TICKER_RE,
        "email": _EMAIL_RE, "url": _URL_RE,
    }
    if spec.semantic_type in type_res:
        fine &= strs.str.fullmatch(type_res[spec.semantic_type].pattern).fillna(False)
    elif spec.semantic_type == "phone":
        fine &= (strs.str.fullmatch(_PHONE_RE.pattern).fillna(False)
                 & (strs.str.count(r"\d") >= 7).fillna(False))
    elif spec.semantic_type in ("company_name", "entity_name", "person_name",
                                "city", "country"):
        fine &= ~strs.str.fullmatch(_PURE_NUMBER_RE).fillna(False)

    return series.index[must_flag | (checkable & ~fine)]


#: Consensus share above which a column without a spec is treated as typed.
_CONSENSUS_SHARE = 0.8


def _column_consensus(series: pd.Series) -> tuple[str, float] | None:
    """Dominant value shape of a column, if any (``(type, share)``)."""
    sample = series.dropna()
    if len(sample) > 10_000:
        sample = sample.head(10_000)
    if len(sample) < 3:
        return None
    counts: dict = {}
    for v in sample:
        counts[detect_value_type(v)] = counts.get(detect_value_type(v), 0) + 1
    counts.pop("null", None)
    if not counts:
        return None
    top, n = max(counts.items(), key=lambda kv: kv[1])
    share = n / sum(counts.values())
    if share >= _CONSENSUS_SHARE and top in ("numeric", "date_like", "email", "url", "phone"):
        return top, share
    return None


def _iqr_outliers(series: pd.Series, k: float = 3.0) -> pd.Series:
    """Boolean mask of extreme numeric values (Tukey fences, conservative k)."""
    nums = pd.to_numeric(series, errors="coerce")
    valid = nums.dropna()
    if len(valid) < 8:
        return pd.Series(False, index=series.index)
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    return (nums < q1 - k * iqr) | (nums > q3 + k * iqr)


def _validate_column(
    col: str,
    series: pd.Series,
    spec: FieldSpec | None,
    policy: RemediationPolicy,
    report: FieldValidationReport,
    *,
    rare_threshold: float,
    outlier_fence: float,
    clean_config: TextCleanConfig | None,
) -> None:
    """Run every per-column check for one column, appending to ``report``."""
    # --- text normalization (audited, never in-place) -------------------------
    cleaned_col: dict = {}
    transforms_col: dict = {}
    if policy.normalize_text:
        cfg = config_for_field(spec.semantic_type if spec else None, clean_config)
        candidates = series
        if _probe_safe(cfg):
            try:
                mask = series.astype("string").str.contains(_DIRTY_PROBE, na=False)
                candidates = series[mask.fillna(False).astype(bool)]
            except (TypeError, ValueError):  # pragma: no cover - exotic payloads
                candidates = series
        for idx, val in candidates.items():
            if isinstance(val, str):
                r = clean_text_value(val, cfg)
                if r.changed:
                    cleaned_col[idx] = r.cleaned
                    transforms_col[idx] = r.transforms
                    report.normalized_cells.append({
                        "row": idx, "column": col,
                        "original": val, "cleaned": r.cleaned,
                        "transforms": list(r.transforms),
                    })

    def cell(idx: Any) -> tuple:
        raw = series.loc[idx]
        return cleaned_col.get(idx, raw), transforms_col.get(idx, ())

    if spec is not None:
        for idx in _suspect_rows(series, spec):
            cleaned, transforms = cell(idx)
            found = _check_value(col, idx, series.loc[idx], cleaned,
                                 transforms, spec, policy)
            if found is not None:
                report.issues.append(found)
        _rare_category_issues(col, series, spec, policy, report, rare_threshold)
        if spec.semantic_type in _NUMERIC_TYPES:
            _outlier_issues(col, series, spec, policy, report, outlier_fence, cell)
        return

    # --- no spec: consensus-based contamination check -------------------------
    consensus = _column_consensus(series)
    if consensus is None:
        return
    ctype, share = consensus
    report.inferred_types[col] = {"type": ctype, "share": round(share, 4)}
    for idx, val in series.items():
        cleaned, transforms = cell(idx)
        if val is None or (not isinstance(val, str) and pd.isna(val)):
            continue
        detected = detect_value_type(cleaned)
        if detected in (ctype, "null"):
            continue
        report.issues.append(CellIssue(
            row=idx, column=col, original=val, cleaned=cleaned,
            classification="semantic_mismatch", severity="error",
            reason=f"{share:.0%} of {col!r} values are {ctype}, but this cell is "
                   f"{detected} ({val!r}); it does not fit the column's inferred type",
            expected=ctype, detected=detected, confidence=share,
            action=policy.action_for("semantic_mismatch"),
            rule="column_consensus", transforms=transforms,
        ))


def _rare_category_issues(
    col: str,
    series: pd.Series,
    spec: FieldSpec,
    policy: RemediationPolicy,
    report: FieldValidationReport,
    rare_threshold: float,
) -> None:
    """Warn about *allowed* but rare categories — rare is never invalid."""
    if spec.allowed_values is None or not 0 < rare_threshold < 1 or len(series) < 20:
        return
    freq = series.astype(str).str.strip().value_counts(normalize=True)
    flagged_rows = {i.row for i in report.issues if i.column == col}
    for value, share in freq.items():
        if share >= rare_threshold or value not in spec.allowed_values:
            continue
        for idx in series.index[series.astype(str).str.strip() == value]:
            if idx in flagged_rows:
                continue
            report.issues.append(CellIssue(
                row=idx, column=col, original=series.loc[idx],
                cleaned=value, classification="categorical_rare",
                severity="warning",
                reason=f"{value!r} is allowed but rare in {col!r} ({share:.1%} of rows)",
                expected=spec.semantic_type or "category",
                detected="rare_category", confidence=1 - share,
                action=policy.action_for("categorical_rare"),
                rule="rare_category",
            ))


def _outlier_issues(
    col: str,
    series: pd.Series,
    spec: FieldSpec,
    policy: RemediationPolicy,
    report: FieldValidationReport,
    outlier_fence: float,
    cell: Callable[[Any], tuple],
) -> None:
    """Flag far numeric outliers as warnings; extreme is not automatically wrong."""
    mask = _iqr_outliers(
        series.map(lambda v: _parse_numeric(str(v)) if isinstance(v, str) else v),
        k=outlier_fence)
    for idx in series.index[mask]:
        report.issues.append(CellIssue(
            row=idx, column=col, original=series.loc[idx],
            cleaned=cell(idx)[0], classification="statistical_outlier",
            severity="warning",
            reason=f"{series.loc[idx]!r} lies far outside the column's interquartile "
                   f"fences (k={outlier_fence}); extreme values may still be legitimate",
            expected=spec.semantic_type or "numeric", detected="numeric",
            confidence=0.7,
            action=policy.action_for("statistical_outlier"),
            rule="iqr_fence",
        ))


def validate_fields(
    df: pd.DataFrame,
    schema: Mapping[str, FieldSpec] | None = None,
    *,
    policy: RemediationPolicy | None = None,
    infer_unspecified: bool = True,
    rare_threshold: float = 0.01,
    outlier_fence: float = 3.0,
    cross_rules: Sequence[Callable[[Mapping[str, Any]], str | None]] = (),
    clean_config: TextCleanConfig | None = None,
) -> FieldValidationReport:
    """Validate every cell of ``df`` in the context of its field.

    Parameters
    ----------
    df:
        Input frame — never modified.
    schema:
        ``{column: FieldSpec}``. Columns absent from the schema are checked by
        *consensus*: when ≥80% of a column's values share one shape, the
        nonconforming minority is reported as ``semantic_mismatch``.
    policy:
        Maps failure classes to actions; defaults are non-destructive.
    infer_unspecified:
        Disable to check only columns present in ``schema``.
    rare_threshold:
        Frequency below which an *allowed* categorical value is additionally
        reported as ``categorical_rare`` (warning only — rare is not invalid).
    outlier_fence:
        Tukey fence multiplier for ``statistical_outlier`` (3.0 = far outliers).
    cross_rules:
        Callables receiving one row as a mapping; return a reason string when
        the row violates a cross-column relationship, else ``None``.
    clean_config:
        Safe text normalization applied (per field type) before validation
        when ``policy.normalize_text`` is true; every change is audited.
    """
    policy = policy or RemediationPolicy()
    schema = dict(schema or {})
    report = FieldValidationReport(n_rows=len(df))

    checked_cols = [c for c in df.columns if c in schema] + (
        [c for c in df.columns if c not in schema] if infer_unspecified else [])
    report.columns_checked = [str(c) for c in checked_cols]

    # required columns missing entirely
    for col, spec in schema.items():
        if spec.required and col not in df.columns:
            report.issues.append(CellIssue(
                row=None, column=str(col), original=None, cleaned=None,
                classification="schema_violation", severity="error",
                reason=f"required column {col!r} is missing from the frame",
                expected=spec.semantic_type or "any", detected="missing_column",
                confidence=1.0, action=policy.action_for("schema_violation"),
                rule="required_column",
            ))

    for col in checked_cols:
        _validate_column(
            str(col), df[col], schema.get(col), policy, report,
            rare_threshold=rare_threshold, outlier_fence=outlier_fence,
            clean_config=clean_config,
        )

    # --- cross-field rules ------------------------------------------------------
    if cross_rules:
        for idx, row in df.iterrows():
            row_map = row.to_dict()
            for n, rule in enumerate(cross_rules):
                reason = rule(row_map)
                if reason:
                    report.issues.append(CellIssue(
                        row=idx, column="*", original=None, cleaned=None,
                        classification="cross_field_inconsistency", severity="warning",
                        reason=reason, expected="consistent row", detected="inconsistent",
                        confidence=1.0,
                        action=policy.action_for("cross_field_inconsistency"),
                        rule=getattr(rule, "__name__", f"cross_rule_{n}"),
                    ))

    return report


@dataclass
class PolicyResult:
    """Outcome of :func:`apply_field_policy` — three disjoint frames + audit."""

    accepted: pd.DataFrame
    quarantined: pd.DataFrame
    rejected: pd.DataFrame
    needs_review: pd.DataFrame
    audit: list

    def summary(self) -> str:
        return (
            f"accepted {len(self.accepted)} / quarantined {len(self.quarantined)} / "
            f"rejected {len(self.rejected)} / needs review {len(self.needs_review)} row(s); "
            f"{len(self.audit)} audit record(s)"
        )


def apply_field_policy(df: pd.DataFrame, report: FieldValidationReport) -> PolicyResult:
    """Split ``df`` by the per-row actions in ``report``. Never mutates ``df``.

    ``replace_with_null`` is applied on the *accepted copy* only, and every
    replacement is written to the audit trail with its original value.
    """
    actions = report.row_actions()
    quarantine_rows = {r for r, a in actions.items() if a == "quarantine"}
    reject_rows = {r for r, a in actions.items() if a == "reject"}
    review_rows = {r for r, a in actions.items() if a == "manual_review"}
    removed = quarantine_rows | reject_rows | review_rows

    accepted = df.loc[[i for i in df.index if i not in removed]].copy()
    audit: list = []

    for issue in report.issues:
        audit.append({**issue.to_dict(), "applied": issue.action})
        if issue.action == "replace_with_null" and issue.row in accepted.index \
                and issue.column in accepted.columns:
            accepted.loc[issue.row, issue.column] = None

    return PolicyResult(
        accepted=accepted,
        quarantined=df.loc[sorted(quarantine_rows, key=str)].copy(),
        rejected=df.loc[sorted(reject_rows, key=str)].copy(),
        needs_review=df.loc[sorted(review_rows, key=str)].copy(),
        audit=audit,
    )
