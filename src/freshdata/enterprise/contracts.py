"""Schema-drift & data-contract monitoring with persisted baselines.

This module records the schema and summary statistics of a *trusted* dataset as
a versioned, JSON-serialisable :class:`DatasetBaseline`, then compares future
datasets against it and returns warn/fail :class:`DriftFinding` results. It
combines three ideas:

* **dbt-style contract enforcement** — declared column types, nullability,
  uniqueness, allowed values, ranges, and regex constraints (:class:`DataContract`).
* **Evidently-style distribution drift** — a dependency-free Kolmogorov–Smirnov
  statistic and Population Stability Index (PSI) over numeric and categorical
  columns, computed from baseline quantile/frequency summaries.
* **Trust-score quality gates** — the monitor can fail when the current frame's
  Data Trust Score drops below a threshold, reusing the existing
  :func:`freshdata.enterprise.metrics.compute_trust_score`.

Baselines are persisted as stable, readable JSON tagged with
``"schema_version": "freshdata-baseline-v1"``. By design they never store raw
sample values unless ``include_samples=True`` is passed explicitly, so a
baseline cannot leak PII.

>>> import freshdata as fd
>>> base = fd.build_baseline(trusted_df, name="customers")
>>> fd.save_baseline(base, "customers.baseline.json")
>>> report = fd.compare_to_baseline(new_df, fd.load_baseline("customers.baseline.json"))
>>> report.passed
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..adapters.polars import to_pandas
from ..findings import QualityFinding
from ..render.mixins import SimpleHtmlReport
from .config import (
    AnonymizationConfig,  # noqa: F401  (re-exported for discoverability)
    DriftConfig,
    PIIDetectionConfig,  # noqa: F401
)
from .metrics import compute_trust_score

try:  # pragma: no cover - trivial
    from .. import __version__ as FRESHDATA_VERSION
except Exception:  # pragma: no cover - defensive
    FRESHDATA_VERSION = "unknown"

SCHEMA_VERSION = "freshdata-baseline-v1"

_Level = Literal["info", "warning", "error"]
_Status = Literal["passed", "warned", "failed"]

#: Probabilities captured by :class:`ColumnBaseline.quantiles`.
_QUANTILE_PROBS: dict[str, float] = {
    "p01": 0.01,
    "p05": 0.05,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p95": 0.95,
    "p99": 0.99,
}
_MAX_SAMPLE_VALUES = 10
_MAX_TOP_CATEGORIES = 50
_EPS = 1e-6


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), ndigits)


def _hash_label(value: str) -> str:
    """Stable, non-reversible category label for PII-safe baselines."""
    return "h:" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _normalize_dtype(dtype: str | None) -> str | None:
    """Collapse a pandas/polars dtype string to a comparable *family*.

    ``int64``/``Int32`` → ``int``; ``float64`` → ``float``; ``object``/``string``
    → ``string``; ``datetime64[ns, UTC]`` → ``datetime``; ``bool`` → ``bool``.
    Keeps drift comparison robust to width and nullable-extension variants.
    """
    if dtype is None:
        return None
    d = dtype.strip().lower()
    if d.startswith(("int", "uint")) or d in ("int", "integer"):
        return "int"
    if d.startswith("float") or d in ("double", "decimal"):
        return "float"
    if d.startswith("bool"):
        return "bool"
    if d.startswith(("datetime", "timestamp")) or "date" in d:
        return "datetime"
    if d.startswith(("object", "string", "str", "category", "utf8")):
        return "string"
    return d


# =====================================================================
# Contract dataclasses
# =====================================================================


@dataclass(frozen=True)
class ColumnContract:
    """Declared expectations for a single column (dbt-style)."""

    name: str
    dtype: str | None = None
    nullable: bool = True
    required: bool = True
    unique: bool = False
    allowed_values: tuple[Any, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    regex: str | None = None
    max_missing_ratio: float | None = None
    max_cardinality: int | None = None
    semantic_type: str | None = None
    description: str | None = None
    #: Minimum fraction of non-null values that must pass the value-level checks
    #: (allowed_values / range / regex / length). 1.0 = every value must pass;
    #: 0.95 tolerates up to 5% violations (reported as a warning instead).
    mostly: float = 1.0
    min_length: int | None = None
    max_length: int | None = None
    #: ISO-8601 bounds checked via ``pd.to_datetime`` (``min_value``/``max_value``
    #: are numeric-only and do not apply to datetime columns).
    min_datetime: str | None = None
    max_datetime: str | None = None
    #: Compare the raw dtype string instead of collapsing to a dtype family.
    dtype_exact: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.mostly <= 1.0):
            raise ValueError(f"mostly must be in (0, 1], got {self.mostly}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "required": self.required,
            "unique": self.unique,
            "allowed_values": list(self.allowed_values),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "regex": self.regex,
            "max_missing_ratio": self.max_missing_ratio,
            "max_cardinality": self.max_cardinality,
            "semantic_type": self.semantic_type,
            "description": self.description,
            "mostly": self.mostly,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "min_datetime": self.min_datetime,
            "max_datetime": self.max_datetime,
            "dtype_exact": self.dtype_exact,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ColumnContract:
        return cls(
            name=d["name"],
            dtype=d.get("dtype"),
            nullable=d.get("nullable", True),
            required=d.get("required", True),
            unique=d.get("unique", False),
            allowed_values=tuple(d.get("allowed_values", ())),
            min_value=d.get("min_value"),
            max_value=d.get("max_value"),
            regex=d.get("regex"),
            max_missing_ratio=d.get("max_missing_ratio"),
            max_cardinality=d.get("max_cardinality"),
            semantic_type=d.get("semantic_type"),
            description=d.get("description"),
            mostly=d.get("mostly", 1.0),
            min_length=d.get("min_length"),
            max_length=d.get("max_length"),
            min_datetime=d.get("min_datetime"),
            max_datetime=d.get("max_datetime"),
            dtype_exact=d.get("dtype_exact", False),
        )


@dataclass(frozen=True)
class DataContract:
    """A named, versioned set of column contracts plus dataset-level policy."""

    name: str
    columns: tuple[ColumnContract, ...]
    version: str = "1.0.0"
    strict_columns: bool = False
    allow_extra_columns: bool = True
    fail_on_missing_required: bool = True
    fail_on_dtype_change: bool = True
    warn_on_extra_columns: bool = True
    trust_score_min: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Absolute row-count bounds for the dataset.
    min_rows: int | None = None
    max_rows: int | None = None
    #: Column groups that must be jointly unique, e.g. ``(("region", "sku"),)``.
    compound_unique: tuple[tuple[str, ...], ...] = ()

    def column(self, name: str) -> ColumnContract | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "columns": [c.to_dict() for c in self.columns],
            "strict_columns": self.strict_columns,
            "allow_extra_columns": self.allow_extra_columns,
            "fail_on_missing_required": self.fail_on_missing_required,
            "fail_on_dtype_change": self.fail_on_dtype_change,
            "warn_on_extra_columns": self.warn_on_extra_columns,
            "trust_score_min": self.trust_score_min,
            "metadata": dict(self.metadata),
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "compound_unique": [list(group) for group in self.compound_unique],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataContract:
        return cls(
            name=d["name"],
            version=d.get("version", "1.0.0"),
            columns=tuple(ColumnContract.from_dict(c) for c in d.get("columns", ())),
            strict_columns=d.get("strict_columns", False),
            allow_extra_columns=d.get("allow_extra_columns", True),
            fail_on_missing_required=d.get("fail_on_missing_required", True),
            fail_on_dtype_change=d.get("fail_on_dtype_change", True),
            warn_on_extra_columns=d.get("warn_on_extra_columns", True),
            trust_score_min=d.get("trust_score_min"),
            metadata=dict(d.get("metadata", {})),
            min_rows=d.get("min_rows"),
            max_rows=d.get("max_rows"),
            compound_unique=tuple(tuple(g) for g in d.get("compound_unique", ())),
        )


# =====================================================================
# Baseline dataclasses
# =====================================================================


@dataclass
class ColumnBaseline:
    """Recorded schema + statistics for one column of a trusted dataset."""

    name: str
    dtype: str
    missing_ratio: float
    cardinality: int
    n_unique: int
    n_rows: int
    sample_values: tuple[str, ...] = ()
    # numeric
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    quantiles: dict[str, float] = field(default_factory=dict)
    # categorical
    top_values: tuple[str, ...] = ()
    frequencies: dict[str, float] = field(default_factory=dict)
    # datetime
    min_timestamp: str | None = None
    max_timestamp: str | None = None
    profiled_at: str = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> Literal["numeric", "categorical", "datetime", "other"]:
        family = _normalize_dtype(self.dtype)
        if family in ("int", "float"):
            return "numeric"
        if family == "datetime":
            return "datetime"
        if family in ("string", "bool"):
            return "categorical"
        return "other"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "missing_ratio": _round(self.missing_ratio),
            "cardinality": self.cardinality,
            "n_unique": self.n_unique,
            "n_rows": self.n_rows,
            "sample_values": list(self.sample_values),
            "min": _round(self.min),
            "max": _round(self.max),
            "mean": _round(self.mean),
            "std": _round(self.std),
            "quantiles": {k: _round(v) for k, v in self.quantiles.items()},
            "top_values": list(self.top_values),
            "frequencies": {k: _round(v) for k, v in self.frequencies.items()},
            "min_timestamp": self.min_timestamp,
            "max_timestamp": self.max_timestamp,
            "profiled_at": self.profiled_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ColumnBaseline:
        return cls(
            name=d["name"],
            dtype=d["dtype"],
            missing_ratio=d.get("missing_ratio", 0.0) or 0.0,
            cardinality=d.get("cardinality", 0),
            n_unique=d.get("n_unique", 0),
            n_rows=d.get("n_rows", 0),
            sample_values=tuple(d.get("sample_values", ())),
            min=d.get("min"),
            max=d.get("max"),
            mean=d.get("mean"),
            std=d.get("std"),
            quantiles={
                k: float(v) for k, v in (d.get("quantiles") or {}).items() if v is not None
            },
            top_values=tuple(d.get("top_values", ())),
            frequencies={
                k: float(v) for k, v in (d.get("frequencies") or {}).items() if v is not None
            },
            min_timestamp=d.get("min_timestamp"),
            max_timestamp=d.get("max_timestamp"),
            profiled_at=d.get("profiled_at", _utcnow()),
            metadata=dict(d.get("metadata", {})),
        )

    def cdf_points(self) -> list[tuple[float, float]]:
        """``(probability, value)`` knots describing the baseline numeric CDF."""
        if self.min is None or self.max is None:
            return []
        pairs: list[tuple[float, float]] = [(0.0, float(self.min))]
        for key, prob in _QUANTILE_PROBS.items():
            if key in self.quantiles and self.quantiles[key] is not None:
                pairs.append((prob, float(self.quantiles[key])))
        pairs.append((1.0, float(self.max)))
        # Sort by probability and drop non-increasing value duplicates.
        pairs.sort(key=lambda pv: pv[0])
        return pairs


@dataclass
class DatasetBaseline:
    """A persisted snapshot of a trusted dataset's schema and statistics."""

    name: str
    row_count: int
    columns: dict[str, ColumnBaseline]
    column_order: tuple[str, ...]
    version: str = "1.0.0"
    created_at: str = field(default_factory=_utcnow)
    freshdata_version: str = FRESHDATA_VERSION
    contract: DataContract | None = None
    trust_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "freshdata_version": self.freshdata_version,
            "row_count": self.row_count,
            "column_order": list(self.column_order),
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "contract": self.contract.to_dict() if self.contract else None,
            "trust_score": _round(self.trust_score, 4),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetBaseline:
        got = d.get("schema_version")
        if got != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported baseline schema_version {got!r}; expected {SCHEMA_VERSION!r}"
            )
        contract = d.get("contract")
        return cls(
            name=d["name"],
            version=d.get("version", "1.0.0"),
            created_at=d.get("created_at", _utcnow()),
            freshdata_version=d.get("freshdata_version", "unknown"),
            row_count=d.get("row_count", 0),
            column_order=tuple(d.get("column_order", ())),
            columns={k: ColumnBaseline.from_dict(v) for k, v in d.get("columns", {}).items()},
            contract=DataContract.from_dict(contract) if contract else None,
            trust_score=d.get("trust_score"),
            metadata=dict(d.get("metadata", {})),
        )


# =====================================================================
# Findings & report
# =====================================================================


@dataclass
class DriftFinding:
    """One drift / contract / quality observation."""

    check_id: str
    level: Literal["info", "warning", "error"]
    status: Literal["passed", "warned", "failed"]
    message: str
    column: str | None = None
    baseline_value: Any = None
    current_value: Any = None
    metric: str | None = None
    threshold: Any = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "level": self.level,
            "status": self.status,
            "column": self.column,
            "message": self.message,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "metric": self.metric,
            "threshold": self.threshold,
            "details": self.details,
        }


@dataclass
class DriftReport(SimpleHtmlReport):
    """The outcome of comparing a frame against a baseline."""

    baseline_name: str
    baseline_version: str
    findings: list[DriftFinding] = field(default_factory=list)
    trust_score: float | None = None
    distribution_drift: dict[str, Any] = field(default_factory=dict)
    contract_results: dict[str, Any] = field(default_factory=dict)
    #: Key-level change counts (added/removed/changed/unchanged rows) when
    #: :func:`compare_to_baseline` was called with ``key=``, else ``None``.
    key_changes: dict[str, Any] | None = None

    @property
    def n_findings(self) -> int:
        return len(self.findings)

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.status == "failed")

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.status == "warned")

    @property
    def passed(self) -> bool:
        return self.n_errors == 0

    @property
    def errors(self) -> list[DriftFinding]:
        return [f for f in self.findings if f.status == "failed"]

    @property
    def warnings(self) -> list[DriftFinding]:
        return [f for f in self.findings if f.status == "warned"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "baseline_version": self.baseline_version,
            "passed": self.passed,
            "n_findings": self.n_findings,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "trust_score": _round(self.trust_score, 4),
            "findings": [f.to_dict() for f in self.findings],
            "distribution_drift": self.distribution_drift,
            "contract_results": self.contract_results,
            **({"key_changes": self.key_changes} if self.key_changes is not None else {}),
        }

    def what_likely_matters(self) -> list[str]:
        """Business-language highlights — the few changes worth a human's time."""
        out: list[str] = []
        added = [f for f in self.findings if f.check_id == "schema.new_column"]
        removed = [f for f in self.findings if f.check_id == "schema.removed_column"]
        if removed:
            out.append(f"{len(removed)} column(s) disappeared — downstream reports "
                       "that use them may break.")
        if added:
            out.append(f"{len(added)} new column(s) appeared — confirm they are expected.")
        dtype_changes = [f for f in self.findings if f.check_id == "schema.dtype_change"]
        if dtype_changes:
            out.append(f"{len(dtype_changes)} column(s) changed type — parsing or maths "
                       "downstream may behave differently.")
        comp = [f for f in self.findings
                if f.check_id and "missing_ratio" in f.check_id and f.status != "passed"]
        if comp:
            out.append(f"Completeness shifted in {len(comp)} column(s) — fields that were "
                       "reliable may now have gaps.")
        if self.distribution_drift:
            drifted = [c for c, d in self.distribution_drift.items()
                       if isinstance(d, dict) and d.get("drifted")]
            if drifted:
                out.append(f"The distribution of {len(drifted)} column(s) shifted "
                           f"meaningfully: {', '.join(map(str, drifted[:5]))}.")
        if self.key_changes:
            kc = self.key_changes
            if kc.get("added"):
                out.append(f"{kc['added']:,} new record(s) since the baseline.")
            if kc.get("removed"):
                out.append(f"{kc['removed']:,} record(s) present before are now gone.")
            if kc.get("changed"):
                out.append(f"{kc['changed']:,} record(s) changed values.")
        if not out:
            out.append("No material drift detected — the data looks consistent with the "
                       "baseline.")
        return out

    # -- HTML ----------------------------------------------------------------

    def _html_title(self) -> str:
        return f"freshdata drift report — {self.baseline_name}"

    def _html_subtitle(self) -> str | None:
        verdict = "PASS" if self.passed else "FAIL"
        return (f"{verdict} · {self.n_errors} error(s), {self.n_warnings} warning(s) "
                f"· baseline v{self.baseline_version}")

    def _html_sections(self) -> list[str]:
        from ..render import html as _H  # noqa: PLC0415

        cards = _H.scorecards([
            ("verdict", "PASS" if self.passed else "FAIL"),
            ("errors", self.n_errors),
            ("warnings", self.n_warnings),
            *([("trust", f"{self.trust_score:.1f}")] if self.trust_score is not None else []),
        ])
        matters = _H.section("What likely matters", "<ul>" + "".join(
            f"<li>{_H.esc(x)}</li>" for x in self.what_likely_matters()) + "</ul>")
        rows = [[f.check_id or "", f.column or "", _H.risk_badge(
            "high" if f.status == "failed" else "medium" if f.status == "warned" else "low"),
            f.message or ""] for f in self.findings if f.status != "passed"]
        findings = _H.section("Findings", _H.filterable_table(
            "fd-drift", ["check", "column", "status", "message"], rows,
            filters={"column": 1}, raw_columns=[2]) if rows
            else "<div class='fd-meta'>no drift findings</div>")
        dl = _H.json_download("drift_report.json", self.to_dict(), "⬇ JSON")
        return [cards, matters, findings, dl]

    def to_findings(self, *, lineage_run_id: str | None = None) -> list:
        """Project warned/failed drift findings into :class:`~freshdata.QualityFinding`."""
        out: list = []
        for f in self.findings:
            if f.status == "passed":
                continue
            expected = None
            if f.metric is not None:
                expected = str(f.metric)
                if f.baseline_value is not None:
                    expected += f" ~ baseline {f.baseline_value}"
                if f.threshold is not None:
                    expected += f" (threshold {f.threshold})"
            out.append(
                QualityFinding.create(
                    severity=f.level,
                    step="drift",
                    column=f.column,
                    rule_name=f.check_id,
                    message=f.message,
                    observed_value=f.current_value,
                    expected_condition=expected,
                    action_taken=f.status,
                    lineage_run_id=lineage_run_id,
                    extra={
                        "metric": f.metric,
                        "baseline_value": f.baseline_value,
                        "threshold": f.threshold,
                        **(f.details or {}),
                    },
                )
            )
        return out

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str, sort_keys=True)

    def to_frame(self) -> pd.DataFrame:
        """One row per finding — sortable/filterable in a notebook or export.

        Columns: ``check_id``, ``category``, ``level``, ``status``, ``column``,
        ``message``, ``baseline_value``, ``current_value``, ``metric``,
        ``threshold``. ``category`` is the dotted ``check_id`` prefix (e.g.
        ``schema``, ``contract``, ``quality``) so callers can group drift,
        contract, and quality defects without string-parsing ``check_id``.
        """
        rows = [
            {
                "check_id": f.check_id,
                "category": f.check_id.split(".", 1)[0],
                "level": f.level,
                "status": f.status,
                "column": f.column,
                "message": f.message,
                "baseline_value": f.baseline_value,
                "current_value": f.current_value,
                "metric": f.metric,
                "threshold": f.threshold,
            }
            for f in self.findings
        ]
        columns = [
            "check_id",
            "category",
            "level",
            "status",
            "column",
            "message",
            "baseline_value",
            "current_value",
            "metric",
            "threshold",
        ]
        return pd.DataFrame(rows, columns=columns)

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"drift report for {self.baseline_name} v{self.baseline_version}: {verdict} "
            f"({self.n_errors} error(s), {self.n_warnings} warning(s))"
        ]
        if self.trust_score is not None:
            lines.append(f"  trust score: {self.trust_score:.1f}")
        for f in self.findings:
            if f.status == "passed":
                continue
            marker = "✗" if f.status == "failed" else "!"
            col = f" `{f.column}`" if f.column else ""
            lines.append(f"  {marker} [{f.check_id}]{col}: {f.message}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# =====================================================================
# Profiling / baseline construction
# =====================================================================


def _profile_column(series: pd.Series, *, n_rows: int, include_samples: bool) -> ColumnBaseline:
    name = str(series.name)
    dtype = str(series.dtype)
    n_missing = int(series.isna().sum())
    missing_ratio = (n_missing / n_rows) if n_rows else 0.0
    non_null = series.dropna()
    n_unique = int(non_null.nunique())

    cb = ColumnBaseline(
        name=name,
        dtype=dtype,
        missing_ratio=missing_ratio,
        cardinality=n_unique,
        n_unique=n_unique,
        n_rows=n_rows,
    )
    family = _normalize_dtype(dtype)

    if include_samples and len(non_null):
        uniques = pd.unique(non_null)[:_MAX_SAMPLE_VALUES]
        cb.sample_values = tuple(str(v) for v in uniques)

    if family in ("int", "float") and len(non_null):
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric):
            cb.min = float(numeric.min())
            cb.max = float(numeric.max())
            cb.mean = float(numeric.mean())
            cb.std = float(numeric.std(ddof=0)) if len(numeric) > 1 else 0.0
            cb.quantiles = {
                key: float(numeric.quantile(prob)) for key, prob in _QUANTILE_PROBS.items()
            }
    elif family == "datetime" and len(non_null):
        ts = pd.to_datetime(non_null, errors="coerce").dropna()
        if len(ts):
            cb.min_timestamp = ts.min().isoformat()
            cb.max_timestamp = ts.max().isoformat()
    elif len(non_null):  # categorical / string / bool / other
        counts = non_null.astype("string").value_counts()
        top = counts.head(_MAX_TOP_CATEGORIES)
        total = int(counts.sum())
        # Category labels can themselves be PII; hash them unless the caller
        # opted into raw samples for trusted, non-sensitive reference data.
        cb.metadata["labels_hashed"] = not include_samples

        def _label(v: Any) -> str:
            s = str(v)
            return s if include_samples else _hash_label(s)

        cb.top_values = tuple(_label(v) for v in top.index)
        cb.frequencies = {_label(k): float(v) / total for k, v in top.items()} if total else {}
    return cb


def build_baseline(
    df: Any,
    *,
    name: str,
    version: str = "1.0.0",
    contract: DataContract | None = None,
    trust_score: float | None = None,
    metadata: dict[str, Any] | None = None,
    include_samples: bool = False,
) -> DatasetBaseline:
    """Profile *df* (pandas or polars) into a persistable :class:`DatasetBaseline`.

    The input frame is never modified. ``include_samples`` defaults to ``False``
    so raw values (potential PII) are *not* stored; set it only for trusted,
    non-sensitive reference data.
    """
    frame = to_pandas(df)
    n_rows = len(frame)
    columns: dict[str, ColumnBaseline] = {}
    for col in frame.columns:
        columns[str(col)] = _profile_column(
            frame[col], n_rows=n_rows, include_samples=include_samples
        )
    return DatasetBaseline(
        name=name,
        version=version,
        row_count=n_rows,
        columns=columns,
        column_order=tuple(str(c) for c in frame.columns),
        contract=contract,
        trust_score=trust_score,
        metadata=dict(metadata or {}),
    )


def save_baseline(baseline: DatasetBaseline, path: str | Path) -> None:
    """Write *baseline* to *path* as stable, human-readable JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(baseline.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def load_baseline(path: str | Path) -> DatasetBaseline:
    """Load a :class:`DatasetBaseline` previously written by :func:`save_baseline`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetBaseline.from_dict(raw)


# =====================================================================
# Distribution drift primitives (dependency-free)
# =====================================================================


def _ks_statistic(cb: ColumnBaseline, current: pd.Series) -> float | None:
    """Approximate two-sample KS using the baseline quantile CDF.

    Builds a piecewise-linear baseline CDF from stored quantiles and compares it
    against the current empirical CDF on the union of both grids. No raw baseline
    samples are required, so baselines stay PII-free.
    """
    pts = cb.cdf_points()
    if len(pts) < 2:
        return None
    vals = pd.to_numeric(current.dropna(), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return None
    xs = np.array([v for _, v in pts], dtype=float)
    ps = np.array([p for p, _ in pts], dtype=float)
    # Ensure strictly increasing xs for interpolation.
    order = np.argsort(xs, kind="stable")
    xs, ps = xs[order], ps[order]
    keep = np.concatenate(([True], np.diff(xs) > 0))
    xs, ps = xs[keep], ps[keep]
    if len(xs) < 2:
        return None

    grid = np.unique(np.concatenate([xs, np.quantile(vals, ps)]))
    f_base = np.interp(grid, xs, ps, left=0.0, right=1.0)
    f_cur = np.searchsorted(np.sort(vals), grid, side="right") / n
    return float(np.max(np.abs(f_base - f_cur)))


def _psi_numeric(cb: ColumnBaseline, current: pd.Series) -> float | None:
    """PSI over numeric bins defined by the baseline quantile edges."""
    pts = cb.cdf_points()
    if len(pts) < 2:
        return None
    vals = pd.to_numeric(current.dropna(), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return None
    probs = [p for p, _ in pts]
    edges = [v for _, v in pts]
    # Merge bins whose edges collapse (repeated quantile values).
    merged_edges: list[float] = [edges[0]]
    merged_expected: list[float] = []
    acc = 0.0
    for i in range(1, len(edges)):
        acc += probs[i] - probs[i - 1]
        if edges[i] > merged_edges[-1]:
            merged_edges.append(edges[i])
            merged_expected.append(acc)
            acc = 0.0
    if acc > 0 and merged_expected:
        merged_expected[-1] += acc
    if len(merged_edges) < 2:
        return None
    inner = np.array(merged_edges[1:-1], dtype=float)
    idx = np.searchsorted(inner, vals, side="right") if len(inner) else np.zeros(n, dtype=int)
    actual_counts = np.bincount(idx, minlength=len(merged_expected)).astype(float)
    actual = actual_counts / n
    expected = np.array(merged_expected, dtype=float)
    expected = expected / expected.sum() if expected.sum() else expected
    return _psi(expected, actual)


def _psi_categorical(cb: ColumnBaseline, current: pd.Series) -> float | None:
    """PSI over the baseline top-k categories plus an ``__OTHER__`` bucket."""
    if not cb.frequencies:
        return None
    cats = list(cb.frequencies.keys())
    cur = current.dropna().astype("string")
    n = len(cur)
    if n == 0:
        return None
    if cb.metadata.get("labels_hashed"):
        cur = cur.map(lambda v: _hash_label(str(v)))
    cur_counts = cur.value_counts(normalize=True)
    expected = np.array([cb.frequencies[c] for c in cats] + [0.0], dtype=float)
    expected[-1] = max(0.0, 1.0 - float(np.sum(expected[:-1])))
    actual_vals = [float(cur_counts.get(c, 0.0)) for c in cats]
    actual = np.array(actual_vals + [max(0.0, 1.0 - sum(actual_vals))], dtype=float)
    s = expected.sum()
    if s:
        expected = expected / s
    return _psi(expected, actual)


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, _EPS, None)
    a = np.clip(actual, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


# =====================================================================
# Comparison
# =====================================================================


def _add(
    findings: list[DriftFinding],
    check_id: str,
    *,
    level: Literal["info", "warning", "error"],
    status: Literal["passed", "warned", "failed"],
    message: str,
    column: str | None = None,
    baseline_value: Any = None,
    current_value: Any = None,
    metric: str | None = None,
    threshold: Any = None,
    details: dict[str, Any] | None = None,
) -> None:
    findings.append(
        DriftFinding(
            check_id=check_id,
            level=level,
            status=status,
            message=message,
            column=column,
            baseline_value=baseline_value,
            current_value=current_value,
            metric=metric,
            threshold=threshold,
            details=details or {},
        )
    )


def _check_schema(
    findings: list[DriftFinding],
    baseline: DatasetBaseline,
    current: dict[str, ColumnBaseline],
    current_order: tuple[str, ...],
    cfg: DriftConfig,
) -> None:
    base_cols = set(baseline.columns)
    cur_cols = set(current)
    fail_level: tuple[_Level, _Status] = (
        ("error", "failed") if cfg.fail_on_schema_drift else ("warning", "warned")
    )

    for col in baseline.column_order:
        if col not in cur_cols:
            _add(
                findings,
                "schema.removed_column",
                level=fail_level[0],
                status=fail_level[1],
                message=f"column {col!r} present in baseline is missing",
                column=col,
            )
    for col in current_order:
        if col not in base_cols:
            _add(
                findings,
                "schema.new_column",
                level="warning",
                status="warned",
                message=f"unexpected new column {col!r} not in baseline",
                column=col,
            )

    for col in baseline.column_order:
        if col not in current:
            continue
        base_family = _normalize_dtype(baseline.columns[col].dtype)
        cur_family = _normalize_dtype(current[col].dtype)
        if base_family != cur_family:
            _add(
                findings,
                "schema.dtype_change",
                level=fail_level[0],
                status=fail_level[1],
                message=f"dtype changed for {col!r}: {base_family} → {cur_family}",
                column=col,
                baseline_value=baseline.columns[col].dtype,
                current_value=current[col].dtype,
                metric="dtype",
            )

    common = [c for c in baseline.column_order if c in current]
    cur_common = [c for c in current_order if c in baseline.columns]
    if common != cur_common:
        _add(
            findings,
            "schema.column_order",
            level="info",
            status="warned",
            message="column order differs from baseline",
            baseline_value=common,
            current_value=cur_common,
        )


def _check_statistics(
    findings: list[DriftFinding],
    baseline: DatasetBaseline,
    current: dict[str, ColumnBaseline],
    current_rows: int,
    cfg: DriftConfig,
) -> None:
    # Row-count drift.
    if baseline.row_count:
        delta_ratio = abs(current_rows - baseline.row_count) / baseline.row_count
        if delta_ratio >= cfg.cardinality_warn_delta_ratio:
            _add(
                findings,
                "stats.row_count",
                level="warning",
                status="warned",
                message=f"row count changed by {delta_ratio:.0%}",
                baseline_value=baseline.row_count,
                current_value=current_rows,
                metric="row_count_delta_ratio",
                threshold=cfg.cardinality_warn_delta_ratio,
            )

    for col, base in baseline.columns.items():
        if col not in current:
            continue
        cur = current[col]
        # Missing-ratio drift.
        delta = cur.missing_ratio - base.missing_ratio
        if delta >= cfg.missing_ratio_fail_delta:
            _add(
                findings,
                "stats.missing_ratio",
                level="error",
                status="failed",
                message=f"missing ratio rose {delta:.2%} (>= fail delta)",
                column=col,
                baseline_value=_round(base.missing_ratio),
                current_value=_round(cur.missing_ratio),
                metric="missing_ratio_delta",
                threshold=cfg.missing_ratio_fail_delta,
            )
        elif delta >= cfg.missing_ratio_warn_delta:
            _add(
                findings,
                "stats.missing_ratio",
                level="warning",
                status="warned",
                message=f"missing ratio rose {delta:.2%}",
                column=col,
                baseline_value=_round(base.missing_ratio),
                current_value=_round(cur.missing_ratio),
                metric="missing_ratio_delta",
                threshold=cfg.missing_ratio_warn_delta,
            )
        # Cardinality drift.
        if base.cardinality:
            card_delta = abs(cur.cardinality - base.cardinality) / base.cardinality
            if card_delta >= cfg.cardinality_warn_delta_ratio:
                _add(
                    findings,
                    "stats.cardinality",
                    level="warning",
                    status="warned",
                    message=f"cardinality changed by {card_delta:.0%}",
                    column=col,
                    baseline_value=base.cardinality,
                    current_value=cur.cardinality,
                    metric="cardinality_delta_ratio",
                    threshold=cfg.cardinality_warn_delta_ratio,
                )
        # Uniqueness drift (ratio of unique to rows).
        base_u = base.n_unique / base.n_rows if base.n_rows else 0.0
        cur_u = cur.n_unique / cur.n_rows if cur.n_rows else 0.0
        if abs(cur_u - base_u) >= cfg.cardinality_warn_delta_ratio:
            _add(
                findings,
                "stats.uniqueness",
                level="warning",
                status="warned",
                message=f"uniqueness ratio shifted {abs(cur_u - base_u):.2f}",
                column=col,
                baseline_value=_round(base_u),
                current_value=_round(cur_u),
                metric="uniqueness_ratio_delta",
                threshold=cfg.cardinality_warn_delta_ratio,
            )


def _check_distribution(
    findings: list[DriftFinding],
    baseline: DatasetBaseline,
    current: dict[str, ColumnBaseline],
    frame: pd.DataFrame,
    cfg: DriftConfig,
) -> dict[str, Any]:
    drift: dict[str, Any] = {}
    for col, base in baseline.columns.items():
        if col not in current or col not in frame.columns:
            continue
        cur = current[col]
        col_drift: dict[str, Any] = {}
        if cur.n_rows < cfg.min_samples_for_distribution:
            continue
        series = frame[col]
        if base.kind == "numeric" and cur.kind == "numeric":
            ks = _ks_statistic(base, series)
            psi = _psi_numeric(base, series)
            if ks is not None:
                col_drift["ks"] = _round(ks)
            if psi is not None:
                col_drift["psi"] = _round(psi)
            _grade_metric(findings, col, "ks", ks, cfg.numeric_ks_warn, cfg.numeric_ks_fail, cfg)
            _grade_metric(findings, col, "psi", psi, cfg.psi_warn, cfg.psi_fail, cfg)
            if base.min is not None and base.max is not None:
                col_drift["range"] = {"baseline": [base.min, base.max]}
        elif base.kind == "categorical" and cur.kind == "categorical":
            if cur.cardinality <= cfg.max_categories_for_categorical_drift:
                psi = _psi_categorical(base, series)
                if psi is not None:
                    col_drift["psi"] = _round(psi)
                _grade_metric(findings, col, "psi", psi, cfg.psi_warn, cfg.psi_fail, cfg)
        elif base.kind == "datetime" and cur.kind == "datetime":
            _check_datetime_range(findings, col, base, cur, col_drift)
        if col_drift:
            drift[col] = col_drift
    return drift


def _grade_metric(
    findings: list[DriftFinding],
    col: str,
    metric: str,
    value: float | None,
    warn: float,
    fail: float,
    cfg: DriftConfig,
) -> None:
    if value is None:
        return
    if value >= fail:
        _add(
            findings,
            f"drift.{metric}",
            level="error",
            status="failed",
            message=f"{metric.upper()} {value:.3f} >= fail threshold {fail}",
            column=col,
            metric=metric,
            current_value=_round(value),
            threshold=fail,
        )
    elif value >= warn and cfg.warn_on_distribution_drift:
        _add(
            findings,
            f"drift.{metric}",
            level="warning",
            status="warned",
            message=f"{metric.upper()} {value:.3f} >= warn threshold {warn}",
            column=col,
            metric=metric,
            current_value=_round(value),
            threshold=warn,
        )


def _check_datetime_range(
    findings: list[DriftFinding],
    col: str,
    base: ColumnBaseline,
    cur: ColumnBaseline,
    col_drift: dict[str, Any],
) -> None:
    col_drift["min_timestamp"] = {"baseline": base.min_timestamp, "current": cur.min_timestamp}
    col_drift["max_timestamp"] = {"baseline": base.max_timestamp, "current": cur.max_timestamp}
    try:
        b_min = pd.Timestamp(base.min_timestamp) if base.min_timestamp else None
        b_max = pd.Timestamp(base.max_timestamp) if base.max_timestamp else None
        c_min = pd.Timestamp(cur.min_timestamp) if cur.min_timestamp else None
        c_max = pd.Timestamp(cur.max_timestamp) if cur.max_timestamp else None
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return
    if b_min is not None and c_min is not None and c_min < b_min:
        _add(
            findings,
            "drift.datetime_range",
            level="warning",
            status="warned",
            message=f"earliest timestamp {c_min} precedes baseline min {b_min}",
            column=col,
            baseline_value=str(b_min),
            current_value=str(c_min),
            metric="min_timestamp",
        )
    if b_max is not None and c_max is not None and c_max > b_max:
        _add(
            findings,
            "drift.datetime_range",
            level="warning",
            status="warned",
            message=f"latest timestamp {c_max} exceeds baseline max {b_max}",
            column=col,
            baseline_value=str(b_max),
            current_value=str(c_max),
            metric="max_timestamp",
        )


def _check_contract_dataset(
    findings: list[DriftFinding],
    contract: DataContract,
    frame: pd.DataFrame,
) -> None:
    """Dataset-level contract checks: row-count bounds and compound uniqueness."""
    n_rows = len(frame)
    if contract.min_rows is not None and n_rows < contract.min_rows:
        _add(
            findings,
            "contract.min_rows",
            level="error",
            status="failed",
            message=f"dataset has {n_rows} row(s), fewer than the required {contract.min_rows}",
            current_value=n_rows,
            threshold=contract.min_rows,
            metric="row_count",
        )
    if contract.max_rows is not None and n_rows > contract.max_rows:
        _add(
            findings,
            "contract.max_rows",
            level="error",
            status="failed",
            message=f"dataset has {n_rows} row(s), more than the allowed {contract.max_rows}",
            current_value=n_rows,
            threshold=contract.max_rows,
            metric="row_count",
        )
    for group in contract.compound_unique:
        cols = list(group)
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            _add(
                findings,
                "contract.compound_unique",
                level="error",
                status="failed",
                message=(
                    f"compound-unique group {cols} references missing column(s) {missing}"
                ),
                metric="compound_unique",
                threshold=cols,
            )
            continue
        n_dups = int(frame.duplicated(subset=cols).sum())
        if n_dups:
            _add(
                findings,
                "contract.compound_unique",
                level="error",
                status="failed",
                message=f"columns {cols} are declared jointly unique but have "
                f"{n_dups} duplicate row(s)",
                current_value=n_dups,
                threshold=cols,
                metric="compound_unique",
            )


def _check_contract(
    findings: list[DriftFinding],
    contract: DataContract,
    current: dict[str, ColumnBaseline],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    cur_cols = set(current)
    declared = {c.name for c in contract.columns}

    _check_contract_dataset(findings, contract, frame)

    if contract.strict_columns or not contract.allow_extra_columns:
        reason = (
            "strict_columns requires an exact schema match"
            if contract.strict_columns
            else "extra columns are forbidden"
        )
        for col in current:
            if col not in declared:
                _add(
                    findings,
                    "contract.unexpected_column",
                    level="error",
                    status="failed",
                    message=f"column {col!r} is not declared and {reason}",
                    column=col,
                )

    for cc in contract.columns:
        col = cc.name
        passes = True
        if col not in cur_cols:
            if contract.strict_columns:
                _add(
                    findings,
                    "contract.missing_required",
                    level="error",
                    status="failed",
                    message=(
                        f"declared column {col!r} is missing and strict_columns "
                        "requires an exact schema match"
                    ),
                    column=col,
                )
            elif cc.required and contract.fail_on_missing_required:
                _add(
                    findings,
                    "contract.missing_required",
                    level="error",
                    status="failed",
                    message=f"required column {col!r} is missing",
                    column=col,
                )
            elif cc.required:
                _add(
                    findings,
                    "contract.missing_required",
                    level="warning",
                    status="warned",
                    message=f"required column {col!r} is missing",
                    column=col,
                )
            results[col] = False
            continue
        cb = current[col]
        passes &= _contract_dtype(findings, contract, cc, cb)
        passes &= _contract_nullable(findings, cc, cb)
        passes &= _contract_unique(findings, cc, cb)
        passes &= _contract_missing_cardinality(findings, cc, cb)
        passes &= _contract_values(findings, cc, frame[col])
        results[col] = passes
    return results


def _contract_dtype(
    findings: list[DriftFinding], contract: DataContract, cc: ColumnContract, cb: ColumnBaseline
) -> bool:
    if cc.dtype is None:
        return True
    if cc.dtype_exact:
        if cc.dtype == cb.dtype:
            return True
    elif _normalize_dtype(cc.dtype) == _normalize_dtype(cb.dtype):
        return True
    pair: tuple[_Level, _Status] = (
        ("error", "failed") if contract.fail_on_dtype_change else ("warning", "warned")
    )
    level, status = pair
    exact_note = " (exact match required)" if cc.dtype_exact else ""
    _add(
        findings,
        "contract.dtype",
        level=level,
        status=status,
        message=f"{cc.name!r} expected dtype {cc.dtype}{exact_note}, found {cb.dtype}",
        column=cc.name,
        baseline_value=cc.dtype,
        current_value=cb.dtype,
        metric="dtype",
    )
    return False


def _contract_nullable(
    findings: list[DriftFinding], cc: ColumnContract, cb: ColumnBaseline
) -> bool:
    if cc.nullable or cb.missing_ratio <= 0:
        return True
    _add(
        findings,
        "contract.nullable",
        level="error",
        status="failed",
        message=f"{cc.name!r} is declared non-nullable but has nulls",
        column=cc.name,
        current_value=_round(cb.missing_ratio),
        metric="missing_ratio",
    )
    return False


def _contract_unique(findings: list[DriftFinding], cc: ColumnContract, cb: ColumnBaseline) -> bool:
    if not cc.unique:
        return True
    non_null = round(cb.n_rows * (1 - cb.missing_ratio))
    if cb.n_unique >= non_null:
        return True
    _add(
        findings,
        "contract.unique",
        level="error",
        status="failed",
        message=f"{cc.name!r} is declared unique but has duplicates",
        column=cc.name,
        baseline_value=non_null,
        current_value=cb.n_unique,
        metric="n_unique",
    )
    return False


def _contract_missing_cardinality(
    findings: list[DriftFinding], cc: ColumnContract, cb: ColumnBaseline
) -> bool:
    ok = True
    if cc.max_missing_ratio is not None and cb.missing_ratio > cc.max_missing_ratio:
        ok = False
        _add(
            findings,
            "contract.max_missing_ratio",
            level="error",
            status="failed",
            message=f"{cc.name!r} missing ratio {cb.missing_ratio:.2%} exceeds max",
            column=cc.name,
            current_value=_round(cb.missing_ratio),
            threshold=cc.max_missing_ratio,
            metric="missing_ratio",
        )
    if cc.max_cardinality is not None and cb.cardinality > cc.max_cardinality:
        ok = False
        _add(
            findings,
            "contract.max_cardinality",
            level="error",
            status="failed",
            message=f"{cc.name!r} cardinality {cb.cardinality} exceeds max {cc.max_cardinality}",
            column=cc.name,
            current_value=cb.cardinality,
            threshold=cc.max_cardinality,
            metric="cardinality",
        )
    return ok


def _value_check(
    findings: list[DriftFinding],
    cc: ColumnContract,
    check_id: str,
    *,
    n_violations: int,
    n_total: int,
    message: str,
    **kwargs: Any,
) -> bool:
    """Record a value-level violation, honouring ``cc.mostly``.

    With the default ``mostly=1.0`` any violation fails. With ``mostly < 1.0``,
    violation ratios up to ``1 - mostly`` are reported as warnings and the
    column still passes. Returns whether the column passes this check.
    """
    if n_violations <= 0 or n_total <= 0:
        return True
    ratio = n_violations / n_total
    tolerated = cc.mostly < 1.0 and ratio <= (1.0 - cc.mostly)
    level: _Level = "warning" if tolerated else "error"
    status: _Status = "warned" if tolerated else "failed"
    details = dict(kwargs.pop("details", {}))
    details["violation_ratio"] = _round(ratio)
    details["n_violations"] = n_violations
    details["mostly"] = cc.mostly
    _add(
        findings,
        check_id,
        level=level,
        status=status,
        message=f"{message} ({n_violations}/{n_total} = {ratio:.2%})",
        column=cc.name,
        details=details,
        **kwargs,
    )
    return tolerated


def _datetime_bound(bound: str, tz_aware: bool) -> pd.Timestamp:
    ts = pd.Timestamp(bound)
    if tz_aware and ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    elif not tz_aware and ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _contract_datetime_bounds(
    findings: list[DriftFinding], cc: ColumnContract, non_null: pd.Series
) -> bool:
    ok = True
    as_dt = pd.to_datetime(non_null, errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(as_dt):
        # Mixed-timezone values come back as object dtype; normalize to UTC.
        as_dt = pd.to_datetime(non_null, errors="coerce", utc=True)
    as_dt = as_dt.dropna()
    if not len(as_dt):
        return True
    tz_aware = getattr(as_dt.dt, "tz", None) is not None
    if cc.min_datetime is not None:
        bound = _datetime_bound(cc.min_datetime, tz_aware)
        ok &= _value_check(
            findings,
            cc,
            "contract.min_datetime",
            n_violations=int((as_dt < bound).sum()),
            n_total=len(as_dt),
            message=f"{cc.name!r} has datetime(s) before {cc.min_datetime}",
            current_value=str(as_dt.min()),
            threshold=cc.min_datetime,
            metric="min_datetime",
        )
    if cc.max_datetime is not None:
        bound = _datetime_bound(cc.max_datetime, tz_aware)
        ok &= _value_check(
            findings,
            cc,
            "contract.max_datetime",
            n_violations=int((as_dt > bound).sum()),
            n_total=len(as_dt),
            message=f"{cc.name!r} has datetime(s) after {cc.max_datetime}",
            current_value=str(as_dt.max()),
            threshold=cc.max_datetime,
            metric="max_datetime",
        )
    return bool(ok)


def _contract_lengths(
    findings: list[DriftFinding], cc: ColumnContract, non_null: pd.Series
) -> bool:
    ok = True
    lengths = non_null.astype("string").str.len().dropna()
    if not len(lengths):
        return True
    if cc.min_length is not None:
        ok &= _value_check(
            findings,
            cc,
            "contract.min_length",
            n_violations=int((lengths < cc.min_length).sum()),
            n_total=len(lengths),
            message=f"{cc.name!r} has string(s) shorter than {cc.min_length}",
            current_value=int(lengths.min()),
            threshold=cc.min_length,
            metric="min_length",
        )
    if cc.max_length is not None:
        ok &= _value_check(
            findings,
            cc,
            "contract.max_length",
            n_violations=int((lengths > cc.max_length).sum()),
            n_total=len(lengths),
            message=f"{cc.name!r} has string(s) longer than {cc.max_length}",
            current_value=int(lengths.max()),
            threshold=cc.max_length,
            metric="max_length",
        )
    return bool(ok)


def _contract_values(findings: list[DriftFinding], cc: ColumnContract, series: pd.Series) -> bool:
    ok = True
    non_null = series.dropna()
    n_total = len(non_null)
    if cc.allowed_values:
        bad_mask = ~non_null.isin(list(cc.allowed_values))
        n_bad = int(bad_mask.sum())
        if n_bad:
            offenders = sorted({str(v) for v in non_null[bad_mask].unique()})
            ok &= _value_check(
                findings,
                cc,
                "contract.allowed_values",
                n_violations=n_bad,
                n_total=n_total,
                message=f"{cc.name!r} has value(s) outside the allowed set",
                threshold=list(cc.allowed_values),
                metric="allowed_values",
                details={"offending_sample": offenders[:5]},
            )
    if cc.min_value is not None or cc.max_value is not None:
        if pd.api.types.is_datetime64_any_dtype(series):
            _add(
                findings,
                "contract.min_value",
                level="warning",
                status="warned",
                message=(
                    f"{cc.name!r} is a datetime column; min_value/max_value are "
                    "numeric-only and were not checked — use min_datetime/max_datetime"
                ),
                column=cc.name,
                metric="min_value" if cc.min_value is not None else "max_value",
            )
            ok = False
        else:
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numeric):
                if cc.min_value is not None:
                    n_bad = int((numeric < cc.min_value).sum())
                    ok &= _value_check(
                        findings,
                        cc,
                        "contract.min_value",
                        n_violations=n_bad,
                        n_total=len(numeric),
                        message=f"{cc.name!r} minimum {float(numeric.min())} "
                        f"below {cc.min_value}",
                        current_value=_round(float(numeric.min())),
                        threshold=cc.min_value,
                        metric="min_value",
                    )
                if cc.max_value is not None:
                    n_bad = int((numeric > cc.max_value).sum())
                    ok &= _value_check(
                        findings,
                        cc,
                        "contract.max_value",
                        n_violations=n_bad,
                        n_total=len(numeric),
                        message=f"{cc.name!r} maximum {float(numeric.max())} "
                        f"above {cc.max_value}",
                        current_value=_round(float(numeric.max())),
                        threshold=cc.max_value,
                        metric="max_value",
                    )
    if cc.min_datetime is not None or cc.max_datetime is not None:
        ok &= _contract_datetime_bounds(findings, cc, non_null)
    if cc.min_length is not None or cc.max_length is not None:
        ok &= _contract_lengths(findings, cc, non_null)
    if cc.regex:
        pattern = re.compile(cc.regex)
        as_str = non_null.astype("string")
        violations = int((~as_str.str.fullmatch(pattern)).fillna(True).sum())
        ok &= _value_check(
            findings,
            cc,
            "contract.regex",
            n_violations=violations,
            n_total=n_total,
            message=f"{cc.name!r} has value(s) not matching {cc.regex!r}",
            threshold=cc.regex,
            current_value=violations,
            metric="regex",
        )
    return bool(ok)


def compare_to_baseline(
    df: Any,
    baseline: DatasetBaseline | Any,
    *,
    contract: DataContract | None = None,
    drift_config: DriftConfig | None = None,
    trust_score: float | None = None,
    key: str | list[str] | None = None,
    event_time: str | None = None,
) -> DriftReport:
    """Compare *df* against *baseline*; return a :class:`DriftReport`.

    Read-only: *df* is never mutated. ``contract`` overrides any contract stored
    in the baseline. ``trust_score`` overrides the computed Data Trust Score for
    the gate (useful to feed a score already computed elsewhere).

    *baseline* may be a prebuilt :class:`DatasetBaseline` **or a raw DataFrame**
    (the "last week's data" case) — a baseline is built from it on the fly. With
    ``key=`` (one or more key columns) and a raw-DataFrame baseline, the report
    also carries key-level change counts (records added/removed/changed since the
    baseline). ``event_time`` names a timestamp column for recency context.
    """
    cfg = drift_config or DriftConfig()
    frame = to_pandas(df)

    # Accept a raw DataFrame baseline (build one), or a prebuilt DatasetBaseline.
    baseline_frame: pd.DataFrame | None = None
    if not isinstance(baseline, DatasetBaseline):
        baseline_frame = to_pandas(baseline)
        baseline = build_baseline(baseline_frame, name="baseline")

    key_changes = None
    if key is not None and baseline_frame is not None:
        key_changes = _key_level_changes(baseline_frame, frame, key, event_time)
    current = {
        str(col): _profile_column(frame[col], n_rows=len(frame), include_samples=False)
        for col in frame.columns
    }
    current_order = tuple(str(c) for c in frame.columns)

    findings: list[DriftFinding] = []
    distribution: dict[str, Any] = {}
    contract_results: dict[str, Any] = {}

    if cfg.enabled:
        _check_schema(findings, baseline, current, current_order, cfg)
        _check_statistics(findings, baseline, current, len(frame), cfg)
        distribution = _check_distribution(findings, baseline, current, frame, cfg)

    active_contract = contract or baseline.contract
    if active_contract is not None:
        contract_results = _check_contract(findings, active_contract, current, frame)

    # Trust-score gate.
    threshold = cfg.trust_score_min
    if active_contract is not None and active_contract.trust_score_min is not None:
        threshold = active_contract.trust_score_min
    score = trust_score
    if score is None and threshold is not None:
        score = float(compute_trust_score(frame).overall)
    if threshold is not None and score is not None and score < threshold:
        _add(
            findings,
            "quality.trust_score",
            level="error",
            status="failed",
            message=f"trust score {score:.1f} below required minimum {threshold:.1f}",
            metric="trust_score",
            current_value=_round(score, 2),
            threshold=threshold,
        )

    return DriftReport(
        baseline_name=baseline.name,
        baseline_version=baseline.version,
        findings=findings,
        trust_score=score,
        distribution_drift=distribution,
        contract_results=contract_results,
        key_changes=key_changes,
    )


def enforce_contract(df: Any, contract: DataContract) -> DriftReport:
    """Check *df* against *contract* alone — no baseline, no drift statistics.

    Read-only: *df* is never mutated. This is the contract-only companion to
    :func:`compare_to_baseline` for when there is no historical baseline, just
    declared expectations.
    """
    frame = to_pandas(df)
    current = {
        str(col): _profile_column(frame[col], n_rows=len(frame), include_samples=False)
        for col in frame.columns
    }
    findings: list[DriftFinding] = []
    contract_results = _check_contract(findings, contract, current, frame)

    score: float | None = None
    if contract.trust_score_min is not None:
        score = float(compute_trust_score(frame).overall)
        if score < contract.trust_score_min:
            _add(
                findings,
                "quality.trust_score",
                level="error",
                status="failed",
                message=(
                    f"trust score {score:.1f} below required minimum "
                    f"{contract.trust_score_min:.1f}"
                ),
                metric="trust_score",
                current_value=_round(score, 2),
                threshold=contract.trust_score_min,
            )

    return DriftReport(
        baseline_name=contract.name,
        baseline_version=contract.version,
        findings=findings,
        trust_score=score,
        contract_results=contract_results,
    )


def _key_level_changes(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    key: str | list[str],
    event_time: str | None,
) -> dict[str, Any]:
    """Count records added/removed/changed/unchanged between two frames by key."""
    keys = [key] if isinstance(key, str) else list(key)
    missing = [k for k in keys if k not in baseline_df.columns or k not in current_df.columns]
    if missing:
        return {"error": f"key column(s) not in both frames: {', '.join(missing)}"}

    b = baseline_df.drop_duplicates(subset=keys).set_index(keys)
    c = current_df.drop_duplicates(subset=keys).set_index(keys)
    b_idx, c_idx = set(b.index), set(c.index)
    added = c_idx - b_idx
    removed = b_idx - c_idx
    common = b_idx & c_idx
    # A moving update timestamp is expected, so it doesn't count as a value change.
    shared_cols = [col for col in b.columns if col in c.columns and col != event_time]
    changed = 0
    if common and shared_cols:
        bc = b.loc[sorted(common), shared_cols]
        cc = c.loc[sorted(common), shared_cols]
        # Row differs if any shared column value differs (NaN-aware).
        ne = (bc != cc) & ~(bc.isna() & cc.isna())
        changed = int(ne.any(axis=1).sum())
    out: dict[str, Any] = {
        "key": keys,
        "baseline_records": int(len(b_idx)),
        "current_records": int(len(c_idx)),
        "added": int(len(added)),
        "removed": int(len(removed)),
        "changed": changed,
        "unchanged": int(len(common) - changed),
    }
    if event_time and event_time in current_df.columns:
        ts = pd.to_datetime(current_df[event_time], errors="coerce")
        if ts.notna().any():
            out["latest_event_time"] = str(ts.max())
    return out


def monitor_contract(
    df: Any,
    *,
    baseline_path: str | Path | None = None,
    baseline: DatasetBaseline | None = None,
    contract: DataContract | None = None,
    drift_config: DriftConfig | None = None,
    trust_score: float | None = None,
    return_report: bool = True,
) -> DriftReport | bool:
    """Convenience monitor: load a baseline and compare *df* against it.

    Provide either ``baseline_path`` or an in-memory ``baseline``. Returns the
    full :class:`DriftReport` when ``return_report`` is true, else a pass/fail
    boolean.
    """
    if baseline is None:
        if baseline_path is None:
            raise ValueError("monitor_contract requires baseline= or baseline_path=")
        baseline = load_baseline(baseline_path)
    report = compare_to_baseline(
        df,
        baseline,
        contract=contract,
        drift_config=drift_config,
        trust_score=trust_score,
    )
    return report if return_report else report.passed


# =====================================================================
# Baseline-free schema diff (contract vs. current frame)
# =====================================================================


class ContractViolation(Exception):
    """Raised when a contract gate fails (a :func:`diff_schema` had errors).

    Carries the full :class:`DriftReport` at ``.report`` so callers can inspect,
    log, or export the violations after catching it.
    """

    def __init__(self, report: DriftReport) -> None:
        self.report = report
        super().__init__(report.summary())


#: Policy keyword -> (finding level, status). ``None`` means "suppress".
_POLICY: dict[str, tuple[_Level, _Status] | None] = {
    "fail": ("error", "failed"),
    "warn": ("warning", "warned"),
    "preserve": ("info", "passed"),
    "ignore": None,
}

#: Minimum normalized name similarity to treat a dtype-compatible column pair as
#: a rename when no semantic-type evidence is available. Conservative on purpose:
#: a baseline-free diff has no values to compare, so a weak signal must not
#: manufacture a rename out of two unrelated same-dtype columns.
_RENAME_NAME_SIMILARITY = 0.6


def _name_similarity(a: str, b: str) -> float:
    """0..1 similarity of two column names (case/separator-insensitive)."""
    norm_a = re.sub(r"[^a-z0-9]", "", a.lower())
    norm_b = re.sub(r"[^a-z0-9]", "", b.lower())
    if not norm_a or not norm_b:
        return 0.0
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()


def _detect_renames(
    declared: dict[str, ColumnContract],
    declared_absent: list[str],
    unexpected: list[str],
    df: Any,
    semantic_now: dict[str, str],
    findings: list[DriftFinding],
    cats: dict[str, Any],
) -> tuple[set[str], set[str]]:
    """Pair absent declared columns with unexpected ones on *evidence* only.

    A baseline-free diff has no values to compare, so dtype compatibility alone
    is not enough — a matching semantic type or a high name similarity is also
    required. The best candidate (by score, then declared order) wins one-to-one;
    ambiguous or weak matches are left to be reported as removed + added.
    """
    renamed_from: set[str] = set()
    renamed_to: set[str] = set()
    for old in declared_absent:
        cc = declared[old]
        best: tuple[float, str] | None = None
        for new in unexpected:
            if new in renamed_to:
                continue
            new_family = _normalize_dtype(str(df[new].dtype))
            if cc.dtype is not None and _normalize_dtype(cc.dtype) != new_family:
                continue
            semantic_match = (
                cc.semantic_type is not None
                and new in semantic_now
                and cc.semantic_type == semantic_now[new]
            )
            similarity = _name_similarity(old, new)
            if not semantic_match and similarity < _RENAME_NAME_SIMILARITY:
                continue
            score = 1.0 if semantic_match else similarity
            if best is None or score > best[0]:
                best = (score, new)
        if best is not None:
            new = best[1]
            renamed_from.add(old)
            renamed_to.add(new)
            cats["renamed"][old] = new
            _add(
                findings,
                "schema.renamed_column",
                level="warning",
                status="warned",
                message=f"declared column {old!r} appears renamed to {new!r}",
                column=new,
                baseline_value=old,
                current_value=new,
                metric="column_name",
                details={"evidence_score": round(best[0], 3)},
            )
    return renamed_from, renamed_to


def _detect_column_drift(
    declared: dict[str, ColumnContract],
    current_set: set[str],
    df: Any,
    contract: DataContract,
    semantic_now: dict[str, str],
    findings: list[DriftFinding],
    cats: dict[str, Any],
) -> None:
    """Emit dtype, nullability, and semantic-domain drift for shared columns."""
    for name, cc in declared.items():
        if name not in current_set:
            continue
        series = df[name]
        cur_dtype = str(series.dtype)
        if cc.dtype is not None and _normalize_dtype(cc.dtype) != _normalize_dtype(cur_dtype):
            cats["dtype_changed"].append(name)
            pair: tuple[_Level, _Status] = (
                ("error", "failed") if contract.fail_on_dtype_change else ("warning", "warned")
            )
            level, status = pair
            _add(
                findings,
                "schema.dtype_change",
                level=level,
                status=status,
                message=f"{name!r} expected dtype {cc.dtype}, found {cur_dtype}",
                column=name,
                baseline_value=cc.dtype,
                current_value=cur_dtype,
                metric="dtype",
            )
        if not cc.nullable and bool(series.isna().any()):
            cats["nullable_changed"].append(name)
            _add(
                findings,
                "schema.nullable_change",
                level="error",
                status="failed",
                message=f"{name!r} is declared non-nullable but contains nulls",
                column=name,
                metric="nullable",
            )
        declared_sem = cc.semantic_type
        current_sem = semantic_now.get(name)
        if declared_sem is not None and current_sem is not None and declared_sem != current_sem:
            cats["semantic_changed"].append(name)
            _add(
                findings,
                "schema.semantic_change",
                level="warning",
                status="warned",
                message=f"{name!r} semantic type changed {declared_sem!r} -> {current_sem!r}",
                column=name,
                baseline_value=declared_sem,
                current_value=current_sem,
                metric="semantic_type",
            )


def diff_schema(
    df: Any,
    *,
    contract: DataContract | dict[str, Any],
    on_unexpected: Literal["fail", "warn", "preserve"] = "warn",
    on_missing: Literal["fail", "warn", "ignore"] = "fail",
) -> DriftReport:
    """Explain how a frame's schema differs from a declared contract.

    Baseline-free counterpart to :func:`monitor_contract`: it compares *df*'s
    columns against a :class:`DataContract` and reports **structural** drift
    *before* any repair runs — added/unexpected columns, missing/removed
    columns, likely renames, dtype changes, nullability changes, and
    semantic-domain changes. It is read-only and never mutates *df*; value-level
    enforcement (ranges, allowed values, regex) stays in the contract gate used
    by :func:`monitor_contract` and the cleaning pipeline.

    Parameters
    ----------
    df:
        The DataFrame whose schema should be checked.
    contract:
        A :class:`DataContract` (or its ``to_dict``/``from_dict`` mapping)
        declaring the expected columns.
    on_unexpected:
        Policy for columns present in *df* but absent from the contract —
        ``"fail"`` (error), ``"warn"`` (warning, default), or ``"preserve"``
        (info only; the column is allowed through untouched).
    on_missing:
        Policy for **required** contract columns absent from *df* — ``"fail"``
        (error, default), ``"warn"`` (warning), or ``"ignore"`` (suppressed).
        Optional (``required=False``) columns that are absent are always
        reported at info level only.

    Returns
    -------
    DriftReport
        ``findings`` carry one entry per drift observation (``schema.*`` and
        ``contract.*`` check ids); ``contract_results`` carries the structured
        categorization (``added``, ``removed``, ``renamed``, ``dtype_changed``,
        ``nullable_changed``, ``semantic_changed``, ``unexpected``). Use
        ``.summary()``, ``.to_dict()``, ``.to_json()`` or ``.to_frame()`` to
        export, and ``.passed`` to gate.

    Notes
    -----
    Semantic-domain drift is detected from an optional, caller-supplied
    ``df.attrs["semantic_types"]`` mapping (``{column: semantic_type}``); when a
    declared ``semantic_type`` and the supplied current one disagree a
    ``schema.semantic_change`` finding is emitted. Rename detection is a
    conservative heuristic: a declared-but-absent column and an unexpected
    column are paired only when their dtype families (and any known semantic
    types) match one-to-one.
    """
    if isinstance(contract, dict):
        contract = DataContract.from_dict(contract)
    if on_unexpected not in ("fail", "warn", "preserve"):
        raise ValueError(f"on_unexpected must be fail|warn|preserve, got {on_unexpected!r}")
    if on_missing not in ("fail", "warn", "ignore"):
        raise ValueError(f"on_missing must be fail|warn|ignore, got {on_missing!r}")

    current_cols = [str(c) for c in df.columns]
    current_set = set(current_cols)
    declared = {c.name: c for c in contract.columns}
    semantic_now: dict[str, str] = dict(getattr(df, "attrs", {}).get("semantic_types", {}) or {})

    findings: list[DriftFinding] = []
    cats: dict[str, Any] = {
        "added": [],
        "removed": [],
        "renamed": {},
        "dtype_changed": [],
        "nullable_changed": [],
        "semantic_changed": [],
        "unexpected": [],
    }

    declared_absent = [name for name in declared if name not in current_set]
    unexpected = [c for c in current_cols if c not in declared]

    renamed_from, renamed_to = _detect_renames(
        declared, declared_absent, unexpected, df, semantic_now, findings, cats
    )

    # --- removed / missing declared columns (excluding inferred renames) ---
    for name in declared_absent:
        if name in renamed_from:
            continue
        cats["removed"].append(name)
        cc = declared[name]
        if not cc.required:
            _add(
                findings,
                "schema.removed_column",
                level="info",
                status="passed",
                message=f"optional declared column {name!r} is absent",
                column=name,
                metric="column_presence",
            )
            continue
        policy = _POLICY[on_missing]
        if policy is None:
            continue
        level, status = policy
        _add(
            findings,
            "schema.removed_column",
            level=level,
            status=status,
            message=f"required declared column {name!r} is missing",
            column=name,
            metric="column_presence",
        )

    # --- unexpected / added columns (excluding inferred renames) ---
    for name in unexpected:
        if name in renamed_to:
            continue
        cats["added"].append(name)
        cats["unexpected"].append(name)
        policy = _POLICY[on_unexpected]
        if policy is None:  # pragma: no cover - not reachable for on_unexpected
            continue
        level, status = policy
        _add(
            findings,
            "contract.unexpected_column",
            level=level,
            status=status,
            message=f"column {name!r} is present but not declared in the contract",
            column=name,
            metric="column_presence",
        )

    _detect_column_drift(declared, current_set, df, contract, semantic_now, findings, cats)

    return DriftReport(
        baseline_name=contract.name,
        baseline_version=contract.version,
        findings=findings,
        contract_results=cats,
    )
