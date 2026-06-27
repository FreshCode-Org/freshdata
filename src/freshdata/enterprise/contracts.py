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
class DriftReport:
    """The outcome of comparing a frame against a baseline."""

    baseline_name: str
    baseline_version: str
    findings: list[DriftFinding] = field(default_factory=list)
    trust_score: float | None = None
    distribution_drift: dict[str, Any] = field(default_factory=dict)
    contract_results: dict[str, Any] = field(default_factory=dict)

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
        }

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
            out.append(QualityFinding.create(
                severity=f.level,
                step="drift",
                column=f.column,
                rule_name=f.check_id,
                message=f.message,
                observed_value=f.current_value,
                expected_condition=expected,
                action_taken=f.status,
                lineage_run_id=lineage_run_id,
                extra={"metric": f.metric, "baseline_value": f.baseline_value,
                       "threshold": f.threshold, **(f.details or {})},
            ))
        return out

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str, sort_keys=True)

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


def _profile_column(
    series: pd.Series, *, n_rows: int, include_samples: bool
) -> ColumnBaseline:
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
        cb.frequencies = (
            {_label(k): float(v) / total for k, v in top.items()} if total else {}
        )
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


def _check_contract(
    findings: list[DriftFinding],
    contract: DataContract,
    current: dict[str, ColumnBaseline],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    cur_cols = set(current)
    declared = {c.name for c in contract.columns}

    if not contract.allow_extra_columns:
        for col in current:
            if col not in declared:
                _add(
                    findings,
                    "contract.unexpected_column",
                    level="error",
                    status="failed",
                    message=f"column {col!r} is not declared and extra columns are forbidden",
                    column=col,
                )

    for cc in contract.columns:
        col = cc.name
        passes = True
        if col not in cur_cols:
            if cc.required and contract.fail_on_missing_required:
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
    if _normalize_dtype(cc.dtype) == _normalize_dtype(cb.dtype):
        return True
    pair: tuple[_Level, _Status] = (
        ("error", "failed") if contract.fail_on_dtype_change else ("warning", "warned")
    )
    level, status = pair
    _add(
        findings,
        "contract.dtype",
        level=level,
        status=status,
        message=f"{cc.name!r} expected dtype {cc.dtype}, found {cb.dtype}",
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


def _contract_unique(
    findings: list[DriftFinding], cc: ColumnContract, cb: ColumnBaseline
) -> bool:
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


def _contract_values(
    findings: list[DriftFinding], cc: ColumnContract, series: pd.Series
) -> bool:
    ok = True
    non_null = series.dropna()
    if cc.allowed_values:
        allowed = set(cc.allowed_values)
        offenders = sorted({v for v in non_null.tolist() if v not in allowed})
        if offenders:
            ok = False
            _add(
                findings,
                "contract.allowed_values",
                level="error",
                status="failed",
                message=f"{cc.name!r} has {len(offenders)} value(s) outside the allowed set",
                column=cc.name,
                threshold=list(cc.allowed_values),
                metric="allowed_values",
                details={"offending_sample": [str(v) for v in offenders[:5]]},
            )
    if cc.min_value is not None or cc.max_value is not None:
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric):
            if cc.min_value is not None and float(numeric.min()) < cc.min_value:
                ok = False
                _add(
                    findings,
                    "contract.min_value",
                    level="error",
                    status="failed",
                    message=f"{cc.name!r} minimum {float(numeric.min())} below {cc.min_value}",
                    column=cc.name,
                    current_value=_round(float(numeric.min())),
                    threshold=cc.min_value,
                    metric="min_value",
                )
            if cc.max_value is not None and float(numeric.max()) > cc.max_value:
                ok = False
                _add(
                    findings,
                    "contract.max_value",
                    level="error",
                    status="failed",
                    message=f"{cc.name!r} maximum {float(numeric.max())} above {cc.max_value}",
                    column=cc.name,
                    current_value=_round(float(numeric.max())),
                    threshold=cc.max_value,
                    metric="max_value",
                )
    if cc.regex:
        pattern = re.compile(cc.regex)
        as_str = non_null.astype("string")
        violations = int((~as_str.str.fullmatch(pattern)).fillna(True).sum())
        if violations:
            ok = False
            _add(
                findings,
                "contract.regex",
                level="error",
                status="failed",
                message=f"{cc.name!r} has {violations} value(s) not matching {cc.regex!r}",
                column=cc.name,
                threshold=cc.regex,
                current_value=violations,
                metric="regex",
            )
    return ok


def compare_to_baseline(
    df: Any,
    baseline: DatasetBaseline,
    *,
    contract: DataContract | None = None,
    drift_config: DriftConfig | None = None,
    trust_score: float | None = None,
) -> DriftReport:
    """Compare *df* against *baseline*; return a :class:`DriftReport`.

    Read-only: *df* is never mutated. ``contract`` overrides any contract stored
    in the baseline. ``trust_score`` overrides the computed Data Trust Score for
    the gate (useful to feed a score already computed elsewhere).
    """
    cfg = drift_config or DriftConfig()
    frame = to_pandas(df)
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
    )


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
