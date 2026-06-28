"""Provenance-aware cleaning for OCR / PDF / document-derived tables.

When a table is extracted from a document, each field carries provenance — the
source file, page, region, a parser confidence, and an extraction timestamp.
FreshData is **not** a PDF parser; it is the post-extraction normalization and
audit layer. Passing ``source_provenance=`` to :func:`freshdata.clean` (or
``clean_enterprise``) preserves that metadata in the report and **warns when a
low-confidence extracted field is cleaned or coerced**, so a reviewer can tell a
trustworthy repair from one that silently "fixed" a mis-read cell.

Provenance is a mapping of *input* column name to a metadata dict::

    {"amount": {"parser_confidence": 0.55, "source_file": "invoice.pdf",
                "page": 3, "region": "table-1", "extracted_at": "2024-05-01T10:00:00Z"}}

Only ``parser_confidence`` drives warnings; the rest is carried through verbatim.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .report import CleanReport

#: Default parser-confidence below which a coercion/repair is flagged for review.
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

#: ``Action.step`` substrings that indicate a value-changing repair (vs. a
#: preserve/review log entry). Used to decide whether a column was "coerced".
_MODIFYING_STEPS = (
    "dtype",
    "whitespace",
    "sentinel",
    "impute",
    "outlier",
    "cast",
    "coerce",
    "normalize",
    "duplicate",
    "fill",
)

_PROVENANCE_KEYS = ("parser_confidence", "source_file", "page", "region", "extracted_at")


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def normalize_provenance(
    source_provenance: dict[str, Any], columns: list[str]
) -> dict[str, dict[str, Any]]:
    """Validate and normalize a ``{column: metadata}`` provenance mapping.

    Unknown metadata keys are kept (carried through for audit); ``page`` is
    coerced to int when present and ``parser_confidence`` to float. A
    ``ValueError`` is raised for a non-mapping or a confidence outside ``[0, 1]``.
    """
    if not isinstance(source_provenance, dict):
        raise TypeError("source_provenance must be a {column: metadata} mapping")
    known = set(columns) | {_normalize_name(c) for c in columns}
    out: dict[str, dict[str, Any]] = {}
    for col, meta in source_provenance.items():
        if not isinstance(meta, dict):
            raise TypeError(f"provenance for {col!r} must be a dict, got {type(meta).__name__}")
        record = dict(meta)
        conf = record.get("parser_confidence")
        if conf is not None:
            conf = float(conf)
            if not 0.0 <= conf <= 1.0:
                raise ValueError(f"parser_confidence for {col!r} must be in [0, 1], got {conf}")
            record["parser_confidence"] = conf
        if record.get("page") is not None:
            record["page"] = int(record["page"])
        record["known_column"] = col in known or _normalize_name(col) in known
        out[str(col)] = record
    return out


def _modified_columns(report: CleanReport) -> set[str]:
    """Columns the report shows as value-changed (coerced/imputed/repaired)."""
    modified: set[str] = set(report.columns_imputed)
    for action in report.actions:
        if action.column is None:
            continue
        step = action.step.lower()
        if any(tok in step for tok in _MODIFYING_STEPS) and (action.count or 0) > 0:
            modified.add(action.column)
    return modified | {_normalize_name(c) for c in modified}


def annotate_provenance(
    report: CleanReport,
    source_provenance: dict[str, Any],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> None:
    """Attach provenance to *report* and warn on low-confidence coercions.

    Mutates *report* in place: sets ``report.source_provenance`` (a JSON-friendly
    per-column summary) and appends a warning + manual-review recommendation for
    every low-confidence field that was coerced or repaired.
    """
    provenance = normalize_provenance(source_provenance, [str(c) for c in report.columns_preserved]
                                      + report.columns_imputed + list(report.columns_dropped))
    modified = _modified_columns(report)
    summary: dict[str, dict[str, Any]] = {}
    for col, meta in provenance.items():
        conf = meta.get("parser_confidence")
        was_modified = col in modified or _normalize_name(col) in modified
        low_conf = conf is not None and conf < confidence_threshold
        flagged = bool(low_conf and was_modified)
        summary[col] = {k: meta.get(k) for k in _PROVENANCE_KEYS}
        summary[col]["modified"] = was_modified
        summary[col]["low_confidence_repair"] = flagged
        if flagged:
            src = meta.get("source_file", "?")
            page = meta.get("page")
            where = f"{src}" + (f" p.{page}" if page is not None else "")
            report.warnings.append(
                f"low-confidence extracted field {col!r} (parser_confidence="
                f"{conf:.2f} from {where}) was coerced/repaired — verify the source"
            )
            report.recommendations.append(
                f"review {col!r}: repaired despite low parser confidence ({conf:.2f}); "
                "confirm the extracted value before trusting the cleaned result"
            )
    report.source_provenance = summary
