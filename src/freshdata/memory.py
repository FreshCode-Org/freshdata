"""Persistent cleaning memory — record human-approved cleaning decisions and
replay them on similar future data.

    memory = fd.learn_cleaning_memory(df, decisions=reviewed, dataset_id="crm")
    cleaned, report = fd.clean(df2, memory=memory, return_report=True)

Memory stores a dataset signature, column roles, accepted/rejected decisions,
thresholds, value-pattern decisions and stakeholder exceptions, with timestamp +
version metadata. On replay it checks the new data still looks like the data it
learned from; if it drifted too much, replay is **blocked and explained** rather
than applied blindly. Storage is local and server-free (JSON, or SQLite).

Light core: stdlib only (``json``, ``sqlite3``, ``hashlib``).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from .render import html as H
from .render.mixins import SimpleHtmlReport

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .report import CleanReport

#: Config fields a memory is allowed to override on replay (safe, declarative).
_ALLOWED_THRESHOLDS = frozenset({
    "missing_threshold_low", "missing_threshold_medium", "missing_threshold_high",
    "duplicate_threshold", "numeric_threshold", "datetime_threshold",
    "outlier_factor", "category_threshold",
})


def _coarse_dtype(dtype: Any) -> str:
    s = str(dtype)
    if s.startswith(("int", "Int", "uint", "UInt")):
        return "integer"
    if s.startswith(("float", "Float")):
        return "float"
    if s.startswith(("datetime", "date")):
        return "datetime"
    if s in ("bool", "boolean"):
        return "boolean"
    if s in ("object", "string", "category"):
        return "text"
    return s


def _signature(df: pd.DataFrame) -> dict[str, Any]:
    cols = {str(c): _coarse_dtype(df[c].dtype) for c in df.columns}
    import hashlib

    digest = hashlib.sha256(
        json.dumps(cols, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {"columns": cols, "n_cols": len(cols), "hash": digest}


def _normalize_decisions(decisions: Any) -> tuple[list[dict], list[dict]]:
    """Split *decisions* into (accepted, rejected) lists of plain dicts.

    Accepts a :class:`CleanReport`, a list of :class:`~freshdata.Action`, a list
    of dicts, or a DataFrame. A decision's ``status`` (``"accepted"``/
    ``"approved"`` vs ``"rejected"``/``"skipped"``) routes it; unmarked decisions
    are treated as accepted.
    """
    rows: list[dict]
    if hasattr(decisions, "to_frame") and hasattr(decisions, "actions"):  # CleanReport
        rows = [{"column": a.column, "step": a.step, "action": a.step,
                 "status": getattr(a, "status", "accepted"),
                 "description": a.description} for a in decisions.actions]
    elif isinstance(decisions, pd.DataFrame):
        rows = decisions.to_dict("records")
    else:
        rows = []
        for d in decisions or []:
            if isinstance(d, dict):
                rows.append(dict(d))
            else:  # Action-like
                rows.append({
                    "column": getattr(d, "column", None),
                    "step": getattr(d, "step", None),
                    "action": getattr(d, "step", None),
                    "status": getattr(d, "status", "accepted"),
                })
    accepted: list[dict] = []
    rejected: list[dict] = []
    for r in rows:
        status = str(r.get("status", "accepted")).lower()
        (rejected if status in ("rejected", "skipped", "declined") else accepted).append(r)
    return accepted, rejected


@dataclass
class MemoryMatch:
    """Whether a memory's signature is compatible with a new frame."""

    ok: bool
    overlap: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class CleaningMemory(SimpleHtmlReport):
    """A reusable record of accepted cleaning decisions for a dataset."""

    dataset_id: str
    signature: dict[str, Any] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    accepted: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    value_patterns: dict[str, Any] = field(default_factory=dict)
    exceptions: list[dict] = field(default_factory=list)
    created_at: str = ""
    freshdata_version: str = ""

    # -- matching / replay ---------------------------------------------------

    def match(self, df: pd.DataFrame, *, min_overlap: float = 0.7) -> MemoryMatch:
        """Return whether *df* is similar enough to safely replay this memory."""
        stored = self.signature.get("columns", {})
        if not stored:
            return MemoryMatch(False, 0.0, ["memory has no stored signature"])
        current = {str(c): _coarse_dtype(df[c].dtype) for c in df.columns}
        present = [c for c in stored if c in current]
        overlap = len(present) / len(stored)
        reasons: list[str] = []
        missing = [c for c in stored if c not in current]
        if missing:
            reasons.append(f"{len(missing)} learned column(s) missing: "
                           f"{', '.join(missing[:5])}")
        dtype_changed = [c for c in present if current[c] != stored[c]]
        if dtype_changed:
            reasons.append(f"{len(dtype_changed)} column(s) changed type: "
                           f"{', '.join(dtype_changed[:5])}")
        ok = overlap >= min_overlap and not dtype_changed
        if ok and not reasons:
            reasons.append("signature matches learned dataset")
        return MemoryMatch(ok, round(overlap, 3), reasons)

    def config_overrides(self) -> dict[str, Any]:
        """Declarative ``CleanConfig`` overrides implied by accepted decisions."""
        overrides: dict[str, Any] = {}
        preserve = [str(d["column"]) for d in self.accepted
                    if d.get("column") and str(d.get("action", "")).startswith("preserve")]
        preserve += [str(e["column"]) for e in self.exceptions if e.get("column")]
        if preserve:
            overrides["preserve_columns"] = tuple(dict.fromkeys(preserve))
        ids = tuple(c for c, r in self.roles.items() if r in ("id", "identifier", "key"))
        if ids:
            overrides["id_columns"] = ids
        target = next((c for c, r in self.roles.items() if r == "target"), None)
        if target:
            overrides["target_column"] = target
        for k, v in self.thresholds.items():
            if k in _ALLOWED_THRESHOLDS:
                overrides[k] = v
        return overrides

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "signature": self.signature,
            "roles": dict(self.roles),
            "accepted": list(self.accepted),
            "rejected": list(self.rejected),
            "thresholds": dict(self.thresholds),
            "value_patterns": dict(self.value_patterns),
            "exceptions": list(self.exceptions),
            "created_at": self.created_at,
            "freshdata_version": self.freshdata_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CleaningMemory:
        return cls(
            dataset_id=payload.get("dataset_id", ""),
            signature=payload.get("signature", {}),
            roles=payload.get("roles", {}),
            accepted=payload.get("accepted", []),
            rejected=payload.get("rejected", []),
            thresholds=payload.get("thresholds", {}),
            value_patterns=payload.get("value_patterns", {}),
            exceptions=payload.get("exceptions", []),
            created_at=payload.get("created_at", ""),
            freshdata_version=payload.get("freshdata_version", ""),
        )

    def to_json(self, path: str | None = None) -> str:
        """Serialize to JSON. With *path*, also write it (``.sqlite``/``.db`` →
        a tiny key/value SQLite store). Returns the JSON string."""
        text = json.dumps(self.to_dict(), indent=2, default=str)
        if path is not None:
            if str(path).endswith((".sqlite", ".db")):
                _sqlite_write(path, self.dataset_id, text)
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
        return text

    def diff(self, other: CleaningMemory) -> dict[str, Any]:
        """Structural diff against *other* memory."""
        def _acts(m: CleaningMemory) -> set[tuple]:
            return {(d.get("column"), d.get("action") or d.get("step")) for d in m.accepted}

        a, b = _acts(self), _acts(other)
        cols_self = set(self.signature.get("columns", {}))
        cols_other = set(other.signature.get("columns", {}))
        return {
            "dataset_id": (self.dataset_id, other.dataset_id),
            "signature_changed": self.signature.get("hash") != other.signature.get("hash"),
            "columns_added": sorted(cols_other - cols_self),
            "columns_removed": sorted(cols_self - cols_other),
            "accepted_added": sorted(str(x) for x in (b - a)),
            "accepted_removed": sorted(str(x) for x in (a - b)),
            "threshold_changes": {
                k: (self.thresholds.get(k), other.thresholds.get(k))
                for k in set(self.thresholds) | set(other.thresholds)
                if self.thresholds.get(k) != other.thresholds.get(k)
            },
        }

    def summary(self) -> str:
        return (
            f"freshdata cleaning memory '{self.dataset_id}'\n"
            f"  learned: {self.created_at or 'n/a'} (v{self.freshdata_version or '?'})\n"
            f"  signature: {self.signature.get('n_cols', 0)} column(s), "
            f"hash {self.signature.get('hash', '?')}\n"
            f"  accepted: {len(self.accepted)}  rejected: {len(self.rejected)}  "
            f"exceptions: {len(self.exceptions)}\n"
            f"  roles: {len(self.roles)}  thresholds: {len(self.thresholds)}"
        )

    def __str__(self) -> str:
        return self.summary()

    # -- HTML ----------------------------------------------------------------

    def _html_title(self) -> str:
        return f"freshdata cleaning memory — {self.dataset_id}"

    def _html_subtitle(self) -> str | None:
        return (f"learned {self.created_at or 'n/a'} · "
                f"signature {self.signature.get('hash', '?')}")

    def _html_sections(self) -> list[str]:
        cards = H.scorecards([
            ("columns", self.signature.get("n_cols", 0)),
            ("accepted", len(self.accepted)),
            ("rejected", len(self.rejected)),
            ("exceptions", len(self.exceptions)),
            ("thresholds", len(self.thresholds)),
        ])
        acc = ""
        if self.accepted:
            rows = [[str(d.get("column") or "—"), str(d.get("action") or d.get("step") or ""),
                     str(d.get("description") or "")] for d in self.accepted]
            acc = H.section("Accepted decisions",
                            H.table(["column", "action", "detail"], rows))
        dl = H.json_download(f"{self.dataset_id}_memory.json", self.to_dict(), "⬇ JSON")
        return [cards, acc, dl]


def learn_cleaning_memory(
    df: pd.DataFrame,
    decisions: Any,
    dataset_id: str,
    *,
    thresholds: dict[str, float] | None = None,
    value_patterns: dict[str, Any] | None = None,
    exceptions: list[dict] | None = None,
    roles: dict[str, str] | None = None,
) -> CleaningMemory:
    """Learn a :class:`CleaningMemory` from reviewed decisions on *df*."""
    from . import __version__
    from .semantic.memory import learn_semantic_repairs

    accepted, rejected = _normalize_decisions(decisions)
    if roles is None:
        try:
            from .api import infer_roles

            roles = {str(r["column"]): str(r["role"])
                     for r in infer_roles(df).to_dict("records")}
        except Exception:  # pragma: no cover - role inference is best-effort
            roles = {}

    vp = dict(value_patterns or {})
    semantic_repairs = learn_semantic_repairs(decisions)
    if semantic_repairs:
        for repair in semantic_repairs:
            repair["dataset_id"] = dataset_id
        vp["semantic_repairs"] = list(vp.get("semantic_repairs", [])) + semantic_repairs

    return CleaningMemory(
        dataset_id=dataset_id,
        signature=_signature(df),
        roles=roles,
        accepted=accepted,
        rejected=rejected,
        thresholds=dict(thresholds or {}),
        value_patterns=vp,
        exceptions=list(exceptions or []),
        created_at=datetime.now(timezone.utc).isoformat(),
        freshdata_version=__version__,
    )


def load_cleaning_memory(path: str) -> CleaningMemory:
    """Load a memory previously saved with :meth:`CleaningMemory.to_json`."""
    if str(path).endswith((".sqlite", ".db")):
        text = _sqlite_read(path)
    else:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    return CleaningMemory.from_dict(json.loads(text))


# -- minimal server-free SQLite key/value store -----------------------------

def _sqlite_write(path: str, dataset_id: str, text: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cleaning_memory "
            "(dataset_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO cleaning_memory(dataset_id, payload) VALUES(?, ?) "
            "ON CONFLICT(dataset_id) DO UPDATE SET payload=excluded.payload",
            (dataset_id, text))
        conn.commit()
    finally:
        conn.close()


def _sqlite_read(path: str, dataset_id: str | None = None) -> str:
    conn = sqlite3.connect(path)
    try:
        if dataset_id is not None:
            row = conn.execute(
                "SELECT payload FROM cleaning_memory WHERE dataset_id=?",
                (dataset_id,)).fetchone()
        else:
            row = conn.execute(
                "SELECT payload FROM cleaning_memory LIMIT 1").fetchone()
        if not row:
            raise KeyError(f"no cleaning memory found in {path!r}")
        return str(row[0])
    finally:
        conn.close()


def annotate_report(report: CleanReport, memory: CleaningMemory, match: MemoryMatch) -> None:
    """Record on *report* whether memory was used, and mark replayed actions.

    Explains plainly when memory was applied and when it was ignored because the
    data drifted too much from what the memory learned.
    """
    from dataclasses import replace

    if not match.ok:
        report.add_warning(
            f"cleaning memory '{memory.dataset_id}' was ignored: the data changed "
            f"too much from what it learned ({'; '.join(match.reasons)})")
        report.add_recommendation(
            f"re-learn cleaning memory for '{memory.dataset_id}' on the new data, "
            "or relax the match tolerance, before relying on replay")
        return

    accepted = {(str(d.get("column")) if d.get("column") is not None else None,
                 str(d.get("action") or d.get("step")))
                for d in memory.accepted}
    n_marked = 0
    new_actions = []
    for a in report.actions:
        key = (str(a.column) if a.column is not None else None, str(a.step))
        # Semantic actions replay via the dedicated, policy-gated semantic-memory
        # pathway (see freshdata.semantic.apply); marking them here too would let
        # this coarse column+step match silently override a suggested/skipped,
        # high-risk semantic decision with "approved".
        if a.step != "semantic" and key in accepted:
            new_actions.append(replace(a, status="approved", memory_influenced=True))
            n_marked += 1
        else:
            new_actions.append(a)
    report.actions = new_actions
    report.add(
        "memory",
        f"applied cleaning memory '{memory.dataset_id}': {n_marked} decision(s) "
        f"replayed (signature overlap {match.overlap:.0%})",
        status="approved", memory_influenced=True,
    )


def apply_memory(
    df: pd.DataFrame, memory: CleaningMemory, base_options: dict[str, Any]
) -> tuple[dict[str, Any], MemoryMatch]:
    """Return ``(effective_options, match)`` for replaying *memory* on *df*.

    When the signature matches, the memory's declarative overrides are merged
    into *base_options*. When it does not, *base_options* is returned unchanged
    (the caller discloses that memory was ignored).
    """
    match = memory.match(df)
    if not match.ok:
        return dict(base_options), match
    options = dict(base_options)
    for key, value in memory.config_overrides().items():
        if key == "preserve_columns":
            existing = tuple(options.get("preserve_columns", ()) or ())
            options[key] = tuple(dict.fromkeys((*existing, *value)))
        else:
            options.setdefault(key, value)
    return options, match
