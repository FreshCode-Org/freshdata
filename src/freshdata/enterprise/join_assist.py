"""Dirty-join assistant — reviewable fuzzy join suggestions for messy keys.

    matches = fd.suggest_join_keys(
        left_df, right_df,
        on=["company_name", "address"],
        exact_within=["country"],
    )

Suggests exact join keys, ranks fuzzy candidate matches with confidence and
per-field explanations, groups candidates, and flags ambiguous ones — and
**never silently joins** low-confidence matches. Reuses the entity-resolution
string-similarity primitives (Jaro-Winkler / Levenshtein); no extra deps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from ..render import html as H
from ..render.mixins import SimpleHtmlReport
from .entity_resolution import jaro_winkler, levenshtein_similarity


def _similarity(a: str, b: str) -> float:
    """Blended string similarity in [0, 1] (Jaro-Winkler + edit distance)."""
    if a == b:
        return 1.0
    return round(0.5 * jaro_winkler(a, b) + 0.5 * levenshtein_similarity(a, b), 4)


def _is_missing(value: Any) -> bool:
    """True for None / NaN / NaT / ``pd.NA`` / ``""`` — a key carrying no evidence."""
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # array-like cell: not a missing scalar
        return False


def _key_text(value: Any) -> str:
    """Comparable text for a key value; integral floats render without ``.0``.

    A numeric key column holding a missing value is promoted to float64, so
    ``101`` and ``101.0`` must compare equal (#273). Only finite, integral
    floats below 2**53 (exactly representable) are rewritten.
    """
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isfinite(f) and f.is_integer() and abs(f) < 2**53:
            return str(int(f))
    return str(value)


def _field_similarity(a: Any, b: Any) -> float:
    """Similarity of two key cells; a missing value on either side scores 0 (#272)."""
    if _is_missing(a) or _is_missing(b):
        return 0.0
    return _similarity(_key_text(a), _key_text(b))


@dataclass(frozen=True)
class JoinCandidate:
    """One suggested (left, right) match with per-field evidence."""

    left_index: Any
    right_index: Any
    score: float
    status: str  # "match" | "ambiguous" | "review"
    field_scores: dict[str, float]
    block: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_index": self.left_index,
            "right_index": self.right_index,
            "score": self.score,
            "status": self.status,
            "field_scores": self.field_scores,
            "block": self.block,
        }


@dataclass
class JoinKeyReport(SimpleHtmlReport):
    """Result of :func:`suggest_join_keys`."""

    on: list[str] = field(default_factory=list)
    exact_within: list[str] = field(default_factory=list)
    exact_keys: list[dict] = field(default_factory=list)
    candidates: list[JoinCandidate] = field(default_factory=list)
    threshold: float = 0.85
    review_threshold: float = 0.65
    pairs_compared: int = 0
    truncated: bool = False

    @property
    def matches(self) -> list[JoinCandidate]:
        return [c for c in self.candidates if c.status == "match"]

    @property
    def ambiguous(self) -> list[JoinCandidate]:
        return [c for c in self.candidates if c.status == "ambiguous"]

    @property
    def review(self) -> list[JoinCandidate]:
        return [c for c in self.candidates if c.status == "review"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [c.to_dict() for c in self.candidates],
            columns=["left_index", "right_index", "score", "status",
                     "field_scores", "block"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "on": list(self.on),
            "exact_within": list(self.exact_within),
            "exact_keys": list(self.exact_keys),
            "threshold": self.threshold,
            "review_threshold": self.review_threshold,
            "pairs_compared": self.pairs_compared,
            "truncated": self.truncated,
            "n_matches": len(self.matches),
            "n_ambiguous": len(self.ambiguous),
            "n_review": len(self.review),
            "candidates": [c.to_dict() for c in self.candidates],
        }

    def summary(self) -> str:
        lines = [
            "freshdata join-key suggestions",
            f"  on: {', '.join(self.on)}"
            + (f"   blocked within: {', '.join(self.exact_within)}"
               if self.exact_within else ""),
        ]
        if self.exact_keys:
            for ek in self.exact_keys:
                lines.append(f"  exact-key candidate: {ek['column']} "
                             f"(value overlap {ek['overlap']:.0%})")
        lines.append(
            f"  {len(self.matches)} confident match(es), {len(self.ambiguous)} ambiguous, "
            f"{len(self.review)} to review ({self.pairs_compared:,} pair(s) compared)")
        if self.truncated:
            lines.append("  ! candidate generation was truncated; add exact_within= "
                         "blocking to compare more safely")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    # -- HTML ----------------------------------------------------------------

    def _html_title(self) -> str:
        return "freshdata join-key suggestions"

    def _html_subtitle(self) -> str | None:
        return (f"on {', '.join(self.on)}"
                + (f" · blocked within {', '.join(self.exact_within)}"
                   if self.exact_within else "")
                + f" · {self.pairs_compared:,} pair(s) compared")

    def _html_sections(self) -> list[str]:
        cards = H.scorecards([
            ("confident", len(self.matches)),
            ("ambiguous", len(self.ambiguous)),
            ("to review", len(self.review)),
        ])
        exact = ""
        if self.exact_keys:
            rows = [[ek["column"], f"{ek['overlap']:.0%}",
                     "yes" if ek["recommended"] else "no"] for ek in self.exact_keys]
            exact = H.section("Exact-key candidates",
                              H.table(["column", "value overlap", "recommended"], rows))
        rows = []
        for c in sorted(self.candidates, key=lambda x: x.score, reverse=True):
            fields = "; ".join(f"{k}={v:.2f}" for k, v in c.field_scores.items())
            badge = {"match": "#1a7f37", "ambiguous": "#9a6700",
                     "review": "#57606a"}.get(c.status, "#57606a")
            rows.append([str(c.left_index), str(c.right_index), f"{c.score:.2f}",
                         H.badge(c.status, badge), fields, c.block])
        tbl = H.section("Candidate matches", H.filterable_table(
            "fd-join", ["left", "right", "score", "status", "field scores", "block"],
            rows, filters={"status": 3}, raw_columns=[3]) if rows
            else "<div class='fd-meta'>no candidate matches above the review threshold</div>")
        dl = H.json_download("join_keys.json", self.to_dict(), "⬇ JSON")
        return [cards, exact, tbl, dl]


def _key_values(series: pd.Series) -> set[str]:
    return {_key_text(v) for v in series.tolist() if not _is_missing(v)}


def _exact_key_overlap(left: pd.DataFrame, right: pd.DataFrame, col: str) -> float:
    lv = _key_values(left[col])
    rv = _key_values(right[col])
    if not lv or not rv:
        return 0.0
    return len(lv & rv) / min(len(lv), len(rv))


def suggest_join_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: list[str],
    exact_within: list[str] | None = None,
    threshold: float = 0.85,
    review_threshold: float = 0.65,
    max_pairs: int = 50_000,
) -> JoinKeyReport:
    """Suggest exact and fuzzy join keys between *left* and *right*.

    Parameters
    ----------
    left, right:
        Frames to be joined (never modified).
    on:
        Candidate join columns compared fuzzily.
    exact_within:
        Columns that must match **exactly** to form a blocking key — candidates
        are only generated within the same block, which keeps the comparison
        cheap and avoids nonsense cross-block matches.
    threshold:
        Score at/above which a single best match is "confident".
    review_threshold:
        Lower bound for showing a near-miss for human review.
    max_pairs:
        Safety cap on comparisons; exceeding it truncates and is disclosed.
    """
    on = [c for c in on if c in left.columns and c in right.columns]
    if not on:
        raise ValueError("none of the `on` columns are present in both frames")
    blocking = [c for c in (exact_within or []) if c in left.columns and c in right.columns]

    exact_keys = []
    for c in on:
        overlap = _exact_key_overlap(left, right, c)
        exact_keys.append(
            {"column": c, "overlap": round(overlap, 4), "recommended": overlap >= 0.95})

    # Rows are addressed by *position* throughout: row labels need not be unique
    # (e.g. after pd.concat), and a label lookup would return a Series (#231).
    # The original labels are only used in the reported candidates.
    def block_keys(frame: pd.DataFrame) -> list[str]:
        """Blocking key per row (exact match on blocking columns); "*" if none."""
        if not blocking:
            return ["*"] * len(frame)
        cols = [frame[c].tolist() for c in blocking]
        return ["|".join(_key_text(v) for v in vals) for vals in zip(*cols)]

    left_blocks: dict[str, list[int]] = {}
    for pos, key in enumerate(block_keys(left)):
        left_blocks.setdefault(key, []).append(pos)
    right_blocks: dict[str, list[int]] = {}
    for pos, key in enumerate(block_keys(right)):
        right_blocks.setdefault(key, []).append(pos)

    left_vals = {c: left[c].tolist() for c in on}
    right_vals = {c: right[c].tolist() for c in on}

    candidates: list[tuple[int, JoinCandidate]] = []
    pairs = 0
    truncated = False
    # best-per-left tracking for ambiguity detection (keyed by left position)
    per_left: dict[int, list[JoinCandidate]] = {}

    for bkey, l_pos in left_blocks.items():
        r_pos = right_blocks.get(bkey, [])
        for lp, rp in product(l_pos, r_pos):
            if pairs >= max_pairs:
                truncated = True
                break
            pairs += 1
            fscores = {
                c: _field_similarity(left_vals[c][lp], right_vals[c][rp]) for c in on
            }
            score = round(sum(fscores.values()) / len(on), 4)
            if score >= review_threshold:
                cand = JoinCandidate(
                    left.index[lp], right.index[rp], score, "review", fscores, bkey)
                candidates.append((lp, cand))
                per_left.setdefault(lp, []).append(cand)
        if truncated:
            break

    # Resolve status: confident vs ambiguous vs review.
    resolved: list[JoinCandidate] = []
    from dataclasses import replace

    for lp, cand in candidates:
        siblings = [c for c in per_left[lp] if c.score >= threshold]
        if cand.score >= threshold:
            if len([c for c in siblings
                    if abs(c.score - max(s.score for s in siblings)) <= 0.05]) > 1:
                resolved.append(replace(cand, status="ambiguous"))
            else:
                best = max(siblings, key=lambda c: c.score)
                resolved.append(
                    replace(cand, status="match" if cand is best else "ambiguous"))
        else:
            resolved.append(cand)  # stays "review"

    resolved.sort(key=lambda c: c.score, reverse=True)
    return JoinKeyReport(
        on=on, exact_within=blocking, exact_keys=exact_keys, candidates=resolved,
        threshold=threshold, review_threshold=review_threshold,
        pairs_compared=pairs, truncated=truncated,
    )
