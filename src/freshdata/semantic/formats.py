"""Deterministic format experts: email, phone (region-aware), reference lists.

Phase-2 additions to the semantic layer. Like every expert in
:mod:`freshdata.semantic.experts`, these operate on a column's *distinct*
values only (never row-by-row), are pure functions of their inputs (no
network, no models, no randomness), and only ever *propose* — the policy gate
in :mod:`freshdata.semantic.policy` decides whether a proposal is applied,
suggested, or skipped.

Design rules encoded here:

- Only unambiguous mechanical repairs (whitespace, casing, one duplicated
  ``@``, separator normalization) score high enough to auto-apply.
- Anything ambiguous — ``"bob[at]gmail.com"``, a fuzzy typo with two close
  reference candidates, a phone number with the wrong digit count — is
  *flagged* (proposal with ``proposed_value=None``) or *suggested*, never
  silently changed.
"""

from __future__ import annotations

import re

import pandas as pd

from .scoring import make_proposal
from .types import SemanticColumnInfo, SemanticEvidence, SemanticProposal

# --------------------------------------------------------------------------- #
# Email
# --------------------------------------------------------------------------- #

#: Conservative "clean email" shape: local@domain.tld with an alphabetic TLD.
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")
#: Punctuation that commonly wraps a pasted address ("<a@b.com>", "'a@b.com',").
_EMAIL_WRAPPERS = "<>\"'`,;:()[]{}"
_AT_SPACING_RE = re.compile(r"^(\S+)\s+@\s*(\S+)$|^(\S+)\s*@\s+(\S+)$")


def repair_email(raw: str) -> tuple[str, list[str]] | None:
    """Mechanically repair *raw* into a valid email, or ``None`` if unsafe.

    Returns ``(repaired, steps)`` where *steps* names each transformation
    applied (the audit evidence). ``None`` means no unambiguous repair exists
    — the value should be flagged, not changed.
    """
    steps: list[str] = []
    s = raw.strip()
    if s != raw:
        steps.append("trimmed surrounding whitespace")

    unwrapped = s.strip(_EMAIL_WRAPPERS).rstrip(".")
    if unwrapped != s:
        steps.append("removed surrounding punctuation")
        s = unwrapped

    # Whitespace around a single "@": "alice @ example.com" -> joined.
    if " " in s or "\t" in s:
        collapsed = re.sub(r"\s*@\s*", "@", s)
        if "@" in s and collapsed.count("@") == 1 and not re.search(r"\s", collapsed):
            steps.append('joined whitespace around "@"')
            s = collapsed
        else:
            return None  # whitespace inside the address is not mechanically fixable

    # Exactly one doubled "@@" (and no other "@") is an unambiguous typo.
    if s.count("@") == 2 and "@@" in s:
        s = s.replace("@@", "@", 1)
        steps.append('collapsed doubled "@@"')

    if s.count("@") != 1:
        return None

    local, _, domain = s.rpartition("@")
    if domain != domain.lower():
        steps.append("lowercased domain part")
        s = f"{local}@{domain.lower()}"

    if not EMAIL_RE.match(s):
        return None
    return s, steps


class EmailExpert:
    """Normalize mechanically-repairable addresses in email columns.

    Column eligibility comes from ``SemanticColumnInfo.email_like`` — an
    explicit ``semantic_type="email"`` hint (e.g. compiled from
    "Emails must be valid") or an email-named column dominated by addresses.
    Values that cannot be repaired unambiguously are flagged for review.
    """

    name = "email"
    issue_type = "email_format"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return info.email_like

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        out: list[SemanticProposal] = []
        for raw, count in _value_counts(series).items():
            if not isinstance(raw, str) or not raw.strip():
                continue
            repaired = repair_email(raw)
            if repaired is not None:
                value, steps = repaired
                if value == raw:
                    continue  # already a clean address
                evidence: tuple[SemanticEvidence, ...] = (
                    SemanticEvidence("pattern", "; ".join(steps), 0.0),
                    SemanticEvidence(
                        "context_hint", f"{info.name!r} is an email column", 0.0
                    ),
                )
                base = 0.96 if any("@@" in s for s in steps) else 0.97
                out.append(
                    make_proposal(
                        column=info.name,
                        raw_value=raw,
                        proposed_value=value,
                        issue_type=self.issue_type,
                        expert=self.name,
                        base_confidence=base,
                        evidence=evidence,
                        count=int(count),
                        rationale=f"{raw!r} mechanically repairs to the valid address {value!r}",
                        info=info,
                    )
                )
            elif not EMAIL_RE.match(raw.strip()):
                # Invalid and not mechanically fixable: flag, never rewrite.
                evidence = (
                    SemanticEvidence(
                        "pattern", f"{raw!r} is not a valid email and has no "
                        "unambiguous mechanical repair", -0.2,
                    ),
                )
                out.append(
                    make_proposal(
                        column=info.name,
                        raw_value=raw,
                        proposed_value=None,
                        issue_type=self.issue_type,
                        expert=self.name,
                        base_confidence=0.60,
                        evidence=evidence,
                        count=int(count),
                        rationale=f"{raw!r} is not a valid email address; flagged for review",
                        info=info,
                        risk_override="high",
                    )
                )
        return out


# --------------------------------------------------------------------------- #
# Phone (region-aware; Phase 2 implements region="IN")
# --------------------------------------------------------------------------- #

_PHONE_STRIP_RE = re.compile(r"[\s\-().]")


def normalize_phone_in(raw: str) -> str | None:
    """Normalize an Indian mobile number to ``+91XXXXXXXXXX``, or ``None``.

    Only structurally unambiguous forms are accepted: a bare 10-digit mobile
    (leading 6–9), the same with a ``0`` trunk prefix, or with an explicit
    ``91`` / ``+91`` country code. Anything else (wrong length, ``00`` prefix,
    embedded letters) has no safe interpretation and returns ``None``.
    """
    s = _PHONE_STRIP_RE.sub("", raw.strip())
    has_plus = s.startswith("+")
    digits = s[1:] if has_plus else s
    if not digits.isdigit():
        return None
    if has_plus:
        if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
            return "+" + digits
        return None
    if len(digits) == 10 and digits[0] in "6789":
        return "+91" + digits
    if len(digits) == 11 and digits[0] == "0" and digits[1] in "6789":
        return "+91" + digits[1:]
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return "+" + digits
    return None


#: Normalizers by ISO region. Phase 2 ships India; the table is the seam for
#: adding more regions without touching the expert.
PHONE_NORMALIZERS = {"IN": normalize_phone_in}


class PhoneExpert:
    """Normalize phone numbers to a canonical E.164-like form, per region.

    Eligibility requires an *explicit* ``semantic_type="phone"`` hint (e.g.
    compiled from "Phone numbers are Indian") plus a supported region — phone
    columns are never guessed from data, and without a region there is no safe
    canonical form, so the expert stays silent.
    """

    name = "phone"
    issue_type = "phone_format"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return info.phone_like and (info.region or "") in PHONE_NORMALIZERS

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        region = info.region or ""
        normalize = PHONE_NORMALIZERS[region]
        out: list[SemanticProposal] = []
        for raw, count in _value_counts(series).items():
            if not isinstance(raw, str) or not raw.strip():
                continue
            value = normalize(raw)
            if value == raw:
                continue  # already canonical
            if value is not None:
                evidence = (
                    SemanticEvidence(
                        "pattern",
                        f"{raw!r} is a structurally unambiguous {region} mobile number",
                        0.0,
                    ),
                    SemanticEvidence("context_hint", f"region={region}", 0.0),
                )
                out.append(
                    make_proposal(
                        column=info.name,
                        raw_value=raw,
                        proposed_value=value,
                        issue_type=self.issue_type,
                        expert=self.name,
                        base_confidence=0.96,
                        evidence=evidence,
                        count=int(count),
                        rationale=f"{raw!r} normalizes to canonical {value!r} (region {region})",
                        info=info,
                    )
                )
            else:
                evidence = (
                    SemanticEvidence(
                        "pattern",
                        f"{raw!r} has no unambiguous {region} interpretation "
                        "(wrong digit count, invalid prefix, or non-digits)",
                        -0.2,
                    ),
                    SemanticEvidence("context_hint", f"region={region}", 0.0),
                )
                out.append(
                    make_proposal(
                        column=info.name,
                        raw_value=raw,
                        proposed_value=None,
                        issue_type=self.issue_type,
                        expert=self.name,
                        base_confidence=0.60,
                        evidence=evidence,
                        count=int(count),
                        rationale=f"{raw!r} is not a valid {region} number; flagged for review",
                        info=info,
                        risk_override="high",
                    )
                )
        return out


# --------------------------------------------------------------------------- #
# Reference lists (allowed values)
# --------------------------------------------------------------------------- #

_REF_NOISE_RE = re.compile(r"[\s\-_./,;:!]+")


def _ref_key(value: object) -> str:
    """Casefolded key with separator/punctuation noise removed."""
    return _REF_NOISE_RE.sub("", str(value).strip().casefold())


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Levenshtein distance, banded and capped at *cap* (returns ``cap`` if >=)."""
    if abs(len(a) - len(b)) >= cap:
        return cap
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        best = i
        for j, cb in enumerate(b, start=1):
            cost = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            )
            current.append(cost)
            best = min(best, cost)
        if best >= cap:
            return cap
        previous = current
    return min(previous[-1], cap)


class ReferenceExpert:
    """Repair values against an explicit allowed-values reference list.

    The list comes from ``semantic_context["columns"][col]["allowed_values"]``
    or a compiled ``ALLOWED_VALUES`` constraint. Exact members are untouched;
    values differing only by whitespace/case/separator noise repair with high
    confidence; a single-candidate fuzzy typo is *suggested*; anything with
    two close candidates is flagged as ambiguous. No clustering, no
    embeddings — only the user's own list is ever consulted.
    """

    name = "reference"
    issue_type = "reference_value"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return bool(info.allowed_values)

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        allowed = list(info.allowed_values)
        exact = set(allowed)
        by_key: dict[str, list[object]] = {}
        for a in allowed:
            by_key.setdefault(_ref_key(a), []).append(a)
        allowed_repr = [str(a) for a in allowed]
        list_evidence = SemanticEvidence(
            "context_hint", f"allowed values: {allowed_repr}", 0.0
        )

        out: list[SemanticProposal] = []
        for raw, count in _value_counts(series).items():
            if raw in exact or not isinstance(raw, str) or not raw.strip():
                continue
            proposal = self._propose_value(raw, int(count), by_key, list_evidence, info)
            if proposal is not None:
                out.append(proposal)
        return out

    def _propose_value(
        self,
        raw: str,
        count: int,
        by_key: dict[str, list[object]],
        list_evidence: SemanticEvidence,
        info: SemanticColumnInfo,
    ) -> SemanticProposal | None:
        key = _ref_key(raw)
        candidates = by_key.get(key)
        if candidates is not None:
            if len(candidates) == 1:
                target = candidates[0]
                if target == raw:
                    return None
                evidence = (
                    SemanticEvidence(
                        "pattern",
                        f"{raw!r} equals allowed value {target!r} up to "
                        "case/whitespace/separator noise",
                        0.0,
                    ),
                    list_evidence,
                )
                return make_proposal(
                    column=info.name,
                    raw_value=raw,
                    proposed_value=target,
                    issue_type=self.issue_type,
                    expert=self.name,
                    base_confidence=0.96,
                    evidence=evidence,
                    count=count,
                    rationale=f"{raw!r} normalizes exactly to allowed value {target!r}",
                    info=info,
                )
            return self._ambiguous(raw, count, candidates, list_evidence, info)

        # Fuzzy: capped edit distance between normalized keys.
        distances = sorted(
            ((_edit_distance(key, k), k) for k in by_key), key=lambda t: t[0]
        )
        if not distances:
            return None
        best_d, best_key = distances[0]
        second_d = distances[1][0] if len(distances) > 1 else 99
        tolerance = 1 if len(key) < 8 else 2
        if best_d <= tolerance and len(by_key[best_key]) == 1:
            if second_d <= best_d + 1:
                close = by_key[best_key] + by_key[distances[1][1]]
                return self._ambiguous(raw, count, close, list_evidence, info)
            target = by_key[best_key][0]
            evidence = (
                SemanticEvidence(
                    "pattern",
                    f"{raw!r} is within edit distance {best_d} of allowed value "
                    f"{target!r} and no other candidate is close",
                    0.0,
                ),
                list_evidence,
            )
            return make_proposal(
                column=info.name,
                raw_value=raw,
                proposed_value=target,
                issue_type=self.issue_type,
                expert=self.name,
                base_confidence=0.80,  # deliberately below auto thresholds
                evidence=evidence,
                count=count,
                rationale=f"{raw!r} looks like a typo of allowed value {target!r}; "
                          "suggested for review, not auto-applied",
                info=info,
            )
        # Unknown value: flag against the reference list, never rewrite.
        evidence = (
            SemanticEvidence(
                "pattern", f"{raw!r} matches no allowed value (closest distance "
                f"{best_d})", -0.2,
            ),
            list_evidence,
        )
        return make_proposal(
            column=info.name,
            raw_value=raw,
            proposed_value=None,
            issue_type=self.issue_type,
            expert=self.name,
            base_confidence=0.55,
            evidence=evidence,
            count=count,
            rationale=f"{raw!r} is not an allowed value; flagged for review",
            info=info,
            risk_override="high",
        )

    def _ambiguous(
        self,
        raw: str,
        count: int,
        candidates: list[object],
        list_evidence: SemanticEvidence,
        info: SemanticColumnInfo,
    ) -> SemanticProposal:
        shown = [str(c) for c in candidates]
        evidence = (
            SemanticEvidence(
                "pattern", f"{raw!r} is close to multiple allowed values {shown}", -0.2
            ),
            list_evidence,
        )
        return make_proposal(
            column=info.name,
            raw_value=raw,
            proposed_value=None,
            issue_type=self.issue_type,
            expert=self.name,
            base_confidence=0.60,
            evidence=evidence,
            count=count,
            rationale=f"{raw!r} is ambiguous between allowed values {shown}; "
                      "held for human review",
            info=info,
            risk_override="high",
        )


def _value_counts(series: pd.Series) -> pd.Series:
    try:
        return series.value_counts(dropna=True)
    except TypeError:  # unhashable payloads
        return pd.Series(dtype="int64")
