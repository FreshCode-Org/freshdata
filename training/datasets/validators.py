"""Validators for the seed registry and corruption labels.

``python -m training.datasets.validators --check-licenses`` is the release
gate for training-data provenance: it fails when any source has a missing
license, missing attribution, non-explicit training permission, unresolved
PII risk, or unclear commercial-use status (including share-alike licenses
without a recorded legal review).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..common import TRAINING_ROOT, read_json

REGISTRY_PATH = TRAINING_ROOT / "seed" / "registry.json"

#: Licenses whose commercial use is unambiguous for training.
ALLOWED_LICENSES = frozenset({
    "CC0-1.0", "PDDL-1.0", "CC-BY-4.0", "Apache-2.0", "MIT",
    "BSD-3-Clause", "OGL-UK-3.0", "OGL-India",
})
#: Share-alike / reciprocal licenses: blocked unless legal review approves.
SHARE_ALIKE_LICENSES = frozenset({
    "CC-BY-SA-4.0", "CC-BY-SA-3.0", "ODbL-1.0", "GPL-3.0", "GPL-2.0",
})
PII_RISKS = frozenset({"none", "synthetic", "review_required"})

_REQUIRED_FIELDS = (
    "source_id", "name", "url", "license", "license_text_path",
    "attribution", "allowed_for_training", "pii_risk",
)


def validate_source(entry: dict[str, Any], seed_dir: Path) -> list[str]:
    """Return a list of human-readable problems for one registry entry."""
    errors: list[str] = []
    sid = str(entry.get("source_id") or "<missing source_id>")

    for field in _REQUIRED_FIELDS:
        if field not in entry or entry[field] in (None, ""):
            errors.append(f"{sid}: missing required field {field!r}")

    license_id = str(entry.get("license") or "")
    if license_id:
        review = entry.get("legal_review") or {}
        if license_id in SHARE_ALIKE_LICENSES:
            approved = (
                isinstance(review, dict)
                and review.get("approved") is True
                and review.get("reviewer")
            )
            if not approved:
                errors.append(
                    f"{sid}: share-alike license {license_id!r} requires an approving "
                    "legal_review record (approved: true, reviewer set)"
                )
        elif license_id not in ALLOWED_LICENSES:
            errors.append(
                f"{sid}: license {license_id!r} has unclear commercial-use status; "
                f"allowed: {', '.join(sorted(ALLOWED_LICENSES))} "
                "(or share-alike with legal review)"
            )

    if "allowed_for_training" in entry and not isinstance(entry["allowed_for_training"], bool):
        errors.append(f"{sid}: allowed_for_training must be an explicit boolean")
    elif entry.get("allowed_for_training") is False:
        errors.append(f"{sid}: allowed_for_training is false — remove the source or approve it")

    pii = entry.get("pii_risk")
    if pii is not None and pii not in PII_RISKS:
        errors.append(f"{sid}: pii_risk must be one of {sorted(PII_RISKS)}, got {pii!r}")
    if pii == "review_required":
        review = entry.get("legal_review") or {}
        if not (isinstance(review, dict) and review.get("approved") is True):
            errors.append(
                f"{sid}: unresolved PII risk (review_required without approving legal_review)")

    text_path = entry.get("license_text_path")
    if text_path and not (seed_dir / str(text_path)).is_file():
        errors.append(
            f"{sid}: license_text_path {text_path!r} does not exist under training/seed/")
    return errors


def check_licenses(registry_path: Path | str = REGISTRY_PATH) -> list[str]:
    """Validate the whole registry; return all problems (empty = pass)."""
    registry_path = Path(registry_path)
    if not registry_path.is_file():
        return [f"registry not found: {registry_path}"]
    data = read_json(registry_path)
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        return ["registry must contain a non-empty 'sources' list"]
    errors: list[str] = []
    seen: set[str] = set()
    for entry in sources:
        if not isinstance(entry, dict):
            errors.append("registry entries must be objects")
            continue
        sid = str(entry.get("source_id") or "")
        if sid in seen:
            errors.append(f"{sid}: duplicate source_id")
        seen.add(sid)
        errors.extend(validate_source(entry, registry_path.parent))
    return errors


# --------------------------------------------------------------------------- #
# Corruption-label validation
# --------------------------------------------------------------------------- #

LABEL_FIELDS = (
    "raw_value", "clean_value", "column", "transform_family", "params",
    "should_repair", "should_auto_apply", "risk", "protected", "ambiguous",
)
RISKS = frozenset({"low", "medium", "high"})


def validate_label(label: dict[str, Any]) -> list[str]:
    """Check one corruption label for internal consistency."""
    errors: list[str] = []
    for field in LABEL_FIELDS:
        if field not in label:
            errors.append(f"label missing field {field!r}")
    risk = label.get("risk")
    if risk not in RISKS:
        errors.append(f"invalid risk {risk!r}")
    if not isinstance(label.get("params", {}), dict):
        errors.append("params must be a dict")
    if label.get("ambiguous") and label.get("should_auto_apply"):
        errors.append("ambiguous corruption must not be auto-apply")
    if label.get("protected") and label.get("should_auto_apply"):
        errors.append("protected-column corruption must never be auto-apply")
    if label.get("should_auto_apply") and not label.get("should_repair"):
        errors.append("should_auto_apply requires should_repair")
    if label.get("should_repair") and label.get("raw_value") == label.get("clean_value"):
        errors.append("should_repair label with identical raw/clean value")
    return errors


def validate_labels(labels: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for i, label in enumerate(labels):
        for problem in validate_label(label):
            errors.append(f"label[{i}]: {problem}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.datasets.validators")
    parser.add_argument("--check-licenses", action="store_true", help="validate the seed registry")
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    parser.add_argument("--labels", default=None, help="optional labels JSONL to validate")
    args = parser.parse_args(argv)

    failures: list[str] = []
    if args.check_licenses:
        failures.extend(check_licenses(args.registry))
    if args.labels:
        from ..common import read_jsonl  # noqa: PLC0415

        failures.extend(validate_labels(read_jsonl(args.labels)))
    if not args.check_licenses and not args.labels:
        parser.error("nothing to do: pass --check-licenses and/or --labels")

    if failures:
        for problem in failures:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print("seed registry / labels: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
