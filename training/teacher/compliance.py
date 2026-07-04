"""Provider-terms compliance ledger and pre-run gate.

Before any teacher run, a human records a compliance check for the provider
in ``training/teacher/compliance_ledger.json``. :func:`require_approved`
blocks the run when the record is missing, stale, unapproved, reviewer-less,
or when the provider's terms disallow using outputs for model training.

CLI::

    python -m training.teacher.compliance check [--provider X --use labeling]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common import read_json, utc_now_iso, write_json

LEDGER_PATH = Path(__file__).resolve().parent / "compliance_ledger.json"
#: A terms check older than this is stale and must be re-verified.
MAX_AGE_DAYS = 180

ALLOWED_USES = (
    "realism_direction",
    "column_role_labeling",
    "context_paraphrase",
    "ambiguity_adjudication",
    "rationale_templates",
    "red_teaming",
)


class ComplianceError(RuntimeError):
    """The teacher run is blocked by the compliance gate."""


def load_ledger(path: Path | str = LEDGER_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {"ledger_version": "1", "providers": []}
    return read_json(path)


def _entry_for(ledger: dict[str, Any], provider: str) -> dict[str, Any] | None:
    for entry in ledger.get("providers", []):
        if entry.get("provider") == provider:
            return entry
    return None


def check_entry(entry: dict[str, Any] | None, provider: str, intended_use: str) -> list[str]:
    """Return blocking problems for one provider/use pair (empty = approved)."""
    problems: list[str] = []
    if entry is None:
        problems.append(f"no terms snapshot recorded for provider {provider!r}")
        return problems
    if not entry.get("terms_url") or not entry.get("terms_snapshot_id"):
        problems.append(f"{provider}: no terms snapshot exists")
    checked = str(entry.get("date_checked", ""))
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(checked)
        if age.days > MAX_AGE_DAYS:
            problems.append(f"{provider}: terms snapshot is stale ({age.days} days old)")
    except ValueError:
        problems.append(f"{provider}: date_checked missing or unparseable")
    if not entry.get("reviewer"):
        problems.append(f"{provider}: reviewer is missing")
    if entry.get("status") != "approved":
        problems.append(f"{provider}: compliance status is {entry.get('status')!r}, not approved")
    if entry.get("training_use_allowed") is not True:
        problems.append(f"{provider}: provider terms do not allow training use of outputs")
    allowed = entry.get("allowed_uses", [])
    if intended_use not in allowed:
        problems.append(
            f"{provider}: intended use {intended_use!r} is not approved (approved: {allowed})"
        )
    return problems


def require_approved(
    provider: str, intended_use: str, *, ledger_path: Path | str = LEDGER_PATH
) -> dict[str, Any]:
    """Return the approved ledger entry or raise :class:`ComplianceError`."""
    if intended_use not in ALLOWED_USES:
        raise ComplianceError(
            f"intended use {intended_use!r} is not a permitted teacher use; "
            f"allowed: {ALLOWED_USES}"
        )
    entry = _entry_for(load_ledger(ledger_path), provider)
    problems = check_entry(entry, provider, intended_use)
    if problems:
        raise ComplianceError("; ".join(problems))
    assert entry is not None
    return entry


def record_check(
    *,
    provider: str,
    terms_url: str,
    terms_snapshot_id: str,
    reviewer: str,
    allowed_uses: list[str],
    training_use_allowed: bool,
    redistribution_allowed: bool,
    status: str = "approved",
    notes: str = "",
    ledger_path: Path | str = LEDGER_PATH,
) -> dict[str, Any]:
    """Append or replace a provider compliance record (manual review result)."""
    ledger = load_ledger(ledger_path)
    entry = {
        "provider": provider,
        "terms_url": terms_url,
        "terms_snapshot_id": terms_snapshot_id,
        "date_checked": utc_now_iso(),
        "reviewer": reviewer,
        "allowed_uses": allowed_uses,
        "training_use_allowed": training_use_allowed,
        "redistribution_allowed": redistribution_allowed,
        "status": status,
        "notes": notes,
    }
    ledger["providers"] = [e for e in ledger.get("providers", []) if e.get("provider") != provider]
    ledger["providers"].append(entry)
    write_json(ledger_path, ledger)
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.teacher.compliance")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="verify compliance for a provider/use")
    check.add_argument("--provider", default=None, help="provider name (default: all recorded)")
    check.add_argument("--use", default="column_role_labeling", choices=ALLOWED_USES)
    check.add_argument("--ledger", default=str(LEDGER_PATH))
    args = parser.parse_args(argv)

    ledger = load_ledger(args.ledger)
    providers = ([args.provider] if args.provider
                 else [e.get("provider", "") for e in ledger.get("providers", [])])
    if not providers:
        print("FAIL: compliance ledger is empty — record a terms check first", file=sys.stderr)
        return 1
    failed = False
    for provider in providers:
        problems = check_entry(_entry_for(ledger, provider), provider, args.use)
        if problems:
            failed = True
            for problem in problems:
                print(f"FAIL: {problem}", file=sys.stderr)
        else:
            print(f"OK: {provider} approved for {args.use}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
