#!/usr/bin/env python3
"""Search and discover online dataset fixtures.

Usage::

    python scripts/search_datasets.py --tag missing
    python scripts/search_datasets.py --domain finance
    python scripts/search_datasets.py --format json
    python scripts/search_datasets.py --id titanic --show-manifest
    python scripts/search_datasets.py --discover --limit 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "tests" / "fixtures" / "online" / "registry.json"
MANIFEST_PATH = ROOT / "tests" / "fixtures" / "online" / "manifest.json"
CACHE_DIR = ROOT / "tests" / "fixtures" / "online" / "cache"


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _matches(
    entry: dict,
    *,
    tag: str | None,
    domain: str | None,
    fmt: str | None,
    tier: int | None,
) -> bool:
    if tag and tag not in entry.get("tags", []):
        return False
    if domain and entry.get("domain") != domain:
        return False
    if fmt and entry.get("format", "csv") != fmt:
        return False
    return tier is None or int(entry.get("tier", 2)) == tier

def _print_row(dataset_id: str, entry: dict, manifest: dict) -> None:
    cached = (CACHE_DIR / f"{dataset_id}.csv").exists()
    pinned = dataset_id in manifest
    sha = manifest.get(dataset_id, {}).get("sha256", "")[:12]
    tags = ",".join(entry.get("tags", []))
    print(
        f"{dataset_id:<22} tier={entry.get('tier', 2)} "
        f"fmt={entry.get('format', 'csv'):<5} "
        f"domain={entry.get('domain', '?'):<14} "
        f"cache={'yes' if cached else 'no':<3} "
        f"pinned={'yes' if pinned else 'no':<3} "
        f"sha256={sha + '…' if sha else '-':<14} "
        f"tags=[{tags}]"
    )
    print(f"  {entry.get('description', '')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search online dataset catalog")
    parser.add_argument("--tag", help="Filter by tag")
    parser.add_argument("--domain", help="Filter by domain")
    parser.add_argument("--format", dest="fmt", help="Filter by format (csv/json/zip)")
    parser.add_argument("--tier", type=int, help="Filter by tier (1 or 2)")
    parser.add_argument("--id", dest="dataset_id", help="Show one dataset")
    parser.add_argument("--show-manifest", action="store_true", help="Print manifest entry")
    parser.add_argument("--discover", action="store_true", help="Fetch unfetched registry ids")
    parser.add_argument("--limit", type=int, default=0, help="Max rows when listing")
    args = parser.parse_args(argv)

    registry = _load(REGISTRY_PATH)
    manifest = _load(MANIFEST_PATH)

    if args.discover:
        missing = [
            k for k in registry
            if k not in manifest or not (CACHE_DIR / f"{k}.csv").exists()
        ]
        if args.limit:
            missing = missing[: args.limit]
        if not missing:
            print("All registry datasets are cached and pinned.")
            return 0
        fetch = ROOT / "scripts" / "fetch_online_fixtures.py"
        cmd = [sys.executable, str(fetch), "--discover", "--update-manifest"]
        for name in missing:
            cmd.extend(["--only", name])
        return subprocess.call(cmd, cwd=ROOT)

    if args.dataset_id:
        if args.dataset_id not in registry:
            print(f"unknown id: {args.dataset_id}", file=sys.stderr)
            return 1
        _print_row(args.dataset_id, registry[args.dataset_id], manifest)
        if args.show_manifest and args.dataset_id in manifest:
            print(json.dumps(manifest[args.dataset_id], indent=2))
        return 0

    count = 0
    for dataset_id in sorted(registry):
        entry = registry[dataset_id]
        if not _matches(entry, tag=args.tag, domain=args.domain, fmt=args.fmt, tier=args.tier):
            continue
        _print_row(dataset_id, entry, manifest)
        count += 1
        if args.limit and count >= args.limit:
            break
    print(f"\n{count} dataset(s) matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
