#!/usr/bin/env python3
"""Download online dataset fixtures and write cached CSV slices for CI.

Usage::

    python scripts/fetch_online_fixtures.py
    python scripts/fetch_online_fixtures.py --refresh
    python scripts/fetch_online_fixtures.py --only titanic
    python scripts/fetch_online_fixtures.py --update-manifest
    python scripts/fetch_online_fixtures.py --discover --update-manifest
    python scripts/fetch_online_fixtures.py --refresh --max-failures 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_loader import load_dataframe, payload_bytes, registry_entry_to_manifest  # noqa: E402
from fixture_download import TransientFetchError, download  # noqa: E402

MANIFEST_PATH = ROOT / "tests" / "fixtures" / "online" / "manifest.json"
REGISTRY_PATH = ROOT / "tests" / "fixtures" / "online" / "registry.json"
CACHE_DIR = ROOT / "tests" / "fixtures" / "online" / "cache"
EXPECTATIONS_DIR = ROOT / "tests" / "fixtures" / "online"
DEFAULT_EXPECTATIONS = {
    "balanced": {"idempotent": True, "max_duration_seconds": 8},
    "aggressive": {"max_duration_seconds": 20},
}
TIER1_EXPECTATIONS = {
    "balanced": {"idempotent": True, "max_duration_seconds": 10},
    "aggressive": {"max_duration_seconds": 20},
}


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _download(url: str) -> bytes:
    return download(url)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _annotate(message: str) -> None:
    """Surface a failure as a GitHub Actions warning annotation when in CI."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::warning::{message}")


def fetch_one(
    name: str,
    entry: dict,
    manifest: dict,
    *,
    refresh: bool,
    update_manifest: bool,
) -> Path | None:
    url = entry["url"]
    max_rows = int(entry.get("max_rows", 2000))
    expected_hash = entry.get("sha256") or manifest.get(name, {}).get("sha256") or ""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{name}.csv"
    cache_exists = out_path.exists()

    if cache_exists and not refresh and not update_manifest:
        print(f"  {name}: cache exists ({out_path.name}), skipping download")
        return out_path

    print(f"  {name}: downloading {url}")
    raw = _download(url)
    digest = _sha256(raw)
    if expected_hash and digest != expected_hash and not refresh and not update_manifest:
        raise SystemExit(
            f"{name}: sha256 mismatch (expected {expected_hash[:12]}…, got {digest[:12]}…). "
            "Use --refresh to re-download or --update-manifest to pin new hash."
        )

    payload = payload_bytes(raw, entry)
    df = load_dataframe(payload, entry)
    if df.empty or df.shape[1] == 0:
        print(f"  {name}: SKIP — empty after parse", file=sys.stderr)
        return None
    df.columns = [str(c) for c in df.columns]
    df = df.head(max_rows)
    df.to_csv(out_path, index=False)
    print(f"  {name}: wrote {len(df)} rows x {df.shape[1]} cols -> {out_path.relative_to(ROOT)}")

    merged = registry_entry_to_manifest(entry)
    merged["sha256"] = digest
    manifest[name] = merged

    exp_path = EXPECTATIONS_DIR / f"{name}.expectations.json"
    if not exp_path.exists():
        tier = int(entry.get("tier", 2))
        stub = TIER1_EXPECTATIONS if tier == 1 else DEFAULT_EXPECTATIONS
        exp_path.write_text(json.dumps(stub, indent=2) + "\n")
        print(f"  {name}: created stub {exp_path.name}")

    return out_path


def _resolve_names(args: argparse.Namespace, registry: dict, manifest: dict) -> list[str]:
    if args.only:
        return args.only
    if args.discover:
        return sorted(registry.keys())
    return sorted(set(manifest.keys()) | set(registry.keys()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch online dataset fixtures")
    parser.add_argument("--refresh", action="store_true", help="Force re-download")
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Pin sha256 hashes in manifest",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Fetch all registry entries (sync manifest from registry)",
    )
    parser.add_argument("--only", action="append", default=[], metavar="ID")
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Tolerate up to N datasets whose download still fails with transient "
            "network errors after retries. Parse, empty, hash and permanent HTTP "
            "failures always fail the run."
        ),
    )
    args = parser.parse_args(argv)

    if not REGISTRY_PATH.exists():
        print(f"registry not found: {REGISTRY_PATH}", file=sys.stderr)
        return 1

    registry = _load_registry()
    manifest = _load_manifest()
    names = _resolve_names(args, registry, manifest)
    unknown = set(names) - set(registry)
    if unknown:
        print(f"unknown dataset id(s): {sorted(unknown)}", file=sys.stderr)
        return 1

    print(f"Fetching {len(names)} online fixture(s)...")
    dataset_failures: list[str] = []
    network_failures: list[str] = []
    for name in names:
        entry = dict(registry[name])
        if name in manifest and not args.discover:
            entry.setdefault("sha256", manifest[name].get("sha256", ""))
        try:
            result = fetch_one(
                name,
                entry,
                manifest,
                refresh=args.refresh or args.update_manifest or args.discover,
                update_manifest=args.update_manifest or args.discover,
            )
            if result is None:
                dataset_failures.append(name)
                manifest.pop(name, None)
        except TransientFetchError as exc:
            # Keep the existing manifest entry: a network blip says nothing
            # about whether the dataset itself is still valid.
            print(f"  {name}: FAILED (network) — {exc}", file=sys.stderr)
            network_failures.append(name)
        except (urllib.error.URLError, ValueError, pd.errors.ParserError) as exc:
            print(f"  {name}: FAILED — {exc}", file=sys.stderr)
            dataset_failures.append(name)
            if args.discover:
                manifest.pop(name, None)

    if args.update_manifest or args.refresh or args.discover:
        _save_manifest(manifest)
        print(f"Updated {MANIFEST_PATH.relative_to(ROOT)} ({len(manifest)} entries)")

    for name in network_failures:
        _annotate(f"online fixture {name!r}: download failed after retries (network)")
    for name in dataset_failures:
        _annotate(f"online fixture {name!r}: dataset failed to download or parse")
    print(
        f"Done. {len(dataset_failures)} dataset failure(s), "
        f"{len(network_failures)} network failure(s) (tolerating {args.max_failures})."
    )
    if dataset_failures or len(network_failures) > args.max_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
