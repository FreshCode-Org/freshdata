#!/usr/bin/env python3
"""Compare clean() across local and online fixtures for debugging.

Usage::

    python scripts/debug_datasets.py
    python scripts/debug_datasets.py --online titanic
    python scripts/debug_datasets.py --online --explain titanic
    python scripts/debug_datasets.py --infer-roles --online adult_income
    python scripts/debug_datasets.py --search missing
    python scripts/debug_datasets.py --refresh --online --compare
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

import freshdata as fd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from expectations import (  # noqa: E402
    ALL_FIXTURES,
    ALL_ONLINE_FIXTURES,
    load_fixture,
    load_online_fixture,
    load_registry,
)


def _load(name: str, *, online: bool) -> pd.DataFrame:
    if online:
        return load_online_fixture(name)
    return load_fixture(name)


def _search_datasets(tag: str) -> list[str]:
    registry = load_registry()
    return sorted(k for k, v in registry.items() if tag in v.get("tags", []))


def _print_compare(df: pd.DataFrame, label: str) -> None:
    table = fd.compare_clean(df)
    print(f"\n=== {label} ({len(df)} rows x {df.shape[1]} cols) ===")
    print(table.to_string(index=False))
    _, report = fd.clean(df, strategy="balanced", return_report=True, verbose=False)
    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")
    recs = getattr(report, "recommendations", None) or []
    if recs:
        print("\nRecommendations:")
        for r in recs:
            print(f"  - {r}")


def _print_explain(df: pd.DataFrame, label: str) -> None:
    explanation = fd.explain_clean(df, strategy="balanced")
    print(f"\n=== explain_clean: {label} ===")
    print(explanation.summary())
    if not explanation.roles.empty:
        print("\nRoles:")
        print(explanation.roles.to_string(index=False))


def _print_roles(df: pd.DataFrame, label: str) -> None:
    roles = fd.infer_roles(df, strategy="balanced")
    print(f"\n=== infer_roles: {label} ===")
    print(roles.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debug freshdata on fixture datasets")
    parser.add_argument("--online", action="store_true", help="Online cached datasets only")
    parser.add_argument("--local", action="store_true", help="Local fixtures only")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch online cache before compare",
    )
    parser.add_argument("--compare", action="store_true", help="Print compare_clean table")
    parser.add_argument("--explain", action="store_true", help="Print explain_clean summary")
    parser.add_argument("--infer-roles", action="store_true", help="Print infer_roles table")
    parser.add_argument("--search", metavar="TAG", help="Filter online datasets by registry tag")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("names", nargs="*", help="Dataset id(s); default all in scope")
    args = parser.parse_args(argv)

    if args.search and not args.names:
        args.names = _search_datasets(args.search)
        args.online = True

    if args.refresh:
        cmd = [sys.executable, str(ROOT / "scripts" / "fetch_online_fixtures.py"), "--refresh"]
        if args.names:
            for name in args.names:
                cmd.extend(["--only", name])
        subprocess.run(cmd, check=True, cwd=ROOT)

    scopes: list[tuple[str, bool]] = []
    if args.online and not args.local:
        scopes = [(n, True) for n in (args.names or ALL_ONLINE_FIXTURES)]
    elif args.local and not args.online:
        scopes = [(n, False) for n in (args.names or ALL_FIXTURES)]
    else:
        names = args.names
        if names:
            scopes = [(n, n in ALL_ONLINE_FIXTURES) for n in names]
        else:
            scopes = [(n, False) for n in ALL_FIXTURES] + [(n, True) for n in ALL_ONLINE_FIXTURES]

    show_compare = args.compare or not (args.explain or args.infer_roles)
    reports: list[dict] = []
    print(f"freshdata {fd.__version__} | comparing {len(scopes)} dataset(s)")
    for name, online in scopes:
        try:
            df = _load(name, online=online)
        except Exception as exc:  # noqa: BLE001 — CLI should continue on bad fixture
            print(f"\n=== {name}: SKIP ({exc}) ===")
            continue
        tag = "online" if online else "local"
        label = f"{name} [{tag}]"
        if show_compare:
            _print_compare(df, label)
        if args.explain:
            _print_explain(df, label)
        if args.infer_roles:
            _print_roles(df, label)
        if args.format == "json":
            explanation = fd.explain_clean(df, strategy="balanced")
            reports.append({"name": name, "online": online, **explanation.to_dict()})

    if args.format == "json" and reports:
        print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
