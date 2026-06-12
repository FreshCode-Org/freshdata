"""Load fixture expectations and assert optimal output criteria."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
from freshdata.steps.columns import snake_case

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ONLINE_DIR = FIXTURES_DIR / "online"
ONLINE_CACHE_DIR = ONLINE_DIR / "cache"
ONLINE_MANIFEST_PATH = ONLINE_DIR / "manifest.json"
ONLINE_REGISTRY_PATH = ONLINE_DIR / "registry.json"

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from dataset_loader import load_dataframe, payload_bytes  # noqa: E402

ALL_FIXTURES = [
    "aqi_sample",
    "sales_export",
    "survey_responses",
    "sensor_timeseries",
    "fraud_signals",
    "tiny_cohort",
    "wide_sparse",
    "large_panel",
    "duplicate_heavy",
    "locale_numbers",
    "mixed_roles",
]


def load_registry() -> dict:
    if not ONLINE_REGISTRY_PATH.exists():
        return {}
    return json.loads(ONLINE_REGISTRY_PATH.read_text())


def load_online_manifest() -> dict:
    if not ONLINE_MANIFEST_PATH.exists():
        return {}
    return json.loads(ONLINE_MANIFEST_PATH.read_text())


ALL_ONLINE_FIXTURES = sorted(load_online_manifest().keys())
_registry = load_registry()
ALL_ONLINE_TIER1 = sorted(k for k, v in _registry.items() if int(v.get("tier", 2)) == 1)
ALL_ONLINE_TIER2 = sorted(k for k, v in _registry.items() if int(v.get("tier", 2)) == 2)


def load_fixture(name: str) -> pd.DataFrame:
    path = FIXTURES_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"fixture {name} not found")
    return pd.read_csv(path)


def _fetch_online_live(name: str) -> pd.DataFrame:
    """Download a dataset slice from manifest URLs (network required)."""
    manifest = load_online_manifest()
    if name not in manifest:
        raise KeyError(f"unknown online fixture: {name}")
    entry = manifest[name]

    url = entry["url"]
    req = urllib.request.Request(url, headers={"User-Agent": "freshdata-fixture-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        raw = resp.read()
    expected = entry.get("sha256") or ""
    if expected:
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            pytest.fail(
                f"{name}: live sha256 mismatch (expected {expected[:12]}…, got {digest[:12]}…)"
            )
    payload = payload_bytes(raw, entry)
    df = load_dataframe(payload, entry)
    df.columns = [str(c) for c in df.columns]
    return df.head(int(entry.get("max_rows", 2000)))


def load_online_fixture(name: str, *, live: bool = False) -> pd.DataFrame:
    """Load cached online slice; optionally fetch live when *live* is True."""
    cache_path = ONLINE_CACHE_DIR / f"{name}.csv"
    if cache_path.exists() and not live:
        df = pd.read_csv(cache_path)
        df.columns = [str(c) for c in df.columns]
        return df
    if live:
        return _fetch_online_live(name)
    pytest.skip(f"online cache missing for {name!r}; run scripts/fetch_online_fixtures.py")


def load_expectations(name: str) -> dict:
    for base in (FIXTURES_DIR, ONLINE_DIR):
        path = base / f"{name}.expectations.json"
        if path.exists():
            return json.loads(path.read_text())
    tier = int(_registry.get(name, {}).get("tier", 2))
    if tier == 1:
        return {"balanced": {"idempotent": True, "max_duration_seconds": 10}}
    return {"balanced": {"idempotent": True, "max_duration_seconds": 15}}


def resolve_column(df: pd.DataFrame, col: str) -> str:
    """Map expectation column name to a label present in *df*."""
    if col in df.columns:
        return col
    for name in df.columns:
        if snake_case(str(name)) == col:
            return str(name)
    return col


def assert_expectations(
    name: str,
    strategy: str,
    df_before: pd.DataFrame,
    cleaned: pd.DataFrame,
    report: fd.CleanReport,
    *,
    duration: float,
) -> None:
    exp = load_expectations(name).get(strategy, {})
    if not exp:
        return
    for col in exp.get("columns_never_dropped", []):
        assert col in cleaned.columns, f"{col} was dropped under {strategy}"
    for col in exp.get("columns_never_imputed", []):
        raw = resolve_column(df_before, col)
        if raw in df_before.columns and col in cleaned.columns:
            assert cleaned[col].isna().sum() == df_before[raw].isna().sum(), (
                f"{col} was imputed under {strategy}"
            )
    for col in exp.get("columns_never_capped", []):
        raw = resolve_column(df_before, col)
        if raw in df_before.columns and col in cleaned.columns:
            before = df_before[raw].dropna()
            after = cleaned[col].dropna()
            if len(before) and len(after):
                assert after.max() >= before.max() or after.min() <= before.min(), (
                    f"{col} values were capped under {strategy}"
                )
    for col in exp.get("columns_preserved_as_text", []):
        raw = resolve_column(df_before, col)
        if col in cleaned.columns and raw in df_before.columns:
            assert cleaned[col].isna().sum() >= df_before[raw].isna().sum(), (
                f"{col} was force-filled under {strategy}"
            )
    for col, dtype in exp.get("required_conversions", {}).items():
        if col in cleaned.columns:
            assert str(cleaned[col].dtype).startswith(dtype.split("[")[0]), (
                f"{col} expected dtype {dtype}, got {cleaned[col].dtype}"
            )
    if exp.get("max_duration_seconds"):
        assert duration <= exp["max_duration_seconds"], (
            f"clean took {duration:.1f}s, limit {exp['max_duration_seconds']}s"
        )
    if exp.get("report_must_warn"):
        assert report.warnings, f"expected warnings for {name}/{strategy}"
    if exp.get("idempotent"):
        again = fd.clean(cleaned, strategy=strategy, verbose=False)
        pd.testing.assert_frame_equal(cleaned, again)
    if exp.get("missing_after") is not None:
        assert int(cleaned.isna().sum().sum()) == exp["missing_after"]
    if exp.get("no_sklearn_warnings"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            fd.clean(df_before, strategy=strategy, verbose=False)


def clean_with_timing(df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, fd.CleanReport, float]:
    started = time.perf_counter()
    out, report = fd.clean(df, return_report=True, verbose=False, **kwargs)
    return out, report, time.perf_counter() - started
