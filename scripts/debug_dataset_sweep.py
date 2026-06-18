"""Multi-dataset debug sweep for freshdata (H1–H5).

Run from repo root::

    python scripts/debug_dataset_sweep.py
    python scripts/debug_dataset_sweep.py --online --tier 1
    python scripts/debug_dataset_sweep.py --online
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_TESTS = _REPO / "tests"
_LOG_PATH = _REPO.parent / ".cursor" / "debug_dataset_sweep.log"

if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import freshdata as fd  # noqa: E402
from expectations import (  # noqa: E402
    ALL_FIXTURES,
    ALL_ONLINE_FIXTURES,
    ALL_ONLINE_TIER1,
    ALL_ONLINE_TIER2,
    load_expectations,
    load_fixture,
    load_online_fixture,
    resolve_column,
)
from freshdata.steps.columns import snake_case  # noqa: E402

ISOLATE = {"drop_empty_rows": False, "drop_duplicates": False, "verbose": False}
_ID_RE = re.compile(r"(?:^|_)(?:id|uuid|guid|key|record_id|transaction_id)s?$", re.I)


def _log(hypothesis_id: str, dataset: str, check: str, passed: bool, detail: str = "") -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hypothesisId": hypothesis_id,
        "location": "debug_dataset_sweep.py",
        "message": check,
        "data": {"dataset": dataset, "pass": passed, "detail": detail},
        "timestamp": int(time.time() * 1000),
    }
    with _LOG_PATH.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")


def _id_candidate(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        label = str(col)
        if _ID_RE.search(label) or label.lower() in {"id", "index", "pk"}:
            return label
    return None


def _preserve_candidates(name: str, df: pd.DataFrame) -> list[str]:
    exp = load_expectations(name).get("balanced", {})
    cols: list[str] = []
    for key in (
        "columns_never_imputed",
        "columns_never_capped",
        "columns_preserved_as_text",
        "columns_never_dropped",
    ):
        for col in exp.get(key, []):
            resolved = resolve_column(df, col)
            if resolved in df.columns and resolved not in cols:
                cols.append(resolved)
    return cols[:3]


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    try:
        pd.testing.assert_frame_equal(a, b)
        return True, ""
    except AssertionError as exc:
        return False, str(exc)[:300]


def check_h1(name: str, df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    base = {**ISOLATE, "strategy": "balanced"}

    id_col = _id_candidate(df)
    if id_col:
        ren = snake_case(id_col)
        kw = {**base, "id_columns": ren, "outlier_action": "cap"}
        a = fd.clean(df, **kw)
        b = fd.clean(df, **{**kw, "id_columns": (ren,)})
        ok, detail = _frames_equal(a, b)
        results.append((f"H1:id_columns:{ren}", ok, detail))

    for col in _preserve_candidates(name, df):
        ren = snake_case(col)
        kw = {**base, "preserve_columns": ren, "outlier_action": "cap"}
        a = fd.clean(df, **kw)
        b = fd.clean(df, **{**kw, "preserve_columns": (ren,)})
        ok, detail = _frames_equal(a, b)
        results.append((f"H1:preserve_columns:{ren}", ok, detail))

    if "duplicate" in name or id_col:
        subset = snake_case(id_col or str(df.columns[0]))
        kw = {**base, "duplicate_subset": subset, "drop_duplicates": True}
        a = fd.clean(df, **kw)
        b = fd.clean(df, **{**kw, "duplicate_subset": (subset,)})
        ok, detail = _frames_equal(a, b)
        results.append((f"H1:duplicate_subset:{subset}", ok, detail))

    kw = {**base, "extra_sentinels": "missing", "drop_empty_rows": False}
    a = fd.clean(df, **kw)
    b = fd.clean(df, **{**kw, "extra_sentinels": ("missing",)})
    ok, detail = _frames_equal(a, b)
    results.append(("H1:extra_sentinels", ok, detail))

    return results


def check_h2(name: str, df: pd.DataFrame) -> tuple[str, bool, str]:
    once = fd.clean(df, strategy="balanced", verbose=False)
    twice = fd.clean(once, strategy="balanced", verbose=False)
    ok, detail = _frames_equal(once, twice)
    return ("H2:idempotent", ok, detail)


def check_h3(name: str, df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    plan = fd.suggest_plan(df, **ISOLATE)
    _, report = fd.clean(df, return_report=True, **ISOLATE)
    for col, cp in plan.column_plans.items():
        if cp.outlier_action:
            oa = [a for a in report if a.step == "outliers" and a.column == col]
            if not oa:
                continue
            exp = cp.outlier_action
            act = oa[0].description
            ok = (exp == "flag" and "flagged" in act) or (exp == "cap" and "capped" in act) or (
                exp in act or exp in oa[0].rationale
            )
            results.append((f"H3:outlier:{col}", ok, f"plan={exp}, got={act[:80]}"))
        if cp.missing and cp.missing.model_id:
            ma = [a for a in report if a.step == "missing" and a.column == col]
            if not ma:
                continue
            action = ma[-1]
            ok = (
                cp.missing.model_id == action.model_id
                or cp.missing.model_id in action.description
                or cp.missing.model_id in action.rationale
            )
            results.append((f"H3:missing:{col}", ok, f"plan={cp.missing.model_id}, got={action.model_id}"))
    if not results:
        results.append(("H3:plan_clean", True, "no engine columns to compare"))
    return results


def check_h4(name: str, df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    id_col = _id_candidate(df)
    if not id_col:
        return [("H4:rename_id", True, "no id-like column")]
    raw_label = str(id_col)
    snake = snake_case(raw_label)
    if raw_label == snake:
        return [("H4:rename_id", True, "column already snake_case")]

    pre = fd.clean(df, id_columns=raw_label, outlier_action="cap", **ISOLATE)
    post = fd.clean(df, id_columns=snake, outlier_action="cap", **ISOLATE)
    ok, detail = _frames_equal(pre, post)
    results.append((f"H4:id_pre_vs_post_name:{raw_label}", ok, detail))
    return results


def check_h5(name: str, df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    exp = load_expectations(name).get("balanced", {})

    for col in exp.get("columns_never_imputed", []):
        raw = resolve_column(df, col)
        if raw not in df.columns:
            continue
        before_na = int(df[raw].isna().sum())
        out_s = fd.clean(df, preserve_columns=col, **ISOLATE)
        out_t = fd.clean(df, preserve_columns=(col,), **ISOLATE)
        ok_na = int(out_s[col].isna().sum()) == before_na == int(out_t[col].isna().sum())
        ok_eq, detail = _frames_equal(out_s, out_t)
        results.append((f"H5:never_imputed:{col}", ok_na and ok_eq, detail or f"na={before_na}"))

    for col in exp.get("columns_never_capped", []):
        raw = resolve_column(df, col)
        if raw not in df.columns:
            continue
        before_max = df[raw].max()
        out_s = fd.clean(df, preserve_columns=col, outlier_action="cap", **ISOLATE)
        out_t = fd.clean(df, preserve_columns=(col,), outlier_action="cap", **ISOLATE)
        ok_cap = out_s[col].max() == out_t[col].max() == before_max
        ok_eq, detail = _frames_equal(out_s, out_t)
        results.append((f"H5:never_capped:{col}", ok_cap and ok_eq, detail or f"max={before_max}"))

    if not results:
        results.append(("H5:expectations", True, "no preserve expectations"))
    return results


def sweep_dataset(name: str, df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    results.extend(check_h1(name, df))
    results.append(check_h2(name, df))
    results.extend(check_h3(name, df))
    results.extend(check_h4(name, df))
    results.extend(check_h5(name, df))
    for check, passed, detail in results:
        hid = check.split(":")[0]
        _log(hid, name, check, passed, detail)
    return results


def load_dataset(name: str, *, online: bool) -> pd.DataFrame:
    if online:
        return load_online_fixture(name, live=False)
    return load_fixture(name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-dataset freshdata debug sweep")
    parser.add_argument("--online", action="store_true", help="use cached online fixtures")
    parser.add_argument("--tier", type=int, default=None, help="online tier filter (1 or 2)")
    args = parser.parse_args()

    if args.online:
        if args.tier == 1:
            names = ALL_ONLINE_TIER1
        elif args.tier == 2:
            names = ALL_ONLINE_TIER2
        else:
            names = ALL_ONLINE_FIXTURES
    else:
        names = ALL_FIXTURES

    failures: list[str] = []
    print(f"{'dataset':<28} {'check':<40} {'result'}")
    print("-" * 80)

    for name in names:
        try:
            df = load_dataset(name, online=args.online)
        except Exception as exc:
            print(f"{name:<28} {'LOAD':<40} SKIP ({exc})")
            continue
        for check, passed, detail in sweep_dataset(name, df):
            status = "PASS" if passed else "FAIL"
            print(f"{name:<28} {check:<40} {status}")
            if not passed:
                failures.append(f"{name}/{check}: {detail}")
                if detail:
                    print(f"  -> {detail[:200]}")

    print("-" * 80)
    print(f"Total: {len(names)} datasets, {len(failures)} failure(s)")
    print(f"Log: {_LOG_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
