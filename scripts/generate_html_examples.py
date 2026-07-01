#!/usr/bin/env python
"""Generate the committed interactive-output HTML samples under docs/examples/.

Run from the repo root:

    python scripts/generate_html_examples.py

The samples are produced from small synthetic frames by the real renderers, so
they always match the current output. They need no optional viz dependencies.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import freshdata as fd

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "docs", "examples")


def _sample_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "email": [f"user{i}@example.com" if i % 9 else None for i in range(n)],
        "revenue": rng.normal(1000, 250, n).round(2),
        "signup_date": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
        "tier": rng.choice(["bronze", "silver", "gold", None], n),
    })
    df.loc[3:7, "revenue"] = [1e6, 9e5, 1.1e6, 8e5, 1e6]  # outliers
    return pd.concat([df, df.iloc[:5]], ignore_index=True)  # duplicates


def _write(name: str, html: str) -> None:
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"<!doctype html><meta charset='utf-8'><title>{name}</title>{html}")
    print(f"wrote {path}")


def main() -> int:
    df = _sample_frame()
    cleaned, report = fd.clean(df, return_report=True)

    _write("profile_cockpit.html", fd.profile(df).to_html())
    _write("action_timeline.html", report.to_html())
    _write("suggest_plan_cards.html", fd.suggest_plan(df).to_html())
    _write("explain_diff.html", fd.explain_clean(df).to_html())
    _write("compare_plans_grid.html", fd.compare_plans(df)._repr_html_())

    baseline = df.iloc[:150].copy()
    diff = fd.compare_to_baseline(df, baseline=baseline, key="customer_id")
    _write("baseline_drift.html", diff.to_html())

    _, gate = fd.evaluate_quality_debt(df, ledger=None)
    _write("quality_debt.html", gate.to_html())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
