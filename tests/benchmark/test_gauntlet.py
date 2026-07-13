"""Smoke tests for the Validation Gauntlet harness itself."""

from __future__ import annotations

import pandas as pd
from benchmarks.gauntlet import build_fixture, compute_metrics, run_fixture
from benchmarks.gauntlet.fixtures import FIXTURES
from benchmarks.gauntlet.report import check_gates, render_markdown, results_payload

SMOKE_ROWS = 80


def test_fixtures_are_deterministic():
    for name in FIXTURES:
        a = build_fixture(name, SMOKE_ROWS, seed=7)
        b = build_fixture(name, SMOKE_ROWS, seed=7)
        pd.testing.assert_frame_equal(a.df, b.df)
        assert a.cells == b.cells


def test_fixture_labels_are_well_formed():
    for name in FIXTURES:
        fx = build_fixture(name, SMOKE_ROWS)
        seen = set()
        for c in fx.cells:
            assert c.expect in ("preserve", "repair", "flag", "review")
            key = (c.row, c.column)
            assert key not in seen, f"{name}: duplicate label at {key}"
            seen.add(key)
            assert fx.df.columns.get_loc(c.column) >= 0
        assert fx.dup_row_count > 0
        assert len(fx.df) == fx.n_rows + fx.dup_row_count
        # pristine() restores every labelled cell
        pristine = fx.pristine()
        for c in fx.cells:
            got = pristine.iloc[c.row, pristine.columns.get_loc(c.column)]
            assert (got == c.replaced) or (pd.isna(got) and pd.isna(c.replaced))


def test_runner_and_gates_on_two_fixtures():
    metrics = {}
    for name in ("finance", "text"):
        run = run_fixture(build_fixture(name, SMOKE_ROWS))
        m = compute_metrics(run)
        metrics[name] = m
        assert m["corruption_count"] == 0
        assert m["preservation_rate"] in (None, 1.0)
        assert m["deterministic"]
        assert m["audit_completeness"] == 1.0
        assert m["detection"]["f1"] >= 0.85

    payload = results_payload(metrics, n_rows=SMOKE_ROWS, seed=42)
    assert check_gates(payload, baseline=None) == []
    md = render_markdown(payload)
    assert "| finance |" in md and "| text |" in md
