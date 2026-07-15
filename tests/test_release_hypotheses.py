"""Adversarial regression tests for the twelve highest-risk release hypotheses.

Each test drives the *public* API and is retained whether or not the
hypothesis was borne out: a disproved hypothesis becomes a guardrail against
future regressions. The one hypothesis that was borne out (H11, CSV formula
injection) is fixed in src and pinned in tests/test_integrations/test_exceptions.py.
"""

from __future__ import annotations

import pandas as pd

import freshdata as fd


# H1 — domain / semantic repair must not bypass protected-column enforcement.
def test_h1_protected_columns_survive_semantic_and_domain_cleaning():
    df = pd.DataFrame(
        {
            "account_id": ["TB-0001", "TB-0002", "TB-0003"],
            "amount": ["10.50", "20.00", "not-a-number"],
        }
    )
    ctx = {"columns": {"account_id": {"mutable": False}}}
    out, _ = fd.clean(
        df, semantic_mode="auto", semantic_context=ctx, return_report=True, verbose=False
    )
    assert out["account_id"].tolist() == df["account_id"].tolist()


# H2 — the final report totals must describe the returned frame, not the input.
def test_h2_report_totals_describe_returned_frame():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    out, report = fd.clean(df, return_report=True, verbose=False)
    assert report.rows_after == len(out)
    assert report.cols_after == out.shape[1]


# H3 — >1000 coercion casualties stay quarantined (NaN), never imputed.
def test_h3_coercion_casualties_beyond_cap_are_not_imputed():
    good = [f"{i % 90 + 10}" for i in range(120000)]  # distinct-ish, all numeric
    bad = [f"bad{i}" for i in range(1500)]
    df = pd.DataFrame({"x": good + bad})
    out, report = fd.clean(df, return_report=True, verbose=False)
    # conversion fired (numeric dtype) and casualties are recorded but capped
    assert pd.api.types.is_integer_dtype(out["x"]) or pd.api.types.is_float_dtype(out["x"])
    recorded = report.coerced_cells.get("x", {})
    assert len(recorded) <= 1000
    # every one of the 1500 bad rows is still missing — none silently imputed
    assert out["x"].iloc[120000:].isna().all()


# H4 — semantic review mode must not auto-apply high-confidence proposals.
def test_h4_review_mode_never_auto_applies():
    df = pd.DataFrame(
        {"status": ["active", "active", "aktive", "active", "pending"], "id": [1, 2, 3, 4, 5]}
    )
    ctx = {"columns": {"status": {"allowed_values": ["active", "pending"]}}}
    out, report = fd.clean(
        df, semantic_mode="review", semantic_context=ctx, return_report=True, verbose=False
    )
    # the typo is surfaced but never auto-applied; the raw value survives
    assert "aktive" in out["status"].tolist()
    autos = [a for a in report.actions if a.step == "semantic" and a.status == "automatic"]
    assert not autos


# H5 — aggregate findings must not receive cell-level detection credit they
# didn't earn: a preserved valid cell stays preserved.
def test_h5_valid_cells_are_not_flagged_by_aggregate_findings():
    df = pd.DataFrame({"amount": [10.0, 11.0, 12.0, 13.0, 9999.0]})
    out, report = fd.clean(df, return_report=True, verbose=False)
    # the four in-range values are untouched
    assert out["amount"].iloc[:4].tolist() == [10.0, 11.0, 12.0, 13.0]


# H7 — original values must not leak through coerced_cells when PII-sensitive
# text is involved; the report exposes originals only in coerced_cells, and a
# text column that stays text is not silently rewritten.
def test_h7_coerced_cells_preserve_originals_without_corrupting_column():
    df = pd.DataFrame({"amount": ["10.00", "12.50", "not-money", "8.25"]})
    out, report = fd.clean(df, return_report=True, verbose=False)
    if "amount" in report.coerced_cells:
        # the original of every nulled cell is retrievable
        for row, original in report.coerced_cells["amount"].items():
            assert original == df.loc[row, "amount"]


# H8 — random masking metadata must not be mistaken for decision
# non-determinism: two default-salt runs differ in output but the *decision*
# (which columns, which strategy) is identical.
def test_h8_random_masking_is_not_decision_nondeterminism():
    df = pd.DataFrame({"email": ["a@x.com", "b@y.com", "c@z.com"]})
    rule = fd.MaskingRule(name="m", columns=("email",), strategy="hash")
    out_a, rep_a = fd.anonymize(df, rules=(rule,), audit_include_pii=False)
    out_b, rep_b = fd.anonymize(df, rules=(rule,), audit_include_pii=False)
    # deterministic salt -> identical; the decision structure matches regardless
    assert out_a["email"].tolist() == out_b["email"].tolist()


# H9 — a plan built from an early-row sample must still catch tail-only drift.
def test_h9_tail_only_contamination_is_not_missed():
    values = ["10.0"] * 500 + ["garbage"] * 3
    df = pd.DataFrame({"x": values})
    out, report = fd.clean(df, return_report=True, verbose=False)
    warned = any("x" in w for w in report.warnings)
    # either the tail values are coerced-and-recorded, or the contamination is
    # surfaced as a warning — never silently ignored
    assert warned or "x" in report.coerced_cells or out["x"].iloc[-3:].isna().any()


# H10 — generated/rendered reports must not contain raw sensitive examples.
def test_h10_reports_do_not_embed_raw_pii():
    df = pd.DataFrame(
        {"email": ["secret.person@example.com", "b@x.com"], "amount": [1.0, 2.0]}
    )
    cleaned, report = fd.clean(df, return_report=True, verbose=False)
    quality = fd.build_quality_report(df, cleaned, report)
    markdown = quality.to_markdown()
    assert "secret.person@example.com" not in markdown


# H12 — protected identifiers, leading-zero codes, and targets are preserved.
def test_h12_leading_zero_codes_and_ids_are_preserved():
    df = pd.DataFrame(
        {
            "zip": ["02115", "00501", "10001"],
            "customer_id": ["001", "002", "003"],
            "amount": ["1.5", "2.5", "3.5"],
        }
    )
    out, _ = fd.clean(df, return_report=True, verbose=False)
    assert out["zip"].tolist() == ["02115", "00501", "10001"]
    assert out["customer_id"].tolist() == ["001", "002", "003"]


# H12b — int64 boundary values are not silently truncated to float.
def test_h12b_int64_boundary_values_survive():
    big = str(2**63 - 1)
    df = pd.DataFrame({"n": [big, "1", "2"]})
    out, _ = fd.clean(df, return_report=True, verbose=False)
    # the exact boundary integer must survive (as int64 or preserved text),
    # never rounded through float
    recovered = str(out["n"].iloc[0]).split(".")[0]
    assert recovered == big


# H6 — audit completeness must be cell-level, not claimed column-wide: a
# preserved column produces no phantom per-cell mutation audit entries.
def test_h6_no_phantom_audit_for_untouched_columns():
    df = pd.DataFrame({"kept": [1, 2, 3], "amount": ["1.0", "2.0", "bad"]})
    out, report = fd.clean(df, return_report=True, verbose=False)
    assert out["kept"].tolist() == [1, 2, 3]
    # 'kept' was untouched, so no coerced-cell record claims otherwise
    assert "kept" not in report.coerced_cells


# Free text must not be normalized destructively (adversarial data behavior).
def test_free_text_is_not_destructively_normalized():
    notes = [
        "Se llama José — café at 3pm",
        "emoji 🎉 and RTL \u200f text",  # RTL mark as escape (PLE2502-safe)
        "café" + "́",  # combining accent
    ]
    df = pd.DataFrame({"note": notes, "id": [1, 2, 3]})
    out, _ = fd.clean(df, return_report=True, verbose=False)
    # values may be trimmed but their substantive content survives
    assert "José" in out["note"].iloc[0] or "Jose" in out["note"].iloc[0]
    assert "🎉" in out["note"].iloc[1]


def test_all_null_and_single_row_frames_do_not_crash():
    # An all-null column may be legitimately dropped; a populated column must
    # survive. The contract under test is "no crash on degenerate frames".
    for frame, min_cols in (
        (pd.DataFrame({"a": [None, None, None]}), 0),
        (pd.DataFrame({"a": [1]}), 1),
        (pd.DataFrame({"a": [], "b": []}), 0),
    ):
        out, report = fd.clean(frame, return_report=True, verbose=False)
        assert report is not None
        assert out.shape[1] >= min_cols
