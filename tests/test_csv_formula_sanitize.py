"""CSV formula-injection (OWASP) sanitization.

Cells starting with ``= + - @ <tab> <cr>`` execute as formulas when a CSV is
opened in Excel / Google Sheets / LibreOffice. Review-queue exports exist to
be opened by humans in spreadsheets, so they sanitize by default (opt-out);
data-pipeline outputs (``fd.clean_csv``, streaming CLI) preserve byte-exact
fidelity by default and offer an explicit opt-in.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata._util import sanitize_csv_formulas
from freshdata.enterprise.cli import main as cli_main
from freshdata.enterprise.entity_resolution import (
    ReviewItem,
    ReviewQueueConfig,
    ReviewQueueReport,
    export_review_queue,
)

PAYLOADS = [
    "=cmd|' /C calc'!A0",
    "+SUM(1,2)",
    "-2+3+cmd",
    "@SUM(A1:A9)",
    "\t=1+1",
    "\r=1+1",
]


# --------------------------------------------------------------------------- #
# Sanitizer unit behavior
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("payload", PAYLOADS)
def test_sanitizer_quotes_every_owasp_trigger(payload):
    df = pd.DataFrame({"note": [payload]})
    out = sanitize_csv_formulas(df)
    assert out.loc[0, "note"] == "'" + payload


def test_sanitizer_leaves_safe_values_and_dtypes_alone():
    df = pd.DataFrame(
        {
            "name": ["Ada", "Grace"],
            "amount": [-12.5, 3.0],  # numeric minus is NOT a formula risk
            "count": [1, 2],
            "note": ["hello = world", "x"],  # trigger not at cell start
        }
    )
    out = sanitize_csv_formulas(df)
    pd.testing.assert_frame_equal(out, df)


def test_sanitizer_covers_column_labels_and_categoricals():
    df = pd.DataFrame({"=evil()": ["a"], "plan": pd.Categorical(["=SUM(A1)"])})
    out = sanitize_csv_formulas(df)
    assert list(out.columns) == ["'=evil()", "plan"]
    assert out["plan"].iloc[0] == "'=SUM(A1)"


def test_sanitizer_does_not_mutate_input():
    df = pd.DataFrame({"note": ["=1+1"]})
    sanitize_csv_formulas(df)
    assert df.loc[0, "note"] == "=1+1"


# --------------------------------------------------------------------------- #
# Review-queue export: sanitizes by default (spreadsheet-bound artifact)
# --------------------------------------------------------------------------- #


def _poisoned_queue() -> ReviewQueueReport:
    item = ReviewItem(
        item_id="item_000",
        left_id="=cmd|' /C calc'!A0",
        right_id=7,
        score=0.6,
        match_weight=1.2,
        comparison_vector={"name": 0.9},
        blocking_rule_ids=("block_000",),
        explanation="=HYPERLINK(\"http://evil\",\"click\") score 0.6",
        created_at="2026-07-11T00:00:00Z",
    )
    return ReviewQueueReport(
        items=(item,),
        created_at="2026-07-11T00:00:00Z",
        config=ReviewQueueConfig(),
    )


def test_export_review_queue_sanitizes_csv_by_default(tmp_path):
    path = tmp_path / "queue.csv"
    export_review_queue(_poisoned_queue(), path, format="csv")
    text = path.read_text()
    assert "'=cmd|" in text
    assert "'=HYPERLINK" in text
    assert "\n=cmd" not in text and ",=" not in text


def test_export_review_queue_sanitize_opt_out(tmp_path):
    path = tmp_path / "queue.csv"
    export_review_queue(_poisoned_queue(), path, format="csv", sanitize_formulas=False)
    text = path.read_text()
    assert "=cmd|" in text and "'=cmd|" not in text


def test_export_review_queue_jsonl_untouched(tmp_path):
    # jsonl is not a spreadsheet format; values must stay byte-exact
    path = tmp_path / "queue.jsonl"
    export_review_queue(_poisoned_queue(), path, format="jsonl")
    assert "'=cmd" not in path.read_text()


# --------------------------------------------------------------------------- #
# fd.clean_csv: fidelity by default, explicit opt-in
# --------------------------------------------------------------------------- #


def _write_input(tmp_path) -> str:
    src = tmp_path / "in.csv"
    pd.DataFrame(
        {"name": ["Ada", "Grace", "Alan"], "note": ["=1+1", "safe", "@SUM(A1)"]}
    ).to_csv(src, index=False)
    return str(src)


def test_clean_csv_preserves_fidelity_by_default(tmp_path):
    out = tmp_path / "out.csv"
    fd.clean_csv(_write_input(tmp_path), output_path=out)
    text = out.read_text()
    assert "=1+1" in text and "'=1+1" not in text


def test_clean_csv_sanitize_formulas_opt_in(tmp_path):
    out = tmp_path / "out.csv"
    result = fd.clean_csv(_write_input(tmp_path), output_path=out, sanitize_formulas=True)
    text = out.read_text()
    assert "'=1+1" in text and "'@SUM(A1)" in text
    # the returned frame is NOT sanitized — only the written artifact is
    assert (result["note"] == "=1+1").any()


# --------------------------------------------------------------------------- #
# Streaming CLI: --sanitize-formulas opt-in
# --------------------------------------------------------------------------- #


def test_stream_cli_sanitize_formulas_flag(tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"a": [1, 2], "note": ["=1+1", "safe"]}).to_csv(src, index=False)

    raw_out = tmp_path / "raw.csv"
    rc = cli_main(["stream", str(src), "-o", str(raw_out), "--quiet"])
    assert rc == 0
    assert "'=1+1" not in raw_out.read_text()  # default: fidelity

    san_out = tmp_path / "san.csv"
    rc = cli_main(
        ["stream", str(src), "-o", str(san_out), "--quiet", "--sanitize-formulas"]
    )
    assert rc == 0
    assert "'=1+1" in san_out.read_text()
