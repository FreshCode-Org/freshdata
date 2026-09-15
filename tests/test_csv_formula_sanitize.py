"""CSV formula-injection (OWASP) sanitization.

Cells starting with ``= + - @ <tab> <cr>`` execute as formulas when a CSV is
opened in Excel / Google Sheets / LibreOffice. Since the production audit
(P1-5) every CSV sink that writes user cell data sanitizes by default —
``fd.clean_csv``, the streaming CLI, the enterprise CLI, review-queue
exports — with ``sanitize_formulas=False`` / ``--no-sanitize-formulas`` as
the explicit opt-out for byte-exact round-trips.
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
# fd.clean_csv: sanitizes by default (audit P1-5), explicit opt-out
# --------------------------------------------------------------------------- #


def _write_input(tmp_path) -> str:
    src = tmp_path / "in.csv"
    pd.DataFrame(
        {"name": ["Ada", "Grace", "Alan"], "note": ["=1+1", "safe", "@SUM(A1)"]}
    ).to_csv(src, index=False)
    return str(src)


def test_clean_csv_sanitizes_by_default(tmp_path):
    out = tmp_path / "out.csv"
    result = fd.clean_csv(_write_input(tmp_path), output_path=out)
    text = out.read_text()
    assert "'=1+1" in text and "'@SUM(A1)" in text
    # the returned frame is NOT sanitized — only the written artifact is
    assert (result["note"] == "=1+1").any()


def test_clean_csv_sanitize_formulas_opt_out(tmp_path):
    out = tmp_path / "out.csv"
    fd.clean_csv(_write_input(tmp_path), output_path=out, sanitize_formulas=False)
    text = out.read_text()
    assert "=1+1" in text and "'=1+1" not in text


# --------------------------------------------------------------------------- #
# Streaming CLI: sanitizes by default, --no-sanitize-formulas opt-out
# --------------------------------------------------------------------------- #


def test_stream_cli_sanitize_formulas_flag(tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"a": [1, 2], "note": ["=1+1", "safe"]}).to_csv(src, index=False)

    san_out = tmp_path / "san.csv"
    rc = cli_main(["stream", str(src), "-o", str(san_out), "--quiet"])
    assert rc == 0
    assert "'=1+1" in san_out.read_text()  # default: sanitized

    raw_out = tmp_path / "raw.csv"
    rc = cli_main(
        ["stream", str(src), "-o", str(raw_out), "--quiet", "--no-sanitize-formulas"]
    )
    assert rc == 0
    assert "'=1+1" not in raw_out.read_text()  # explicit opt-out: fidelity


# --------------------------------------------------------------------------- #
# Multi-row headers, index labels and axis names
# --------------------------------------------------------------------------- #

_LIVE_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _multi_header_input(tmp_path) -> str:
    src = tmp_path / "in.csv"
    src.write_text('=HYPERLINK("http://example.invalid"),b\n@SUM(A1),c\n1,x\n2,y\n')
    return str(src)


def test_clean_csv_guards_every_level_of_a_multi_row_header(tmp_path):
    out = tmp_path / "out.csv"
    fd.clean_csv(
        _multi_header_input(tmp_path), output_path=out, verbose=False,
        read_csv_kwargs={"header": [0, 1]},
    )
    lines = out.read_text().splitlines()
    assert lines[0] == '"\'=HYPERLINK(""http://example.invalid"")",b'
    assert lines[1] == "'@SUM(A1),c"
    back = pd.read_csv(out, header=[0, 1])
    assert not any(str(v).startswith(_LIVE_PREFIXES) for t in back.columns for v in t)


def test_clean_csv_multi_row_header_opt_out_keeps_raw_header(tmp_path):
    out = tmp_path / "out.csv"
    fd.clean_csv(
        _multi_header_input(tmp_path), output_path=out, verbose=False,
        read_csv_kwargs={"header": [0, 1]}, sanitize_formulas=False,
    )
    lines = out.read_text().splitlines()
    assert lines[0] == '"=HYPERLINK(""http://example.invalid"")",b'
    assert lines[1] == "@SUM(A1),c"


def test_sanitizer_guards_three_level_header_and_leaves_non_str_levels():
    columns = pd.MultiIndex.from_tuples(
        [("=a", -1, "+b"), ("ok", 2, "@c"), (" =d", -3, "e")],
        names=["=lvl0", 7, "safe"],
    )
    df = pd.DataFrame([[1, 2, 3]], columns=columns)
    out = sanitize_csv_formulas(df)
    assert isinstance(out.columns, pd.MultiIndex)
    assert list(out.columns) == [
        ("'=a", -1, "'+b"), ("ok", 2, "'@c"), ("' =d", -3, "e"),
    ]
    assert list(out.columns.names) == ["'=lvl0", 7, "safe"]
    # negative ints in a level are numbers, not formulas
    assert list(out.columns.get_level_values(1)) == [-1, 2, -3]
    assert out.to_numpy().tolist() == [[1, 2, 3]]


def test_sanitizer_guards_index_labels_and_names():
    df = pd.DataFrame(
        {"v": [1, 2, 3]},
        index=pd.Index(["=cmd|' /C calc'!A0", "safe", "@SUM(A1)"], name="+idx"),
    )
    df.columns.name = "-cols"
    out = sanitize_csv_formulas(df)
    assert list(out.index) == ["'=cmd|' /C calc'!A0", "safe", "'@SUM(A1)"]
    assert out.index.name == "'+idx"
    assert out.columns.name == "'-cols"
    assert out["v"].tolist() == [1, 2, 3]


def test_sanitizer_guards_multiindex_index_levels():
    index = pd.MultiIndex.from_tuples([("=a", 1), ("b", -2)], names=["@k", None])
    out = sanitize_csv_formulas(pd.DataFrame({"v": [1, 2]}, index=index))
    assert list(out.index) == [("'=a", 1), ("b", -2)]
    assert list(out.index.names) == ["'@k", None]


def test_sanitizer_keeps_numeric_and_datetime_axes_as_is():
    df = pd.DataFrame(
        {0: [1, 2]}, index=pd.date_range("2024-01-01", periods=2, name="when")
    )
    out = sanitize_csv_formulas(df)
    pd.testing.assert_frame_equal(out, df)
    assert isinstance(out.index, pd.DatetimeIndex)
    ranged = pd.DataFrame({"a": [-1, -2]})
    assert isinstance(sanitize_csv_formulas(ranged).index, pd.RangeIndex)


def test_sanitizer_does_not_mutate_input_axes():
    columns = pd.MultiIndex.from_tuples([("=a", "b")], names=["=n", "m"])
    index = pd.Index(["=x"], name="@i")
    df = pd.DataFrame([[1]], columns=columns, index=index)
    sanitize_csv_formulas(df)
    assert list(df.columns) == [("=a", "b")]
    assert list(df.columns.names) == ["=n", "m"]
    assert list(df.index) == ["=x"]
    assert df.index.name == "@i"


def test_clean_csv_guards_index_and_names_when_index_written(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("=id,name\n=a1,x\n@b2,y\n")
    out = tmp_path / "out.csv"
    result = fd.clean_csv(
        src, output_path=out, verbose=False,
        read_csv_kwargs={"index_col": 0}, to_csv_kwargs={"index": True},
    )
    assert out.read_text().splitlines() == ["'=id,name", "'=a1,x", "'@b2,y"]
    # the returned frame is not sanitized
    assert list(result.index) == ["=a1", "@b2"]
    assert result.index.name == "=id"


def test_clean_excel_guards_multi_row_header_and_index(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in (['=HYPERLINK("http://example.invalid")', "b"], ["@SUM(A1)", "c"],
                [1, "x"], [2, "y"]):
        ws.append(row)
    for row in ws.iter_rows(max_row=2):
        for cell in row:
            cell.data_type = "s"  # header text in the input, not formulas
    xlsx_in = tmp_path / "in.xlsx"
    wb.save(xlsx_in)

    out = tmp_path / "out.xlsx"
    # MultiIndex columns can only be written to Excel with index=True.
    fd.clean_excel(
        xlsx_in, output_path=out, verbose=False,
        read_excel_kwargs={"header": [0, 1]}, to_excel_kwargs={"index": True},
    )
    cells = [
        c for row in openpyxl.load_workbook(out).active.iter_rows()
        for c in row if c.value is not None
    ]
    assert not [c.coordinate for c in cells if c.data_type == "f"]
    values = {c.coordinate: c.value for c in cells}
    assert values["B1"] == '\'=HYPERLINK("http://example.invalid")'
    assert values["B2"] == "'@SUM(A1)"
    assert not [v for v in values.values() if str(v).startswith(_LIVE_PREFIXES)]
