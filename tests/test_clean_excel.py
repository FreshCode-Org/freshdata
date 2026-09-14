"""``fd.clean_excel``: the Excel companion to ``fd.clean_csv`` (#166)."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

import freshdata as fd

openpyxl = pytest.importorskip("openpyxl")


def _write_workbook(path, sheets):
    """Write ``{sheet: rows}`` with openpyxl so ``=`` strings stay text cells."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    cell.data_type = "s"
    wb.save(path)
    return path


MESSY = [
    ["Name ", "Age", "note"],
    [" Ada", 36, "=1+1"],
    ["Grace", None, "ok"],
    ["Alan", 41, "@SUM(A1)"],
]


def _cells(path):
    ws = openpyxl.load_workbook(path).active
    return {c.value: c.data_type for row in ws.iter_rows() for c in row if c.value is not None}


def test_round_trip_matches_clean(tmp_path):
    src = _write_workbook(tmp_path / "in.xlsx", {"data": MESSY})
    out = tmp_path / "out.xlsx"

    result = fd.clean_excel(src, output_path=out, sanitize_formulas=False)

    expected = fd.clean(pd.read_excel(src))
    pd.testing.assert_frame_equal(pd.DataFrame(result), pd.DataFrame(expected))
    assert list(result.columns) == ["name", "age", "note"]
    written = pd.read_excel(out)
    safe_cols = ["name", "age"]
    pd.testing.assert_frame_equal(
        written[safe_cols], pd.DataFrame(result)[safe_cols], check_dtype=False
    )


def test_return_report_and_report_alias(tmp_path):
    src = _write_workbook(tmp_path / "in.xlsx", {"data": MESSY})

    cleaned, report = fd.clean_excel(src, return_report=True)
    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(report, fd.CleanReport)

    cleaned_alias, report_alias = fd.clean_excel(src, report=True)
    assert isinstance(report_alias, fd.CleanReport)


def test_sheet_selection(tmp_path):
    src = _write_workbook(
        tmp_path / "in.xlsx",
        {"first": MESSY, "second": [["id", "value"], [1, "a"], [2, "b"]]},
    )

    cleaned = fd.clean_excel(src, read_excel_kwargs={"sheet_name": "second"})
    assert list(cleaned.columns) == ["id", "value"]
    assert len(cleaned) == 2

    with pytest.raises(TypeError, match="single sheet"):
        fd.clean_excel(src, read_excel_kwargs={"sheet_name": None})


def test_sanitizes_formulas_by_default(tmp_path):
    src = _write_workbook(
        tmp_path / "in.xlsx", {"data": [["=HDR", "note"], ["x", "=1+1"], ["y", "@SUM(A1)"]]}
    )
    out = tmp_path / "out.xlsx"

    result = fd.clean_excel(src, output_path=out)

    cells = _cells(out)
    assert cells.get("'=1+1") == "s"
    assert cells.get("'@SUM(A1)") == "s"
    assert not any(t == "f" for t in cells.values())
    # the returned frame is NOT sanitized — only the written artifact is
    assert (result["note"] == "=1+1").any()


def test_sanitize_formulas_opt_out_writes_formula_cells(tmp_path):
    src = _write_workbook(tmp_path / "in.xlsx", {"data": [["note"], ["=1+1"], ["ok"]]})
    out = tmp_path / "out.xlsx"

    fd.clean_excel(src, output_path=out, sanitize_formulas=False)

    assert _cells(out).get("=1+1") == "f"


def test_signature_matches_clean_csv():
    def names(fn, renames=None):
        renames = renames or {}
        return [renames.get(p, p) for p in inspect.signature(fn).parameters]

    assert names(fd.clean_excel) == names(
        fd.clean_csv,
        {"read_csv_kwargs": "read_excel_kwargs", "to_csv_kwargs": "to_excel_kwargs"},
    )


def test_exported():
    assert "clean_excel" in fd.__all__
    assert fd.clean_excel is not None
