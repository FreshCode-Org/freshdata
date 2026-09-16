"""File ingestion and round-trip safety for ``fd.clean_csv`` / ``fd.clean_excel``.

The entry points in :mod:`freshdata.api` do their own I/O, so the dtypes a caller
gets back are decided by the reader *before* any cleaning step runs. These tests
pin what survives that boundary:

* Excel serial dates (``45000``) — read as plain integers, never as dates.
* duplicate column headers in CSV and Excel.
* non-UTF-8 (cp1252/latin-1) input and mojibake.
* the full ``raw -> clean -> save -> reload -> clean`` cycle for both formats,
  where the second clean should be a no-op.

Two assertions below deliberately pin behaviour that loses information. They are
marked ``DEFECT`` / ``LIMITATION`` inline so that a fix fails the test loudly
instead of slipping through unnoticed.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
from freshdata._csv_io import LEADING_ZERO_SCAN_ROWS, leading_zero_dtypes

openpyxl = pytest.importorskip("openpyxl")

# Excel's day-zero epoch (the 1900 leap-year bug included), for serial -> date.
EXCEL_EPOCH = "1899-12-30"


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_xlsx(path: Path, rows: list[list[object]]) -> Path:
    """Write *rows* to a workbook, keeping ``str`` values as genuine text cells."""
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    for row in ws.iter_rows():
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    wb.save(path)
    return path


# --------------------------------------------------------------------------
# Excel serial dates
# --------------------------------------------------------------------------


def test_excel_serial_date_stays_an_integer_while_a_formatted_date_does_not(tmp_path):
    """A serial written as a bare number is data; only a date-formatted cell is a date."""
    path = tmp_path / "serials.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["id", "serial_date", "formatted_date"])
    for offset in range(3):
        ws.append([offset + 1, 45000 + offset, dt.datetime(2023, 3, 15 + offset)])
    wb.save(path)

    cleaned = fd.clean_excel(path)

    # The serial column is indistinguishable from any other integer column.
    assert cleaned["serial_date"].dtype == "int64"
    assert list(cleaned["serial_date"]) == [45000, 45001, 45002]
    # A cell openpyxl stored with a date format is read as a real timestamp.
    assert pd.api.types.is_datetime64_any_dtype(cleaned["formatted_date"])
    assert cleaned["formatted_date"].iloc[0] == pd.Timestamp("2023-03-15")


def test_excel_serial_date_is_not_converted_even_when_the_column_is_named_date(tmp_path):
    """The column name carries no weight: ``order_date`` of serials stays numeric."""
    path = _write_xlsx(
        tmp_path / "orders.xlsx",
        [["order_date", "amount"], *([45000 + i, 10 + i] for i in range(5))],
    )

    cleaned = fd.clean_excel(path)

    assert cleaned["order_date"].dtype == "int64"
    assert list(cleaned["order_date"]) == [45000, 45001, 45002, 45003, 45004]

    # The caller-side conversion FreshData does not perform, for the record: the
    # serials are real dates, and cleaning a frame that already holds them keeps them.
    converted = cleaned.assign(
        order_date=pd.to_datetime(cleaned["order_date"], unit="D", origin=EXCEL_EPOCH)
    )
    assert converted["order_date"].iloc[0] == pd.Timestamp("2023-03-15")
    assert pd.api.types.is_datetime64_any_dtype(fd.clean(converted)["order_date"])


# --------------------------------------------------------------------------
# Duplicate headers
# --------------------------------------------------------------------------


def test_duplicate_csv_headers_are_mangled_by_pandas_then_normalized(tmp_path):
    path = _write_csv(tmp_path / "dupes.csv", "id,name,name\n1,Ada,Lovelace\n2,Alan,Turing\n")

    # pandas de-duplicates on read; FreshData only normalizes the mangled label.
    assert list(pd.read_csv(path).columns) == ["id", "name", "name.1"]

    cleaned = fd.clean_csv(path)

    assert list(cleaned.columns) == ["id", "name", "name_1"]
    assert list(cleaned["name"]) == ["Ada", "Alan"]
    assert list(cleaned["name_1"]) == ["Lovelace", "Turing"]


def test_headers_colliding_only_after_normalization_keep_distinct_names(tmp_path):
    """``Name`` / ``name `` / ``NAME`` are distinct to pandas but collide once normalized."""
    path = _write_csv(tmp_path / "collide.csv", "Name,name ,NAME\n1,2,3\n")
    assert list(pd.read_csv(path).columns) == ["Name", "name ", "NAME"]

    cleaned = fd.clean_csv(path)

    assert list(cleaned.columns) == ["name", "name_2", "name_3"]
    assert cleaned.shape == (1, 3)
    assert list(cleaned.iloc[0]) == [1, 2, 3]


def test_duplicate_excel_headers_behave_exactly_like_csv(tmp_path):
    path = _write_xlsx(
        tmp_path / "dupes.xlsx",
        [["id", "name", "name"], [1, "Ada", "Lovelace"], [2, "Alan", "Turing"]],
    )

    cleaned = fd.clean_excel(path)

    assert list(cleaned.columns) == ["id", "name", "name_1"]
    assert list(cleaned["name_1"]) == ["Lovelace", "Turing"]


def test_deduplicated_header_names_are_stable_across_a_save_and_reload(tmp_path):
    path = _write_csv(tmp_path / "dupes.csv", "id,name,name\n1,Ada,Lovelace\n2,Alan,Turing\n")
    first = fd.clean_csv(path, output_path=tmp_path / "once.csv")

    second = fd.clean_csv(tmp_path / "once.csv")

    # ``name_1`` must not accrue another suffix on the second pass.
    assert list(second.columns) == list(first.columns) == ["id", "name", "name_1"]


# --------------------------------------------------------------------------
# Non-UTF-8 input
# --------------------------------------------------------------------------


CP1252_TEXT = "name,city,note\nJosé,Malmö,café\nRenée,Zürich,naïve\n"


def test_cp1252_csv_without_an_encoding_raises_the_bare_pandas_error(tmp_path):
    """SPEC GAP: ``clean_csv`` has no encoding option and adds no guidance to the error."""
    path = tmp_path / "cp1252.csv"
    path.write_bytes(CP1252_TEXT.encode("cp1252"))

    with pytest.raises(UnicodeDecodeError) as excinfo:
        fd.clean_csv(path)

    message = str(excinfo.value)
    assert "utf-8" in message
    assert "freshdata" not in message.lower()
    assert "encoding" not in message.lower()


def test_cp1252_csv_read_with_an_explicit_encoding_keeps_every_accent(tmp_path):
    path = tmp_path / "cp1252.csv"
    path.write_bytes(CP1252_TEXT.encode("cp1252"))

    cleaned = fd.clean_csv(path, read_csv_kwargs={"encoding": "cp1252"})

    assert list(cleaned["name"]) == ["José", "Renée"]
    assert list(cleaned["city"]) == ["Malmö", "Zürich"]
    assert list(cleaned["note"]) == ["café", "naïve"]


def test_cp1252_smart_quotes_read_as_latin1_become_control_characters(tmp_path):
    """latin-1 has no 0x93/0x94, so a cp1252 file read as latin-1 yields C1 controls."""
    path = tmp_path / "quotes.csv"
    path.write_bytes("name,v\n“Ada”,1\nBob,2\n".encode("cp1252"))

    as_latin1 = fd.clean_csv(path, read_csv_kwargs={"encoding": "latin-1"})
    as_cp1252 = fd.clean_csv(path, read_csv_kwargs={"encoding": "cp1252"})

    assert as_latin1["name"].iloc[0] == "\x93Ada\x94"
    assert as_cp1252["name"].iloc[0] == "“Ada”"


def test_mojibake_survives_clean_csv_untouched_but_the_text_lint_flags_it(tmp_path):
    """UTF-8 bytes mis-decoded as cp1252 are not repaired; ``lint_text_encoding`` sees them."""
    mojibake = "café".encode().decode("cp1252")  # 'cafÃ©'
    path = _write_csv(tmp_path / "mojibake.csv", f"note,v\n{mojibake},1\nok,2\n")

    cleaned = fd.clean_csv(path, read_csv_kwargs={"encoding": "utf-8"})

    assert list(cleaned["note"]) == [mojibake, "ok"]
    report = fd.lint_text_encoding(cleaned)
    assert "mojibake" in str(report)


# --------------------------------------------------------------------------
# raw -> clean -> save -> reload -> clean
# --------------------------------------------------------------------------


ROUND_TRIP_CSV = (
    "ZIP ,Account Id,Qty,When,Name\n"
    "02134,007,5,2023-01-05,Ada \n"
    "00501,042,,2023-02-06, grace\n"
    "10001,100,7,2023-03-07,Alan\n"
    "60601,999,9,2023-04-08,Bob\n"
    "94105,123,11,2023-05-09,Eve\n"
)


def test_csv_round_trip_second_clean_is_a_no_op(tmp_path):
    """Leading zeros, a nullable numeric column and parsed dates all survive the cycle."""
    raw = _write_csv(tmp_path / "raw.csv", ROUND_TRIP_CSV)
    once = tmp_path / "once.csv"
    twice = tmp_path / "twice.csv"

    first = fd.clean_csv(raw, output_path=once)
    second = fd.clean_csv(once, output_path=twice)

    assert list(first["zip"]) == ["02134", "00501", "10001", "60601", "94105"]
    assert list(first["account_id"]) == ["007", "042", "100", "999", "123"]
    assert pd.api.types.is_datetime64_any_dtype(first["when"])
    assert first["qty"].isna().sum() == 1

    pd.testing.assert_frame_equal(first, second)
    assert once.read_bytes() == twice.read_bytes()


def test_csv_round_trip_keeps_quoting_embedded_newlines_and_empty_cells(tmp_path):
    raw = _write_csv(
        tmp_path / "quoted.csv",
        'id,note,blank\n1,"a,b",\n2,"line1\nline2",\n3,plain,\n4,"say ""hi""",\n',
    )
    once = tmp_path / "once.csv"
    twice = tmp_path / "twice.csv"

    first = fd.clean_csv(raw, output_path=once)
    second = fd.clean_csv(once, output_path=twice)

    assert list(first["note"]) == ["a,b", "line1\nline2", "plain", 'say "hi"']
    # The all-empty column is dropped once and never comes back.
    assert "blank" not in first.columns
    pd.testing.assert_frame_equal(first, second)
    assert once.read_bytes() == twice.read_bytes()


def test_excel_round_trip_second_clean_is_a_no_op(tmp_path):
    raw = _write_xlsx(
        tmp_path / "raw.xlsx",
        [
            ["Order Id", "When", "Qty", "Name "],
            [1, dt.datetime(2023, 1, 5), 5, "Ada "],
            [2, dt.datetime(2023, 2, 6), None, " grace"],
            [3, dt.datetime(2023, 3, 7), 7, "Alan"],
            [4, dt.datetime(2023, 4, 8), 6, "Bob"],
        ],
    )
    once = tmp_path / "once.xlsx"

    first = fd.clean_excel(raw, output_path=once)
    second = fd.clean_excel(once, output_path=tmp_path / "twice.xlsx")

    assert list(first.columns) == ["order_id", "when", "qty", "name"]
    assert pd.api.types.is_datetime64_any_dtype(first["when"])
    assert list(first["name"]) == ["Ada", "grace", "Alan", "Bob"]
    pd.testing.assert_frame_equal(first, second)


def test_formula_sanitizing_rewrites_values_on_reload_then_stabilizes(tmp_path):
    """The written ``'`` prefix is real data on reload: cycle 1 -> 2 is NOT a no-op."""
    raw = _write_csv(
        tmp_path / "formulas.csv",
        "id,note\n1,=1+1\n2,ok\n3,@SUM(A1)\n4,-notice\n5,+44 20\n",
    )
    once = tmp_path / "once.csv"
    twice = tmp_path / "twice.csv"
    thrice = tmp_path / "thrice.csv"

    first = fd.clean_csv(raw, output_path=once)
    second = fd.clean_csv(once, output_path=twice)
    third = fd.clean_csv(twice, output_path=thrice)

    # The returned frame is never sanitized...
    assert list(first["note"]) == ["=1+1", "ok", "@SUM(A1)", "-notice", "+44 20"]
    # ...but the file is, so the next read sees different values.
    assert list(second["note"]) == ["'=1+1", "ok", "'@SUM(A1)", "'-notice", "'+44 20"]
    # The prefix is applied at most once: the cycle is idempotent from here on.
    pd.testing.assert_frame_equal(second, third)
    assert twice.read_bytes() == thrice.read_bytes()


def test_sanitize_formulas_false_round_trips_the_values_byte_exactly(tmp_path):
    raw = _write_csv(tmp_path / "formulas.csv", "id,note\n1,=1+1\n2,ok\n3,@SUM(A1)\n")
    once = tmp_path / "once.csv"
    twice = tmp_path / "twice.csv"

    first = fd.clean_csv(raw, output_path=once, sanitize_formulas=False)
    second = fd.clean_csv(once, output_path=twice, sanitize_formulas=False)

    assert list(second["note"]) == ["=1+1", "ok", "@SUM(A1)"]
    pd.testing.assert_frame_equal(first, second)
    assert once.read_bytes() == twice.read_bytes()


# --------------------------------------------------------------------------
# Leading zeros across the format boundary
# --------------------------------------------------------------------------


def test_excel_ingestion_drops_leading_zeros_that_the_csv_path_preserves(tmp_path):
    """DEFECT: ``preserve_leading_zeros`` is a no-op for ``clean_excel``.

    ``clean_csv`` pre-scans the file (``_csv_io.leading_zero_dtypes``) so a
    zero-padded numeric column is read as text. ``clean_excel`` calls
    ``pd.read_excel`` directly, and pandas' type inference turns the *text* cell
    ``"02134"`` into the integer ``2134`` — silently, with no report entry, even
    though ``clean_excel`` documents "the same options" as ``clean_csv``.
    """
    rows = [["zip", "city"], ["02134", "Boston"], ["00501", "Holtsville"], ["10001", "NY"]]
    xlsx = _write_xlsx(tmp_path / "zips.xlsx", rows)
    csv_path = _write_csv(
        tmp_path / "zips.csv", "zip,city\n02134,Boston\n00501,Holtsville\n10001,NY\n"
    )

    # The workbook really does hold text cells, so the loss is pandas' inference.
    stored = [c.value for c in openpyxl.load_workbook(xlsx).active["A"]]
    assert stored == ["zip", "02134", "00501", "10001"]

    assert list(fd.clean_csv(csv_path)["zip"]) == ["02134", "00501", "10001"]
    assert list(fd.clean_excel(xlsx)["zip"]) == [2134, 501, 10001]  # DEFECT
    assert list(fd.clean_excel(xlsx, preserve_leading_zeros=True)["zip"]) == [2134, 501, 10001]

    # The supported workaround, which callers must know to reach for.
    forced = fd.clean_excel(xlsx, read_excel_kwargs={"dtype": {"zip": str}})
    assert list(forced["zip"]) == ["02134", "00501", "10001"]


def test_csv_to_excel_round_trip_loses_the_leading_zeros_csv_had_kept(tmp_path):
    """The same defect seen end to end: a correct CSV clean is undone by the Excel hop."""
    csv_path = _write_csv(tmp_path / "zips.csv", "zip,v\n02134,1\n00501,2\n10001,3\n")
    xlsx = tmp_path / "zips.xlsx"

    first = fd.clean_csv(csv_path)
    first.to_excel(xlsx, index=False)
    second = fd.clean_excel(xlsx)

    assert list(first["zip"]) == ["02134", "00501", "10001"]
    assert list(second["zip"]) == [2134, 501, 10001]  # DEFECT: silent, no report entry


def test_leading_zero_prescan_stops_at_the_documented_row_limit(tmp_path):
    """LIMITATION (documented in ``_csv_io``): padding first seen past the scan is lost."""
    body = [f"1{i:04d},{i}" for i in range(LEADING_ZERO_SCAN_ROWS)]
    late = _write_csv(tmp_path / "late.csv", "\n".join(["zip,v", *body, "02134,999"]) + "\n")
    early = _write_csv(tmp_path / "early.csv", "\n".join(["zip,v", "02134,0", *body[:50]]) + "\n")

    assert leading_zero_dtypes(late) == {}
    assert leading_zero_dtypes(early) == {"zip": str}
    assert fd.clean_csv(late)["zip"].iloc[-1] == 2134
    assert fd.clean_csv(early)["zip"].iloc[0] == "02134"


# --------------------------------------------------------------------------
# Malformed and degenerate files
# --------------------------------------------------------------------------


def test_a_row_with_too_many_fields_raises_the_bare_pandas_parser_error(tmp_path):
    """SPEC GAP: no FreshData-level handling or hint for a ragged long row."""
    path = _write_csv(tmp_path / "long.csv", "a,b\n1,2\n3,4,5\n6,7\n")

    with pytest.raises(pd.errors.ParserError) as excinfo:
        fd.clean_csv(path)

    assert "Expected 2 fields in line 3, saw 3" in str(excinfo.value)


def test_a_row_with_too_few_fields_is_padded_with_missing_values(tmp_path):
    path = _write_csv(tmp_path / "short.csv", "a,b,c\n1,2,3\n4,5\n6,7,8\n")

    cleaned = fd.clean_csv(path)

    assert cleaned.shape == (3, 3)
    assert pd.isna(cleaned["c"].iloc[1])


def test_a_header_only_file_yields_an_empty_frame_that_keeps_its_columns(tmp_path):
    path = _write_csv(tmp_path / "header.csv", "a,b,c\n")

    cleaned = fd.clean_csv(path)

    assert cleaned.shape == (0, 3)
    assert list(cleaned.columns) == ["a", "b", "c"]


def test_an_empty_file_raises_empty_data_error(tmp_path):
    path = _write_csv(tmp_path / "empty.csv", "")

    with pytest.raises(pd.errors.EmptyDataError):
        fd.clean_csv(path)
