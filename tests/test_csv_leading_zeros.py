"""CSV entry points keep zero-padded ZIP/ID columns as text (#228).

``pandas.read_csv`` turns ``"02134"`` into ``2134`` before cleaning starts, so the
CLI ``clean``/``stream`` commands and ``fd.clean_csv`` pre-scan the file and read
zero-padded numeric columns as ``str``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

import freshdata as fd
from freshdata import _csv_io
from freshdata._csv_io import leading_zero_dtypes
from freshdata.enterprise.cli import main

ROWS = [
    ("02134", "007", "Boston", 10),
    ("00501", "042", "Holtsville", 20),
    ("10001", "100", "New York", 30),
    ("94105", "123", "San Francisco", 40),
    ("60601", "999", "Chicago", 50),
]


def _write_input(path: Path) -> Path:
    lines = ["zip,account_id,city,amount"]
    lines += [",".join(map(str, row)) for row in ROWS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _assert_padding_kept(rows: list[dict[str, str]]) -> None:
    assert [r["zip"] for r in rows] == [row[0] for row in ROWS]
    assert [r["account_id"] for r in rows] == [row[1] for row in ROWS]
    # The unpadded numeric column is still written as plain integers.
    assert [r["amount"] for r in rows] == [str(row[3]) for row in ROWS]


def test_scan_selects_only_zero_padded_numeric_columns(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text(
        "zip,code,ratio,label,amount\n02134,007,0.5,007x,1\n10001,abc,1.5,y,2\n",
        encoding="utf-8",
    )
    # 'code' mixes a padded number with text (pandas keeps it as text anyway),
    # 'ratio' starts with "0." (not padding), 'amount' has no padding.
    assert leading_zero_dtypes(src) == {"zip": str}


def test_cli_clean_preserves_leading_zeros(tmp_path):
    src = _write_input(tmp_path / "in.csv")
    out = tmp_path / "out.csv"

    assert main(["clean", str(src), "-o", str(out), "--quiet"]) == 0

    _assert_padding_kept(_read_rows(out))


def test_cli_stream_preserves_leading_zeros_across_chunks(tmp_path):
    src = _write_input(tmp_path / "in.csv")
    out = tmp_path / "out.csv"

    rc = main(["stream", str(src), "-o", str(out), "--batch-size", "2", "--quiet"])

    assert rc == 0
    _assert_padding_kept(_read_rows(out))


def test_cli_stream_uses_the_first_chunk_mapping_for_every_chunk(tmp_path, monkeypatch):
    src = _write_input(tmp_path / "in.csv")
    real_read_csv = pd.read_csv
    chunked_dtypes: list[object] = []

    def spy(path, **kwargs):
        if kwargs.get("chunksize"):
            chunked_dtypes.append(kwargs.get("dtype"))
        return real_read_csv(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    rc = main(
        ["stream", str(src), "-o", str(tmp_path / "out.csv"), "--batch-size", "2", "--quiet"]
    )

    assert rc == 0
    assert chunked_dtypes == [{"zip": str, "account_id": str}]


def test_clean_csv_preserves_leading_zeros(tmp_path):
    src = _write_input(tmp_path / "in.csv")
    out = tmp_path / "out.csv"

    cleaned = fd.clean_csv(src, output_path=out, verbose=False)

    assert cleaned["zip"].tolist() == [row[0] for row in ROWS]
    assert cleaned["account_id"].tolist() == [row[1] for row in ROWS]
    assert pd.api.types.is_integer_dtype(cleaned["amount"])
    _assert_padding_kept(_read_rows(out))


def test_clean_csv_explicit_dtype_skips_the_scan(tmp_path, monkeypatch):
    src = _write_input(tmp_path / "in.csv")
    calls: list[dict[str, object]] = []
    real_read_csv = pd.read_csv

    def spy(path, **kwargs):
        calls.append(kwargs)
        return real_read_csv(path, **kwargs)

    monkeypatch.setattr(_csv_io.pd, "read_csv", spy)
    cleaned = fd.clean_csv(src, read_csv_kwargs={"dtype": {"city": str}}, verbose=False)

    assert len(calls) == 1  # no pre-scan read
    assert calls[0]["dtype"] == {"city": str}
    assert cleaned["zip"].tolist() == [int(row[0]) for row in ROWS]


def test_clean_csv_preserve_leading_zeros_false_skips_the_scan(tmp_path):
    src = _write_input(tmp_path / "in.csv")

    cleaned = fd.clean_csv(src, preserve_leading_zeros=False, verbose=False)

    assert cleaned["zip"].tolist() == [int(row[0]) for row in ROWS]
    assert leading_zero_dtypes(src, read_csv_kwargs={"converters": {"zip": int}}) == {}


def test_clean_csv_forwards_read_options_to_the_scan(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("zip;amount\n02134;1\n00501;2\n", encoding="utf-8")

    cleaned = fd.clean_csv(src, read_csv_kwargs={"sep": ";"}, verbose=False)

    assert cleaned["zip"].tolist() == ["02134", "00501"]
