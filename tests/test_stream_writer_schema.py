"""``freshdata stream`` output stays well-formed when batch dtypes/columns vary (#248a).

The first batch fixes the output columns (CSV) and schema (Parquet); later batches
are reindexed / cast to it, and anything that does not fit raises. Output goes to a
sibling ``<output>.partial`` that only replaces the final path on success.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from freshdata.enterprise.cli import main
from freshdata.streaming._cli import _BatchWriter, _run_stream

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


class _PassThroughCleaner:
    """Stand-in for StreamingCleaner that yields the batches unchanged."""

    _gate_failures: list[object] = []

    def clean_batches(self, batches):
        for i, df in enumerate(batches):
            yield df, SimpleNamespace(streaming={"batch_id": i}, to_dict=dict)

    def finalize(self):
        return SimpleNamespace(streaming={}, to_dict=dict)


def _run(batches: list[pd.DataFrame], out: Path) -> int:
    return _run_stream(
        _PassThroughCleaner(), iter(batches), _BatchWriter(str(out)), report_dir=None, quiet=True
    )


def _partial(out: Path) -> Path:
    return out.with_name(out.name + ".partial")


def test_csv_batches_are_aligned_to_the_first_batch_columns(tmp_path):
    out = tmp_path / "out.csv"
    first = pd.DataFrame(
        {"a": [0, 1], "b": [0, 10], "a_flag": [False, False], "b_flag": [False, True]}
    )
    # 'a' flips to text and its flag column is missing; columns arrive reordered.
    second = pd.DataFrame({"b_flag": [True, False], "b": [30, 40], "a": ["oops", "4"]})

    assert _run([first, second], out) == 0

    with out.open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["a", "b", "a_flag", "b_flag"]
    assert all(len(r) == 4 for r in rows)
    assert rows[3] == ["oops", "30", "", "True"]
    assert rows[4] == ["4", "40", "", "False"]
    assert not _partial(out).exists()


def test_parquet_batches_are_cast_to_the_first_schema(tmp_path):
    out = tmp_path / "out.parquet"
    first = pd.DataFrame({"id": [1, 2], "v": [1, 2]})
    second = pd.DataFrame({"id": [3, 4], "v": [None, 4.0]})  # int64 -> double

    assert _run([first, second], out) == 0

    table = pq.read_table(out)
    assert table.schema.field("v").type == pa.int64()
    assert table.num_rows == 4
    assert table.column("v").to_pylist() == [1, 2, None, 4]
    assert not _partial(out).exists()


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_unexpected_new_column_raises_and_leaves_no_output(tmp_path, suffix):
    out = tmp_path / f"out{suffix}"
    first = pd.DataFrame({"a": [1, 2]})
    second = pd.DataFrame({"a": [3, 4], "surprise": ["x", "y"]})

    with pytest.raises(ValueError, match="surprise"):
        _run([first, second], out)

    assert not out.exists()
    assert not _partial(out).exists()


def test_parquet_uncastable_batch_raises_and_keeps_previous_output(tmp_path):
    out = tmp_path / "out.parquet"
    pd.DataFrame({"v": [7]}).to_parquet(out, index=False)
    first = pd.DataFrame({"v": [1, 2]})
    second = pd.DataFrame({"v": [2.5, 3.0]})  # would truncate into int64

    with pytest.raises(ValueError, match="Parquet schema"):
        _run([first, second], out)

    assert pd.read_parquet(out)["v"].tolist() == [7]  # untouched
    assert not _partial(out).exists()


def test_cli_stream_parquet_with_dtype_change_writes_every_row(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("id,v\n1,1\n2,2\n3,\n4,4\n", encoding="utf-8")
    out = tmp_path / "out.parquet"

    rc = main(["stream", str(src), "-o", str(out), "--batch-size", "2", "--quiet"])

    assert rc == 0
    assert len(pd.read_parquet(out)) == 4
    assert not _partial(out).exists()


def test_cli_stream_csv_anomaly_flags_stay_under_their_headers(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text(
        "ts,a,b\n2024-01-01,0,0\n2024-01-02,1,10\n2024-01-03,2,20\n"
        "2024-01-04,oops,30\n2024-01-05,4,40\n2024-01-06,5,50\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"

    rc = main(
        [
            "stream",
            str(src),
            "-o",
            str(out),
            "--batch-size",
            "3",
            "--quiet",
            "--timestamp",
            "ts",
            "--anomaly",
            "mad",
        ]
    )

    assert rc == 0
    with out.open(newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    assert "b_anomaly" in header
    assert all(len(r) == len(header) for r in rows)
    assert len(rows) == 7
    assert not _partial(out).exists()
