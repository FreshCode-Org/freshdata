"""Tests for exception-table building and writing."""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata import QualityFinding, build_exception_table
from freshdata.findings import REDACT_TOKEN
from freshdata.integrations.exceptions import EXCEPTION_COLUMNS, write_exception_table


def _findings() -> list[QualityFinding]:
    return [
        QualityFinding.create(severity="error", step="privacy", column="ssn",
                              rule_name="US_SSN", message="pii", row_index=1,
                              observed_value="123-45-6789", action_taken="hash",
                              sensitive=True, lineage_run_id="RUN"),
        QualityFinding.create(severity="warning", step="domain", column="qty",
                              rule_name="qty_range", message="out of range",
                              row_index=2, lineage_run_id="RUN"),
    ]


def test_columns_and_redaction_by_default():
    table = build_exception_table(None, _findings())
    assert list(table.columns) == list(EXCEPTION_COLUMNS)
    assert len(table) == 2
    # observed_value redacted by default
    assert set(table["observed_value"]) == {REDACT_TOKEN}
    assert table.loc[table["column"] == "ssn", "lineage_run_id"].iloc[0] == "RUN"
    assert table.loc[table["column"] == "ssn", "source_row_id"].iloc[0] == 1
    assert (table["created_at"].str.endswith("Z")).all()


def test_include_pii_reveals_observed():
    table = build_exception_table(None, _findings(), include_pii=True)
    assert table.loc[table["column"] == "ssn", "observed_value"].iloc[0] == "123-45-6789"


def test_observed_value_enriched_from_df():
    df = pd.DataFrame({"qty": [10, 20, 999]})
    f = QualityFinding.create(severity="warning", step="domain", column="qty",
                              rule_name="r", message="m", row_index=2)
    table = build_exception_table(df, [f], include_pii=True)
    assert table["observed_value"].iloc[0] == "999"


def test_exception_id_is_stable():
    a = build_exception_table(None, _findings())["exception_id"].tolist()
    b = build_exception_table(None, _findings())["exception_id"].tolist()
    assert a == b


def test_accepts_a_report_object():
    import freshdata as fd
    _, report = fd.clean(pd.DataFrame({"a": [1, 2]}), return_report=True)
    table = build_exception_table(None, report)
    assert list(table.columns) == list(EXCEPTION_COLUMNS)


def test_write_csv(tmp_path):
    table = build_exception_table(None, _findings())
    path = tmp_path / "exc.csv"
    write_exception_table(table, str(path))
    back = pd.read_csv(path)
    assert list(back.columns) == list(EXCEPTION_COLUMNS)
    assert len(back) == 2


def test_write_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    table = build_exception_table(None, _findings())
    path = tmp_path / "exc.parquet"
    write_exception_table(table, str(path), format="parquet")
    back = pd.read_parquet(path)
    assert len(back) == 2


def test_write_duckdb(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    table = build_exception_table(None, _findings())
    path = tmp_path / "exc.duckdb"
    write_exception_table(table, str(path), format="duckdb")
    con = duckdb.connect(str(path))
    try:
        count = con.execute("SELECT count(*) FROM exceptions").fetchone()[0]
    finally:
        con.close()
    assert count == 2


def test_format_inferred_from_extension(tmp_path):
    table = build_exception_table(None, _findings())
    path = tmp_path / "exc.csv"
    write_exception_table(table, str(path), format=None)
    assert path.exists()


def test_unknown_format_raises(tmp_path):
    table = build_exception_table(None, _findings())
    with pytest.raises(ValueError, match="unknown exception-table format"):
        write_exception_table(table, str(tmp_path / "x.csv"), format="xml")
