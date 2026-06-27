"""Arrow Table / RecordBatch as first-class inputs and outputs.

DuckDB and Polars operate on Arrow natively (no pandas materialization), and the
clean can round-trip Arrow in -> Arrow out.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd

pa = pytest.importorskip("pyarrow")
pytest.importorskip("polars")
pytest.importorskip("duckdb")


@pytest.fixture
def arrow_table() -> pa.Table:
    df = pd.DataFrame(
        {
            "Customer ID": ["X10", "Y20", "X10", "Z30", "Y20"],
            "amount": [1.0, 2.0, 1.0, 3.0, 2.0],
            " Note ": [" a ", "N/A", "a", "-", "b"],
            "empty": [None, None, None, None, None],
        }
    )
    return pa.Table.from_pandas(df, preserve_index=False)


def _native_config():
    return fd.CleanConfig(strategy="conservative", fix_dtypes=False, verbose=False)


@pytest.mark.parametrize("engine", ["polars", "duckdb"])
def test_arrow_table_input(arrow_table, engine):
    out, rep = fd.clean(arrow_table, config=_native_config(), engine=engine,
                        return_report=True)
    out = out if isinstance(out, pd.DataFrame) else out.to_pandas()
    assert "customer_id" in out.columns  # rename applied
    assert "empty" not in out.columns      # all-null column dropped
    assert rep.backend == engine


@pytest.mark.parametrize("engine", ["polars", "duckdb"])
def test_arrow_record_batch_input(arrow_table, engine):
    batch = arrow_table.to_batches()[0]
    out = fd.clean(batch, config=_native_config(), engine=engine)
    out = out if isinstance(out, pd.DataFrame) else out.to_pandas()
    assert "customer_id" in out.columns


@pytest.mark.parametrize("engine", ["polars", "duckdb"])
def test_arrow_roundtrip(arrow_table, engine):
    out = fd.clean(arrow_table, config=_native_config(), engine=engine,
                   output_format="arrow")
    assert isinstance(out, pa.Table)
    # Round-tripping back through pandas keeps the cleaned, renamed schema.
    names = out.schema.names
    assert "customer_id" in names
    assert "empty" not in names


def test_auto_engine_selects_for_arrow(arrow_table):
    # A bare Arrow table with default options routes through the engine layer.
    out, rep = fd.clean(arrow_table, config=_native_config(), return_report=True)
    out = out if isinstance(out, pd.DataFrame) else out.to_pandas()
    assert rep.backend in ("polars", "duckdb")
    assert "customer_id" in out.columns
