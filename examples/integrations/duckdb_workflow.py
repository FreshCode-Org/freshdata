"""examples/integrations/duckdb_workflow.py — DuckDB SQL & Out-of-Core Cleaning.

Problem:
--------
Modern data engineering pipelines use DuckDB for fast local analytical SQL
queries on Parquet and CSV files. However, querying messy tables directly with
SQL requires endless `CASE WHEN`, `TRY_CAST`, and `COALESCE` expressions.
FreshData provides a direct cleaning step on DuckDB tables and relations.

Installation:
-------------
pip install "freshdata-cleaner[duckdb]"
"""

from __future__ import annotations

import duckdb

import freshdata as fd


def main() -> None:
    print("=== Integration: DuckDB & FreshData ===")

    # 1. Setup in-memory DuckDB database with dirty table
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE sales_raw (
            transaction_id VARCHAR,
            customer_name VARCHAR,
            order_amount VARCHAR,
            discount_pct VARCHAR
        );
        INSERT INTO sales_raw VALUES
            ('TX-001', ' Alice Corp ', '$12,450.00', '5%'),
            ('TX-002', 'Bob LLC', 'N/A', '0%'),
            ('TX-003', 'Charlie Inc', '$8,900.50', 'missing'),
            ('TX-004', 'Diana Tech', '$34,000.00', '10%');
    """)

    print("\n[1] Raw DuckDB Table:")
    print(con.execute("SELECT * FROM sales_raw").fetchdf())

    # 2. Extract DuckDBPyRelation and pass directly to FreshData
    relation = con.table("sales_raw")
    cleaned_df = fd.clean(
        relation,
        preserve_columns=["transaction_id"],
        strategy="conservative",
    )

    print("\n[2] Cleaned DataFrame from DuckDB relation:")
    print(cleaned_df)

    # 3. Register cleaned data back to DuckDB for high-speed analytical SQL
    con.register("sales_clean", cleaned_df)
    print("\n[3] Analytical SQL Query on Cleaned Table:")
    result = con.execute("""
        SELECT
            customer_name,
            order_amount
        FROM sales_clean
        WHERE order_amount IS NOT NULL
        ORDER BY order_amount DESC
    """).fetchdf()
    print(result)


if __name__ == "__main__":
    main()
