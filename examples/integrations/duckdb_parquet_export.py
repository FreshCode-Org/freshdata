"""examples/integrations/duckdb_parquet_export.py — Partitioned Parquet Cleaning & Export.

Problem:
--------
Analytical data pipelines often store large datasets as partitioned Parquet files
(e.g., partitioned by region or year). Raw partitions often contain messy values,
inconsistent types, and unstandardized strings. FreshData cleans the dataset, and
DuckDB writes the cleaned result back out into a partitioned Parquet destination.

Installation:
-------------
pip install "freshdata-cleaner[duckdb]"
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb

import freshdata as fd


def main() -> None:
    print("=== Integration: DuckDB Partitioned Parquet Export & FreshData ===")

    # 1. Manage DuckDB connection using a context manager
    with duckdb.connect() as con:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)
            raw_parquet_dir = base_path / "raw_sales"
            cleaned_parquet_dir = base_path / "cleaned_sales"

            # Escape single quotes for SQL literal interpolation
            raw_parquet_path = raw_parquet_dir.as_posix().replace("'", "''")
            cleaned_parquet_path = cleaned_parquet_dir.as_posix().replace("'", "''")

            # 2. Setup raw partitioned Parquet dataset
            print("\n[1] Creating raw partitioned Parquet dataset...")
            con.execute(f"""
                CREATE TABLE raw_data AS SELECT * FROM (VALUES
                    ('TX-101', 'North', '  Acme Corp  ', '$15,000.50', '2024-01-10'),
                    ('TX-102', 'North', 'Beta Ltd', 'N/A', '2024-01-12'),
                    ('TX-201', 'South', 'Gamma LLC', '$8,400.00', 'invalid_date'),
                    ('TX-202', 'South', 'Delta Inc', '$22,100.00', '2024-02-01')
                ) AS t(transaction_id, region, company_name, revenue, sale_date);

                COPY raw_data TO '{raw_parquet_path}' (
                    FORMAT PARQUET,
                    PARTITION_BY (region),
                    OVERWRITE_OR_IGNORE 1
                );
            """)

            # 3. Read partitioned Parquet files directly into a DuckDB relation using glob pattern
            print("\n[2] Reading partitioned Parquet dataset into DuckDB relation...")
            raw_relation = con.read_parquet(
                f"{raw_parquet_dir.as_posix()}/**/*.parquet",
                hive_partitioning=True,
            )
            print(raw_relation.df())

            # 4. Clean messy dataset using FreshData (materialize relation via fetchdf() to avoid catalog isolation issues)
            print("\n[3] Cleaning data with FreshData...")
            cleaned_df = fd.clean(
                raw_relation.fetchdf(),
                preserve_columns=["transaction_id", "region"],
                strategy="conservative",
            )
            print(cleaned_df)

            # 5. Register and re-export cleaned data back into partitioned Parquet destination
            print("\n[4] Exporting cleaned data to partitioned Parquet using DuckDB...")
            con.register("cleaned_table", cleaned_df)
            con.execute(f"""
                COPY (SELECT * FROM cleaned_table) TO '{cleaned_parquet_path}' (
                    FORMAT PARQUET,
                    PARTITION_BY (region),
                    OVERWRITE_OR_IGNORE 1
                );
            """)

            # 6. Verify row count and partition existence for self-validation
            print("\n[5] Verifying exported Parquet files:")
            verified_relation = con.read_parquet(
                f"{cleaned_parquet_dir.as_posix()}/**/*.parquet",
                hive_partitioning=True,
            )
            print(verified_relation.df())

            assert len(verified_relation) == 4
            assert (cleaned_parquet_dir / "region=North").exists()

            print("\nSuccessfully exported cleaned partitioned dataset!")


if __name__ == "__main__":
    main()