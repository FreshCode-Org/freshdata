"""Benchmark harness for the freshdata out-of-core execution engine.

Generates synthetic Parquet at a target row count (never materialising the whole
dataset in memory) and times ``fd.clean`` across the pandas / polars / duckdb
backends, recording wall time, peak resident memory, and throughput.

Run from the command line::

    python -m freshdata.benchmarks.run_benchmarks --sizes 10k,1m,10m \\
        --engines pandas,polars,duckdb

Install the extras first: ``pip install 'freshdata-cleaner[outofcore,bench]'``.
"""

from ._data_gen import generate_parquet
from ._harness import BenchmarkHarness, BenchmarkResult

__all__ = ["generate_parquet", "BenchmarkHarness", "BenchmarkResult"]
