# FreshData Benchmark Suite

A comprehensive, scientifically rigorous comparative benchmark suite for `freshdata-cleaner` using [Airspeed Velocity (ASV)](https://asv.readthedocs.io/).

This suite measures the performance of FreshData's data cleaning operations against the Python data ecosystem:
- **Pandas** (baseline)
- **Polars**
- **pyjanitor**
- **Scikit-learn**
- **Feature-engine**
- **AutoClean**

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install asv

# 3. Configure ASV machine (run once)
asv machine --yes

# 4. Run the benchmarks
asv run

# 5. Generate reports
python scripts/generate_reports.py
python scripts/generate_plots.py

# 6. View ASV web interface
asv publish
asv preview
```

## Benchmark Matrix

The suite covers 13 domains using synthetic datasets (10K to 10M rows) with realistic data quality issues (NaNs, extreme outliers, sentinels, unicode anomalies, whitespaces):

1. **Pipeline** (`benchmark_pipeline.py`) - Full auto-cleaning pipeline
2. **Missing Values** (`benchmark_missing.py`) - Drop, mean, median, ffill
3. **Duplicates** (`benchmark_duplicates.py`) - Detect and drop
4. **Strings** (`benchmark_strings.py`) - Trim, lowercase, regex
5. **Types** (`benchmark_types.py`) - Downcasting, numeric parsing
6. **Columns** (`benchmark_columns.py`) - Renames and selections
7. **Encoding** (`benchmark_encoding.py`) - One-hot and ordinal
8. **Outliers** (`benchmark_outliers.py`) - IQR and Z-score
9. **Scaling** (`benchmark_scaling.py`) - Standard and MinMax
10. **Validation** (`benchmark_validation.py`) - Schema profiling
11. **Memory** (`benchmark_memory.py`) - Peak RSS tracking
12. **I/O** (`benchmark_io.py`) - Combined CSV/Parquet read + clean
13. **Scaling** (`benchmark_scaling_curves.py`) - Big-O efficiency analysis

## CI/CD and Published Results

Benchmarks are executed weekly via GitHub Actions on standard CI hardware (2 vCPU, 7GB RAM).
Results are automatically published to the `gh-pages` branch.

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for details on fairness enforcement, cache eviction, deep copies, and dataset generation.
