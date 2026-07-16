# FreshData Benchmark Suite

A comparative benchmark suite for `freshdata-cleaner` using
[Airspeed Velocity (ASV)](https://asv.readthedocs.io/). This suite is run
locally and its raw ASV results are committed under `.asv/results/`; it is
separate from the repository's CI `Benchmark` workflow, which runs the
[CleanBench harness](../benchmarks/) in `benchmarks/`.

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

## FreshData vs Pandas — Focused Comparison

The headline benchmark compares **FreshData** and **Pandas** head-to-head across six core data operations at three dataset sizes (10K, 100K, 1M rows):

| # | Operation | FreshData | Pandas |
|---|-----------|-----------|--------|
| 1 | **Loading** (CSV read + type inference) | `fd.clean(df, strategy="conservative")` | `pd.read_csv()` |
| 2 | **Missing values** (fill with mean) | `fd.clean(df, strategy="aggressive")` | `df.fillna(df.mean())` |
| 3 | **Outlier detection** (IQR flagging) | `fd.clean(df, outlier_method="iqr")` | Manual IQR per column |
| 4 | **Duplicate resolution** | `fd.clean(df, strategy="conservative")` | `df.drop_duplicates()` |
| 5 | **Group aggregations** | `fd.profile(df)` | `df.groupby().agg()` |
| 6 | **Full pipeline** | `fd.clean(df, strategy="balanced")` | Manual 7-step chain |

### Run the focused comparison only

```bash
# One-shot: installs deps, runs benchmarks, generates reports
bash bench/run_benchmarks.sh

# Quick mode (fewer iterations, for fast validation)
bash bench/run_benchmarks.sh --quick
```

## Reproduce Locally

To reproduce benchmark results exactly:

```bash
# Clone and enter the repo
git clone https://github.com/FreshCode-Org/freshdata.git
cd freshdata/freshdata-benchmarks

# Pin all randomness and threading for deterministic results
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Use the all-in-one runner (creates its own venv)
bash bench/run_benchmarks.sh

# Or run manually with ASV
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt asv virtualenv
asv machine --yes
asv run --bench "BenchmarkFreshDataVsPandas"
asv publish && asv preview
```

**Environment capture**: The runner script automatically saves machine details (Python version, OS, CPU, RAM, package versions) to `bench/artifacts/env_info.json`. You can also generate this manually:

```bash
python scripts/capture_env.py
```

## Interpreting Results

| Metric prefix | What it measures | Unit |
|---------------|------------------|------|
| `time_*` | Wall-clock execution time | seconds |
| `peakmem_*` | Peak resident set size (RSS) | bytes |
| `track_throughput_*` | Rows processed per second | rows/s |
| `track_speedup_*` | Speedup ratio vs Pandas baseline | ratio (×) |

- **Lower is better** for `time_*` and `peakmem_*`.
- **Higher is better** for `track_throughput_*` and `track_speedup_*`.
- ASV calibrates loop counts automatically and reports the **median** of ≥3 repeats.
- Results include warmup iterations (untimed) before each measurement.
- Standard deviations are captured to identify noisy measurements.

## Benchmark Matrix

The suite covers 15 domains using synthetic datasets (10K to 10M rows) with realistic data quality issues (NaNs, extreme outliers, sentinels, unicode anomalies, whitespaces):

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
14. **Group Aggregations** (`benchmark_groupagg.py`) - Groupby/profile comparisons
15. **FreshData vs Pandas** (`benchmark_freshdata_vs_pandas.py`) - Focused 6-operation head-to-head

## Published Results

This ASV suite is not wired into CI: run it locally with the commands above
and browse the results with `asv publish && asv preview`. Raw per-machine
results live in `.asv/results/` so runs are comparable over time. The
repository's CI `Benchmark` workflow runs the separate CleanBench harness
(`benchmarks/` at the repo root) on every push to `main` and weekly.

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for details on fairness enforcement, cache eviction, deep copies, and dataset generation.

