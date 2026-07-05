# Benchmarking Methodology

This document outlines the principles used to ensure fairness and reproducibility across the FreshData ASV benchmark suite.

## 1. Fairness Guarantees

Comparing libraries with entirely different architectures (e.g. Pandas eager execution vs. Polars lazy evaluation vs. Scikit-learn estimator arrays) requires strict fairness controls.

### Identical Inputs (`fresh_copy`)
Every library operates on the exact same dataset. The dataset is generated once in the `setup()` block using a fixed random seed.
Crucially, before timing begins, the benchmark calls `df.copy(deep=True)`. This prevents one library's in-place mutations from affecting subsequent runs or giving an unfair cache advantage.

### Isolation (`gc_collect`)
Garbage collection is explicitly triggered before each benchmark run to ensure residual objects from previous runs do not skew peak memory (`peakmem_`) measurements.

### Equivalent Operations
The benchmark measures *semantic* equivalence. The adapters in `benchmarks/utils/library_wrappers.py` map operations. If a library natively supports an operation (e.g. `polars.drop_nulls()`), it is used. If it requires manual chaining (e.g. `pandas.isnull().any()`), the most idiomatic pandas chain is used.

## 2. Statistical Approach

- **ASV Calibration**: ASV automatically calibrates the number of loops per benchmark so that the total execution time is statistically significant (defaulting to ~10ms total).
- **Repetitions**: Each benchmark is repeated at least 3 times. ASV reports the median time to reject outliers caused by OS background tasks.
- **Warmup**: ASV performs un-timed warmup runs before measuring.

## 3. Data Generation

The `benchmarks.datasets.data_generator` module produces a deterministic dataset specifically designed to stress data cleaning libraries. 

It does not generate "clean" random data. It generates:
- Numerics with 2% extreme outliers and 15% missing values
- Strings contaminated with trailing/leading whitespaces and varied casing
- High-cardinality categoricals
- Formatted currencies, URLs, and emails containing intentional syntax errors
- Duplicated rows representing ~5% of the total dataset length

## 4. Hardware and Threading

Numeric libraries (NumPy, SciPy via sklearn, Polars) use varying numbers of threads by default, which can obscure algorithmic efficiency. 

We configure `asv.conf.json` to enforce single-threaded execution during benchmarking:
```json
"env": {
    "OPENBLAS_NUM_THREADS": ["1"],
    "MKL_NUM_THREADS": ["1"],
    "OMP_NUM_THREADS": ["1"]
}
```
Multi-threaded performance scales linearly with hardware and is generally uninteresting for comparing underlying algorithmic complexity.
