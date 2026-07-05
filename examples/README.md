# freshdata examples

Runnable, self-contained scripts. Each generates its own synthetic data, so you
can run any of them directly:

```bash
pip install "freshdata-cleaner[ml]"
python examples/01_missing_values.py
```

| Script | What it shows |
|---|---|
| [`01_missing_values.py`](01_missing_values.py) | Smart, role-aware missing-value imputation |
| [`02_outliers.py`](02_outliers.py) | Outlier detection — flag vs remove |
| [`03_normalization.py`](03_normalization.py) | Feature normalization for machine learning |
| [`04_profiling.py`](04_profiling.py) | Read-only data profiling and EDA |
| [`05_ml_pipeline.py`](05_ml_pipeline.py) | End-to-end ML preprocessing with scikit-learn |
| [`06_large_dataset.py`](06_large_dataset.py) | Cleaning a large synthetic dataset, with timing |
| [`07_pandas_integration.py`](07_pandas_integration.py) | Dropping freshdata into an existing pandas workflow |
| [`08_csv_automation.py`](08_csv_automation.py) | Batch CSV cleaning automation with audit logs |

See the [documentation](https://freshcode-org.github.io/freshdata/) for full guides.
