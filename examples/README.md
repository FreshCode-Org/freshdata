# freshdata examples

Runnable, self-contained scripts. Each generates its own synthetic data, so you
can run any of them directly:

```bash
pip install "freshdata-cleaner[ml]"
python examples/01_missing_values.py
```

| Script | What it shows |
|---|---|
| [`freshdata_ai_copilot_demo.py`](freshdata_ai_copilot_demo.py) | **Flagship demo** — messy customer data → masked, validated, audit-ready (uses the bundled [`data/messy_customers.csv`](data/messy_customers.csv)) |
| [`01_missing_values.py`](01_missing_values.py) | Smart, role-aware missing-value imputation |
| [`02_outliers.py`](02_outliers.py) | Outlier detection — flag vs remove |
| [`03_normalization.py`](03_normalization.py) | Feature normalization for machine learning |
| [`04_profiling.py`](04_profiling.py) | Read-only data profiling and EDA |
| [`05_ml_pipeline.py`](05_ml_pipeline.py) | End-to-end ML preprocessing with scikit-learn |
| [`06_large_dataset.py`](06_large_dataset.py) | Cleaning a large synthetic dataset, with timing |
| [`07_pandas_integration.py`](07_pandas_integration.py) | Dropping freshdata into an existing pandas workflow |
| [`08_csv_automation.py`](08_csv_automation.py) | Batch CSV cleaning automation with audit logs |
| [`09_pandera_recipe.py`](09_pandera_recipe.py) | Validating with pandera before and after freshdata cleaning |
| [`10_pyjanitor_interop.py`](10_pyjanitor_interop.py) | Combining explicit PyJanitor transforms with FreshData quality repair |

The PyJanitor example has one optional dependency. FreshData 2.0 supports
pandas 1.5–2.x, so install the compatible PyJanitor 0.31 line before running it:

```bash
pip install "pyjanitor<0.32"
python examples/10_pyjanitor_interop.py
```

See the [documentation](https://freshcode-org.github.io/freshdata/) for full guides.
