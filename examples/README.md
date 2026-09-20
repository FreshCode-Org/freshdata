# FreshData Flagship & Integration Examples

Self-contained, runnable Python recipes demonstrating FreshData across real-world workflows, engines, and data engineering frameworks.

Each example generates its own synthetic data or uses bundled datasets, so you can run them directly:

```bash
# Core examples
python examples/01_csv_cleaning.py

# With optional extras
pip install "freshdata-cleaner[polars,duckdb,ml,airflow]"
python examples/integrations/polars_workflow.py
```

---

## Flagship Problem-Solving Recipes

| Recipe | Focus | Key Concepts Demonstrated |
|---|---|---|
| [`01_csv_cleaning.py`](01_csv_cleaning.py) | **Messy CSV repair** | Snake_case headers, whitespace trimming, dirty sentinels (`"N/A"`, `"-"`), currency parsing, JSON audit export |
| [`02_missing_values.py`](02_missing_values.py) | **Role-aware missing values** | Why identifiers & targets are preserved; median vs mode imputation; explicit rationale & confidence |
| [`03_duplicate_detection.py`](03_duplicate_detection.py) | **Duplicate detection & resolution** | Safe reporting default (no silent drops) vs. opt-in resolution with `duplicate_keep` semantics |
| [`04_outlier_handling.py`](04_outlier_handling.py) | **Non-destructive outlier handling** | Flagging outliers in boolean indicators (`_outlier`) vs controlled winsorization/capping vs removal |
| [`05_ml_preprocessing.py`](05_ml_preprocessing.py) | **ML pipeline readiness** | Zero target leakage invariant (`target_column`), scikit-learn integration, evaluation accuracy |

---

## Framework & Engine Integrations

Located in [`examples/integrations/`](integrations/):

| Integration | File | Description |
|---|---|---|
| **pandas** | [`pandas_workflow.py`](integrations/pandas_workflow.py) | Drop-in cleaning step within an existing pandas aggregation workflow. |
| **Polars** | [`polars_workflow.py`](integrations/polars_workflow.py) | Native `Polars in -> Polars out` execution with downstream Polars query expressions. |
| **DuckDB** | [`duckdb_workflow.py`](integrations/duckdb_workflow.py) | Direct cleaning of `DuckDBPyRelation` objects and registering back to DuckDB for SQL. |
| **scikit-learn** | [`sklearn_pipeline.py`](integrations/sklearn_pipeline.py) | Custom `FreshDataCleaner` transformer for native `sklearn.pipeline.Pipeline` usage. |
| **Apache Airflow** | [`airflow_task.py`](integrations/airflow_task.py) | Quality gating with `evaluate_trust_gate` and `FreshDataCleanOperator` DAG pattern. |
| **Pandera** | [`09_pandera_recipe.py`](09_pandera_recipe.py) | Pre- and post-validation using declarative Pandera DataFrame schemas. |
| **PyJanitor** | [`10_pyjanitor_interop.py`](10_pyjanitor_interop.py) | Combining explicit PyJanitor transforms with FreshData quality repair. |
| **Great Expectations** | [`11_great_expectations_recipe.py`](11_great_expectations_recipe.py) | Repairing with freshdata, then validating through a Great Expectations checkpoint. |

---

## Specialized Guides & Demos

* [`freshdata_ai_copilot_demo.py`](freshdata_ai_copilot_demo.py) — Flagship offline, deterministic AI Copilot cleaning demo.
* [`cleaning_memory_replay.py`](cleaning_memory_replay.py) — Learning cleaning rules from one dataset and replaying them on future batches.
* [`interactive_html_reports.py`](interactive_html_reports.py) — Generating standalone interactive HTML audit dashboards for non-technical stakeholders.
