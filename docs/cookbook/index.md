---
title: FreshData Cookbook
description: Copy-paste, problem-focused recipes for automated and explainable tabular data cleaning in Python.
keywords: python data cleaning cookbook, pandas recipes, polars cleaning, clean dataframe recipes
---

# FreshData Cookbook

Practical, copy-paste recipes targeting common data cleaning, validation, and pipeline challenges in Python.

Every recipe is self-contained with runnable code, realistic sample outputs, and architectural explanations.

---

## High-Intent Recipes

<div class="grid cards" markdown>

- :material-table: **[How to automatically clean a pandas DataFrame](clean-pandas-dataframe.md)**
    Standardize column headers, trim whitespace, fix dtypes, and normalize sentinel strings in one line.
    *(Search intent: automatic pandas data cleaning, clean dataframe python)*

- :material-help-box: **[How to safely handle missing values](safely-handle-missing-values.md)**
    Role-aware imputation: why identifiers and targets must be preserved while numerics are safely imputed.
    *(Search intent: safe missing value imputation, fillna without corruption)*

- :material-content-copy: **[How to detect and resolve duplicate rows](detect-and-resolve-duplicates.md)**
    Report duplicate clusters without silent row drops, and apply controlled deduplication policies.
    *(Search intent: detect duplicate rows pandas, safe deduplication python)*

- :material-robot: **[How to prepare tabular data for machine learning](prepare-data-for-machine-learning.md)**
    Zero-leakage data hygiene for scikit-learn, XGBoost, and PyTorch pipelines.
    *(Search intent: ml tabular data preprocessing, prevent target leakage)*

- :material-lightning-bolt: **[How to clean Polars DataFrames](clean-polars-dataframes.md)**
    Native `Polars in -> Polars out` data cleaning with zero pandas conversion overhead.
    *(Search intent: polars clean dataframe, polars data quality)*

- :material-pipe: **[How to add data cleaning to an ETL pipeline](data-cleaning-in-etl-pipeline.md)**
    Integrate FreshData and quality trust gates into Airflow, Prefect, and Dagster workflows.
    *(Search intent: etl data cleaning pipeline, airflow data quality gate)*

- :material-clipboard-check: **[How to audit automated data cleaning](audit-automated-data-cleaning.md)**
    Generate machine-readable JSON and Markdown audit trails documenting every transformed cell.
    *(Search intent: data cleaning audit trail, explainable data cleaning)*

</div>
