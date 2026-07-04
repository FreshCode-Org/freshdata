---
title: Installation
description: >-
  How to install freshdata for Python and pandas, including optional extras for
  scikit-learn machine-learning imputation, the enterprise layer, and Polars.
keywords: install freshdata, pip install freshdata, pandas data cleaning install
---

# Installation

`freshdata` requires **Python ≥ 3.9** and **pandas ≥ 1.5**.

## Basic install

```bash
pip install freshdata
```

This installs the pandas + NumPy core plus FreshData's standard reporting and
self-contained HTML visualization. You do not need an extra to call
`fd.clean(df).summary()`, `fd.clean(df).report()`, or `fd.clean(df).visualize()`.

## Optional extras

Install only what you need:

=== "Machine learning"

    ```bash
    pip install "freshdata[ml]"
    ```

    Adds **scikit-learn** for KNN imputation and IsolationForest outlier
    detection (used in `strategy="aggressive"`).

=== "Enterprise"

    ```bash
    pip install "freshdata[enterprise]"
    ```

    Adds **polars, pyarrow, requests, pyyaml** for the enterprise layer:
    fuzzy clustering, PII masking, semantic validation, trust scoring,
    OpenLineage metadata, and the batch CLI.

=== "Everything"

    ```bash
    pip install "freshdata[all]"
    ```

    All extras above plus **cleanlab** for ML label-noise detection.

=== "Polars only"

    ```bash
    pip install "freshdata[polars]"
    ```

    Pass a Polars DataFrame to `fd.clean` and get a Polars DataFrame back.

## Verify the installation

```bash
python -c "import freshdata as fd; print(fd.__version__)"
```

```python
import pandas as pd
import freshdata as fd

df = pd.DataFrame({"a": [1, 2, 2, None], "b": [" x ", "y", "y", "z"]})
print(fd.clean(df))
```

## Note on naming

The PyPI distribution is **`freshdata`**, but the import name is simply
**`freshdata`** — so you install one and import the other:

```bash
pip install freshdata
```

```python
import freshdata as fd
```
