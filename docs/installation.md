---
title: Installation
description: >-
  How to install freshdata for Python and pandas, including optional extras for
  scikit-learn machine-learning imputation, the enterprise layer, and Polars.
keywords: install freshdata, pip install freshdata-cleaner, pandas data cleaning install
---

# Installation

`freshdata` requires **Python ≥ 3.9** and **pandas ≥ 1.5**.

## Basic install

```bash
pip install freshdata-cleaner
```

This installs the pandas + NumPy core plus FreshData's standard reporting and
self-contained HTML visualization. You do not need an extra to call
`fd.clean(df).summary()`, `fd.clean(df).report()`, or `fd.clean(df).visualize()`.

## Optional extras

Install only what you need:

=== "Machine learning"

    ```bash
    pip install "freshdata-cleaner[ml]"
    ```

    Adds **scikit-learn** for KNN imputation and IsolationForest outlier
    detection (used in `strategy="aggressive"`).

=== "Enterprise"

    ```bash
    pip install "freshdata-cleaner[enterprise]"
    ```

    Adds **polars, pyarrow, requests, pyyaml** for the enterprise layer:
    fuzzy clustering, PII masking, semantic validation, trust scoring,
    OpenLineage metadata, and the batch CLI.

=== "Everything"

    ```bash
    pip install "freshdata-cleaner[all]"
    ```

    All extras above plus **cleanlab** for ML label-noise detection.

=== "Polars only"

    ```bash
    pip install "freshdata-cleaner[polars]"
    ```

    Pass a Polars DataFrame to `fd.clean` and get a Polars DataFrame back.

!!! note "Python 3.9 on Linux aarch64: the `privacy` extra builds from source"
    `freshdata-cleaner[privacy]` (and `[all]`) pulls in spaCy through Presidio.
    On Python 3.9 spaCy is capped at 3.8.7, which requires `thinc>=8.3.4,<8.4`,
    and neither thinc 8.3.4 nor blis 1.2.0 publishes a cp39 Linux aarch64 wheel.
    pip therefore compiles thinc and blis from source there, which needs a C/C++
    toolchain and takes several minutes. Python 3.10+ or x86-64 installs use
    prebuilt wheels.

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
pip install freshdata-cleaner
```

```python
import freshdata as fd
```
