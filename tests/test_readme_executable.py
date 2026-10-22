# tests/test_readme_executable.py
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import freshdata as fd
from freshdata.config import CleanConfig


def test_readme_fenced_python_blocks_execute() -> None:
    """Parse every runnable fenced python block in README.md and execute in an isolated namespace."""
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    if not readme_path.exists():
        return
    content = readme_path.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", content, flags=re.DOTALL)

    sample_df = pd.DataFrame({
        "customer_id": pd.Series([1001, 1002, 1003, 1004], dtype="Int64"),
        "name": pd.Series([" Alice ", "Bob", "null", "Diana"], dtype="string"),
        "amount": pd.Series([100.0, 200.0, np.nan, 400.0], dtype="Float64"),
        "is_churned": [0, 1, 0, 1],
    })

    for idx, block in enumerate(blocks):
        if "read_csv(" in block:
            continue  # Skip external file I/O blocks or mock them
        ns = {"fd": fd, "pd": pd, "pl": pl, "df": sample_df.copy(deep=True)}
        exec(compile(block, f"<README_block_{idx}>", "exec"), ns)


def test_readme_core_claims_locked() -> None:
    """Lock every checkable claim in README.md to deterministic assertions."""
    df = pd.DataFrame({
        "customer_id": [1001.0, 1002.0, np.nan, 1004.0, 1005.0],
        "target": [0, 1, np.nan, 0, 1],
        "feature": [" 10.5 ", "20.0", "N/A", "30.0", "20.0"],
    })
    cleaned, report = fd.clean(
        df,
        config=CleanConfig(target_column="target", fix_dtypes=True, impute_missing=True),
        return_report=True,
    )
    # Claim 1: Never imputes an identifier
    assert pd.isna(cleaned["customer_id"].iloc[2])
    # Claim 2: Never modifies target_column
    assert pd.isna(cleaned["target"].iloc[2])
    # Claim 3: Nothing happens silently (every dtype change and imputation recorded)
    assert "feature" in report.dtype_changes
    assert report.values_imputed == 1
    # Claim 4: Native Polars DataFrame round-trip
    pl_in = pl.DataFrame({"a": [" x ", "y"], "b": [1, 2]})
    pl_out = fd.clean(pl_in)
    assert isinstance(pl_out, pl.DataFrame)
