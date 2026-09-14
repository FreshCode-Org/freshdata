"""pyarrow requirement errors name the feature that needs it (#215)."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

import freshdata as fd
from freshdata.execution._lazy import require_pyarrow
from freshdata.execution._metadata import MetadataScanner


@pytest.fixture
def no_pyarrow(monkeypatch):
    # A ``None`` entry makes ``import pyarrow`` (and its submodules) raise ImportError.
    monkeypatch.setitem(sys.modules, "pyarrow", None)


def test_arrow_output_error_names_arrow_output(no_pyarrow):
    with pytest.raises(ImportError) as exc:
        fd.clean(pd.DataFrame({"a": [1.0, None]}), output_format="arrow", verbose=False)
    message = str(exc.value)
    assert "Arrow output" in message
    assert "Parquet" not in message
    assert "freshdata-cleaner[pyarrow]" in message


def test_parquet_metadata_error_names_parquet(no_pyarrow):
    with pytest.raises(ImportError, match="Reading Parquet metadata requires pyarrow"):
        MetadataScanner.from_parquet_path("missing.parquet")


def test_default_purpose(no_pyarrow):
    with pytest.raises(ImportError, match="This feature requires pyarrow"):
        require_pyarrow()


def test_parquet_metadata_in_fresh_process(tmp_path):
    # ``pyarrow.parquet`` is not an attribute of a bare ``import pyarrow``; the
    # scanner must import the submodule itself rather than rely on a caller.
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    path = tmp_path / "x.parquet"
    pd.DataFrame({"a": [1.0, None, 3.0]}).to_parquet(path)
    code = (
        "from freshdata.execution._metadata import MetadataScanner as M;"
        f"m = M.from_parquet_path({str(path)!r})[0];"
        "print(m.row_count, m.non_null_count)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["3", "2"]
