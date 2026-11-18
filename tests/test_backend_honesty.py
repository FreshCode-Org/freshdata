import polars as pl
import pandas as pd
import pytest
from freshdata._core.backends import resolve_backend
from freshdata._core.models import CleanConfig, ActionRecord, CleanPlan

def test_lazyframe_preserved_without_collect(monkeypatch):
    called = []
    original_collect = pl.LazyFrame.collect
    def guarded_collect(self, *args, **kwargs):
        called.append(True)
        return original_collect(self, *args, **kwargs)
    monkeypatch.setattr(pl.LazyFrame, "collect", guarded_collect)

    lazy_df = pl.LazyFrame({"a": [" hello ", "world"], "b": [1, 2]})
    backend = resolve_backend(lazy_df)
    plan = CleanPlan(actions=(
        ActionRecord("strings", "a", "strip_strings", {}),
    ))
    res = backend.apply_plan(lazy_df, plan, CleanConfig())
    assert isinstance(res, pl.LazyFrame)
    assert len(called) == 0, "LazyFrame must not be collected eagerly during apply_plan"

def test_to_pandas_never_called(monkeypatch):
    called = []
    def guarded_to_pandas(self, *args, **kwargs):
        called.append(True)
        raise RuntimeError("Forbidden to_pandas call!")
    monkeypatch.setattr(pl.DataFrame, "to_pandas", guarded_to_pandas)

    pl_df = pl.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    backend = resolve_backend(pl_df)
    report = backend.profile(pl_df)
    assert report.row_count == 3
    assert len(called) == 0

def test_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported frame type"):
        resolve_backend({"a": [1, 2, 3]})

def test_import_and_clean_without_polars():
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys; sys.modules['polars'] = None\n"
        "import pandas as pd, freshdata\n"
        "out = freshdata.clean(pd.DataFrame({'A b': [1.0, None, 3.0]}))\n"
        "assert out['a_b'].tolist() == [1.0, 2.0, 3.0]\n"
    )
    src = Path(__file__).resolve().parents[1] / "src"
    subprocess.run([sys.executable, "-c", code], check=True, env={"PYTHONPATH": str(src)})

def test_polars_profile_counts_match_pandas():
    data = {"a": [1, 2, None, 2], "b": ["x", None, None, "y"]}
    pd_rep = resolve_backend(pd.DataFrame(data)).profile(pd.DataFrame(data))
    pl_rep = resolve_backend(pl.DataFrame(data)).profile(pl.DataFrame(data))
    for col in data:
        p, q = pd_rep.columns[col], pl_rep.columns[col]
        assert (q.null_count, q.unique_count, q.null_ratio) == (p.null_count, p.unique_count, p.null_ratio)
