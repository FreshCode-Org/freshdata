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
