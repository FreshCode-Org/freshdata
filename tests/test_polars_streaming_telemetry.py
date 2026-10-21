# tests/test_polars_streaming_telemetry.py
from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.execution import EngineConfig, StreamingUnsupportedError


def test_polars_streaming_telemetry_reports_non_streaming_stages_honestly() -> None:
    """Fixes #53, #124, #191: Default drop_duplicates with maintain_order=True cannot stream in Polars; report it."""
    df = pd.DataFrame({
        "a": pd.Series([" x ", " x ", "y", "z"], dtype="string"),
        "b": pd.Series([1, 1, 2, 3], dtype="Int64"),
    })
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False, drop_duplicates=True, duplicate_keep="first")
    eng_cfg = EngineConfig(engine="polars", streaming=True, strict_streaming=False)

    cleaned, report = fd.clean(df, config=cfg, engine="polars", engine_config=eng_cfg, return_report=True)

    assert report.streaming_requested is True
    assert report.streaming_executed_fully is False
    assert "drop_duplicates:maintain_order" in report.non_streaming_stages
    assert report.row_order_deterministic is True
    assert len(cleaned) == 3


def test_polars_strict_streaming_raises_on_non_streamable_stage() -> None:
    """When strict_streaming=True, any stage incompatible with Polars streaming must raise StreamingUnsupportedError."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False, drop_duplicates=True, duplicate_keep="first")
    eng_cfg = EngineConfig(engine="polars", streaming=True, strict_streaming=True)

    with pytest.raises(StreamingUnsupportedError, match="drop_duplicates:maintain_order"):
        fd.clean(df, config=cfg, engine="polars", engine_config=eng_cfg)


def test_polars_strict_streaming_succeeds_when_all_stages_streamable() -> None:
    """When drop_duplicates=False (or streaming_dedup_unordered=True), Polars executes fully in streaming mode."""
    df = pd.DataFrame({
        "a": pd.Series([" alpha ", "beta ", "N/A"], dtype="string"),
        "b": pd.Series([10, 20, 30], dtype="Int64"),
    })
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False, drop_duplicates=False, impute_missing=False)
    eng_cfg = EngineConfig(engine="polars", streaming=True, strict_streaming=True)

    cleaned, report = fd.clean(df, config=cfg, engine="polars", engine_config=eng_cfg, return_report=True)

    assert report.streaming_requested is True
    assert report.streaming_executed_fully is True
    assert report.non_streaming_stages == []
    assert report.row_order_deterministic is True
    assert cleaned["a"].iloc[0] == "alpha"
    assert pd.isna(cleaned["a"].iloc[2])
