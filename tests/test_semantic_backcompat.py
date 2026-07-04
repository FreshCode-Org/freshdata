"""Phase-3 backward compatibility: defaults, configs, report shape, wheel."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig, merge_options


def test_default_semantic_backends_unchanged():
    assert CleanConfig().semantic_backends == ("deterministic",)


def test_clean_default_signature_unchanged(messy):
    out = fd.clean(messy, verbose=False)
    assert isinstance(out, pd.DataFrame)
    out2, report = fd.clean(messy, context=None, verbose=False, return_report=True)
    pd.testing.assert_frame_equal(out, out2)
    assert report.fallback_events == []


def test_config_pickle_roundtrip():
    config = merge_options(
        None, semantic_mode="auto", semantic_backends=("deterministic", "embedding")
    )
    clone = pickle.loads(pickle.dumps(config))
    assert clone == config
    assert clone.semantic_embedding_cache_size == config.semantic_embedding_cache_size


def test_old_style_config_kwargs_still_work():
    """A pre-Phase-3 caller's kwargs construct a valid config (new fields default)."""
    config = merge_options(
        None,
        semantic_mode="review",
        semantic_auto_threshold=0.97,
        semantic_max_distinct_values=100,
    )
    assert config.semantic_backends == ("deterministic",)
    assert config.semantic_budget is None
    assert config.semantic_embedding_cache_size == 65_536


def test_report_json_only_gains_keys(messy):
    _, report = fd.clean(messy, semantic_mode="auto", verbose=False, return_report=True)
    data = report.to_dict()
    # The Phase-2 report surface every consumer may rely on.
    for key in (
        "rows_before",
        "rows_after",
        "cols_before",
        "cols_after",
        "actions",
        "warnings",
        "duration_seconds",
    ):
        assert key in data
    for action in data["actions"]:
        assert {"step", "description", "status"} <= set(action)


def test_semantic_action_metadata_is_json_safe():
    df = pd.DataFrame({"status": ["activ", "active", "pending"], "n": [1, 2, 3]})
    _, report = fd.clean(
        df,
        semantic_mode="auto",
        semantic_context={
            "columns": {"status": {"allowed_values": ["active", "inactive", "pending"]}}
        },
        verbose=False,
        return_report=True,
    )
    semantic = [a for a in report.actions if a.step == "semantic"]
    assert semantic
    for action in semantic:
        assert action.metadata["backend"] == "deterministic"
        json.dumps(action.metadata, default=str)  # must serialize


@pytest.mark.large
def test_wheel_contains_no_model_weights(tmp_path):
    """The wheel ships the calibration JSON but never model weights."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path), str(root)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"wheel build unavailable: {result.stderr[-200:]}")
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    names = zipfile.ZipFile(wheels[0]).namelist()
    assert not [n for n in names if n.endswith((".onnx", ".pt", ".bin", ".safetensors"))]
    assert any(n.endswith("semantic/data/calib_default.json") for n in names)
    assert wheels[0].stat().st_size < 2_000_000  # weights can never sneak in quietly
