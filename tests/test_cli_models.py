"""CLI: freshdata models status|pull and clean --semantic-backends."""

from __future__ import annotations

import dataclasses
import hashlib

import pandas as pd
import pytest

from freshdata.enterprise.cli import main
from freshdata.models import download as dl
from freshdata.models import registry as reg


@pytest.fixture
def model_home(tmp_path, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("FRESHDATA_STUB_ENCODER", raising=False)
    return tmp_path


def test_models_status_without_semantic_extra(model_home, capsys):
    assert main(["models", "status"]) == 0
    out = capsys.readouterr().out
    assert "fd-col-encoder-v1" in out
    assert "calib-v1" in out
    assert "missing" in out
    assert "model directory:" in out


def test_models_pull_unpublished_errors_cleanly(model_home, capsys):
    assert main(["models", "pull", "fd-col-encoder-v1"]) == 2
    out = capsys.readouterr().out
    assert "FRESHDATA_MODEL_URL_BASE" in out


def test_models_pull_unknown_model(model_home, capsys):
    assert main(["models", "pull", "fd-nope-v9"]) == 2
    assert "Known models" in capsys.readouterr().out


def test_models_pull_downloads_with_mocked_fetch(model_home, monkeypatch, capsys):
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/m")
    monkeypatch.setattr(dl, "_fetch", lambda url, dest: dest.write_bytes(b"payload"))
    assert main(["models", "pull", "fd-col-encoder-v1"]) == 0
    assert "pulled fd-col-encoder-v1" in capsys.readouterr().out
    status_out_code = main(["models", "status"])
    assert status_out_code == 0
    assert "installed" in capsys.readouterr().out


def test_models_pull_existing_mismatched_files_exits_2(model_home, monkeypatch, capsys):
    """#346: pull verifies already-installed files instead of reporting success."""
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/m")
    pinned = dataclasses.replace(
        reg.get_config("fd-intent-v1"), sha256=hashlib.sha256(b"GOOD").hexdigest()
    )
    monkeypatch.setitem(reg.REGISTRY, "fd-intent-v1", pinned)
    target = model_home / "models" / "fd-intent-v1"
    target.mkdir(parents=True)
    (target / "model.onnx").write_bytes(b"CORRUPT")

    def no_fetch(url, dest):  # pragma: no cover - must not run
        raise AssertionError("unexpected download")

    monkeypatch.setattr(dl, "_fetch", no_fetch)
    assert main(["models", "pull", "fd-intent-v1"]) == 2
    out = capsys.readouterr().out
    assert "Checksum mismatch" in out
    assert "pulled" not in out


def test_clean_with_embedding_missing_model_prints_skip(model_home, tmp_path, capsys):
    df = pd.DataFrame(
        {
            "status": ["activ", "inactive", "pendng", "active"],
            "amount": [1, 2, 3, 4],
        }
    )
    source = tmp_path / "in.csv"
    output = tmp_path / "out.csv"
    df.to_csv(source, index=False)
    rules = tmp_path / "rules.txt"
    rules.write_text("Allowed status values are active, inactive, pending.\n")
    code = main(
        [
            "clean",
            str(source),
            "-o",
            str(output),
            "--context-file",
            str(rules),
            "--semantic-mode",
            "auto",
            "--semantic-backends",
            "deterministic,memory,embedding",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert output.exists()
    assert "semantic backend 'embedding' skipped" in out
    assert "note: semantic backend 'memory' skipped" in out


def test_plan_accepts_semantic_backends(model_home, tmp_path, capsys):
    df = pd.DataFrame({"status": ["activ", "active", "pending"], "n": [1, 2, 3]})
    source = tmp_path / "in.csv"
    df.to_csv(source, index=False)
    rules = tmp_path / "rules.txt"
    rules.write_text("Allowed status values are active, inactive, pending.\n")
    plan_path = tmp_path / "plan.json"
    code = main(
        [
            "plan",
            str(source),
            "--context-file",
            str(rules),
            "--semantic-backends",
            "deterministic,embedding",
            "--out",
            str(plan_path),
        ]
    )
    assert code == 0
    assert plan_path.exists()
