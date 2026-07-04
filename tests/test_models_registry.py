"""Model registry: status, paths, checksum policy, env override, air-gap flow."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

import freshdata as fd
from freshdata.models import (
    ModelChecksumError,
    ModelConfig,
    ModelNotInstalledError,
    UnknownModelError,
)
from freshdata.models import registry as reg


@pytest.fixture
def model_home(tmp_path, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_DIR", str(tmp_path))
    return tmp_path


def _place(model_home: Path, model_id: str, contents: bytes = b"weights") -> Path:
    cfg = reg.get_config(model_id)
    base = model_home / model_id
    base.mkdir(parents=True)
    for name in cfg.files:
        (base / name).write_bytes(contents)
    return base / cfg.files[0]


def test_fd_models_lazy_export():
    assert fd.models.list_available()
    assert "models" in dir(fd)


def test_list_available_has_required_ids():
    ids = [c.model_id for c in fd.models.list_available()]
    assert ids == sorted(ids)
    assert {"fd-col-encoder-v1", "fd-intent-v1", "calib-v1"} <= set(ids)


def test_model_config_frozen_hashable():
    cfg = reg.get_config("fd-col-encoder-v1")
    assert isinstance(cfg, ModelConfig)
    assert isinstance(hash(cfg), int)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model_id = "nope"  # type: ignore[misc]


def test_unknown_model_id():
    with pytest.raises(UnknownModelError, match="Known models"):
        reg.get_config("fd-unknown-v9")


def test_model_dir_env_override(model_home):
    assert reg.model_dir() == model_home


def test_status_missing_models(model_home):
    status = fd.models.status()
    encoder = status["fd-col-encoder-v1"]
    assert encoder["installed"] is False
    assert encoder["verified"] is None
    assert "not yet published" in encoder["note"]


def test_status_never_creates_directories(model_home):
    fd.models.status()
    assert list(model_home.iterdir()) == []


def test_airgapped_manual_placement_detected(model_home):
    _place(model_home, "fd-col-encoder-v1")
    status = fd.models.status()["fd-col-encoder-v1"]
    assert status["installed"] is True
    assert status["verified"] is False  # no hash pinned yet
    assert "unverified" in status["note"]
    assert fd.models.path("fd-col-encoder-v1").name == "model.onnx"


def test_partial_placement_not_installed(model_home):
    base = model_home / "fd-col-encoder-v1"
    base.mkdir()
    (base / "model.onnx").write_bytes(b"weights")  # tokenizer.json missing
    assert reg.is_installed("fd-col-encoder-v1") is False


def test_path_missing_model_raises(model_home):
    with pytest.raises(ModelNotInstalledError, match="pull"):
        fd.models.path("fd-intent-v1")


def test_pinned_checksum_match_and_mismatch(model_home, monkeypatch):
    payload = b"real-weights"
    primary = _place(model_home, "fd-intent-v1", payload)
    pinned = dataclasses.replace(
        reg.get_config("fd-intent-v1"), sha256=hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setitem(reg.REGISTRY, "fd-intent-v1", pinned)
    assert reg.verify("fd-intent-v1") is True
    assert fd.models.status()["fd-intent-v1"]["verified"] is True

    primary.write_bytes(b"tampered")
    with pytest.raises(ModelChecksumError, match="Refusing to load"):
        reg.verify("fd-intent-v1")
    status = fd.models.status()["fd-intent-v1"]
    assert status["verified"] is False
    assert "mismatch" in status["note"].lower()


def test_min_lib_version_gate(model_home, monkeypatch):
    demanding = dataclasses.replace(reg.get_config("fd-intent-v1"), min_lib_version="99.0.0")
    monkeypatch.setitem(reg.REGISTRY, "fd-intent-v1", demanding)
    with pytest.raises(fd.models.ModelVersionError, match="99.0.0"):
        reg.check_lib_version("fd-intent-v1")
