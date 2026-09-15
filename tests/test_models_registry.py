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


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_status_false_for_mismatched_secondary_file(model_home, monkeypatch):
    """#346: status() reports verified=True only when every file matches."""
    base = model_home / "fd-col-encoder-v1"
    base.mkdir()
    (base / "model.onnx").write_bytes(b"GENUINE-MODEL")
    (base / "tokenizer.json").write_bytes(b'{"modified": true}')
    pinned = dataclasses.replace(
        reg.get_config("fd-col-encoder-v1"),
        file_sha256=(
            ("model.onnx", _sha(b"GENUINE-MODEL")),
            ("tokenizer.json", _sha(b'{"vocab": []}')),
        ),
    )
    monkeypatch.setitem(reg.REGISTRY, "fd-col-encoder-v1", pinned)
    with pytest.raises(ModelChecksumError, match="tokenizer.json"):
        reg.verify("fd-col-encoder-v1")
    status = fd.models.status()["fd-col-encoder-v1"]
    assert status["installed"] is True
    assert status["verified"] is False
    assert "tokenizer.json" in status["note"]

    (base / "tokenizer.json").write_bytes(b'{"vocab": []}')
    assert reg.verify("fd-col-encoder-v1") is True
    assert fd.models.status()["fd-col-encoder-v1"]["verified"] is True


def test_verify_partially_pinned_model_raises(model_home, monkeypatch):
    _place(model_home, "fd-col-encoder-v1", b"weights")
    pinned = dataclasses.replace(reg.get_config("fd-col-encoder-v1"), sha256=_sha(b"weights"))
    monkeypatch.setitem(reg.REGISTRY, "fd-col-encoder-v1", pinned)
    with pytest.raises(ModelChecksumError, match="'tokenizer.json'"):
        reg.verify("fd-col-encoder-v1")
    assert fd.models.status()["fd-col-encoder-v1"]["verified"] is False


def test_pinned_checksums_combines_primary_and_file_pins():
    cfg = dataclasses.replace(
        reg.get_config("fd-col-encoder-v1"),
        sha256="a" * 64,
        file_sha256=(("model.onnx", "a" * 64), ("tokenizer.json", "b" * 64)),
    )
    assert reg.pinned_checksums(cfg) == {"model.onnx": "a" * 64, "tokenizer.json": "b" * 64}
    unpinned = reg.get_config("fd-col-encoder-v1")
    assert reg.pinned_checksums(dataclasses.replace(unpinned, sha256=None)) == {}


def test_pinned_checksums_rejects_unknown_file():
    cfg = dataclasses.replace(
        reg.get_config("fd-intent-v1"), file_sha256=(("weights.bin", "a" * 64),)
    )
    with pytest.raises(ModelChecksumError, match="weights.bin"):
        reg.pinned_checksums(cfg)


def test_pinned_checksums_rejects_conflicting_primary_pin():
    cfg = dataclasses.replace(
        reg.get_config("fd-intent-v1"),
        sha256="a" * 64,
        file_sha256=(("model.onnx", "b" * 64),),
    )
    with pytest.raises(ModelChecksumError, match="conflicting"):
        reg.pinned_checksums(cfg)


@pytest.mark.parametrize("model_id", sorted(reg.REGISTRY))
def test_registry_pins_are_empty_or_cover_every_file(model_id):
    cfg = reg.REGISTRY[model_id]
    pins = reg.pinned_checksums(cfg)
    assert pins == {} or set(pins) == set(cfg.files)


def test_min_lib_version_gate(model_home, monkeypatch):
    demanding = dataclasses.replace(reg.get_config("fd-intent-v1"), min_lib_version="99.0.0")
    monkeypatch.setitem(reg.REGISTRY, "fd-intent-v1", demanding)
    with pytest.raises(fd.models.ModelVersionError, match="99.0.0"):
        reg.check_lib_version("fd-intent-v1")
