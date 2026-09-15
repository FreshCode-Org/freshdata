"""Explicit pull(): unpublished state, mocked downloads, checksum refusal."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
from freshdata.models import ModelChecksumError, ModelNotPublishedError
from freshdata.models import download as dl
from freshdata.models import registry as reg


@pytest.fixture
def model_home(tmp_path, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_DIR", str(tmp_path))
    return tmp_path


def test_pull_unpublished_raises_actionable_error(model_home):
    with pytest.raises(ModelNotPublishedError) as exc:
        fd.models.pull("fd-col-encoder-v1")
    message = str(exc.value)
    assert "FRESHDATA_MODEL_URL_BASE" in message
    assert "manually" in message
    assert list(model_home.iterdir()) == []  # nothing half-created


def test_pull_downloads_all_files(model_home, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/models")
    fetched: list[str] = []

    def fake_fetch(url: str, dest: Path) -> None:
        fetched.append(url)
        dest.write_bytes(b"payload:" + url.encode())

    monkeypatch.setattr(dl, "_fetch", fake_fetch)
    primary = fd.models.pull("fd-col-encoder-v1")
    assert primary.name == "model.onnx"
    assert primary.is_file()
    assert (model_home / "fd-col-encoder-v1" / "tokenizer.json").is_file()
    assert fetched == [
        "https://example.test/models/fd-col-encoder-v1/model.onnx",
        "https://example.test/models/fd-col-encoder-v1/tokenizer.json",
    ]
    assert not list(primary.parent.glob("*.part"))

    # Second pull is a no-op unless forced.
    fetched.clear()
    fd.models.pull("fd-col-encoder-v1")
    assert fetched == []
    fd.models.pull("fd-col-encoder-v1", force=True)
    assert len(fetched) == 2


def test_pull_checksum_mismatch_discards_download(model_home, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/models")
    pinned = dataclasses.replace(
        reg.get_config("fd-intent-v1"), sha256=hashlib.sha256(b"expected").hexdigest()
    )
    monkeypatch.setitem(reg.REGISTRY, "fd-intent-v1", pinned)
    monkeypatch.setattr(dl, "_fetch", lambda url, dest: dest.write_bytes(b"evil"))
    with pytest.raises(ModelChecksumError, match="discarded"):
        fd.models.pull("fd-intent-v1")
    target = model_home / "fd-intent-v1"
    assert not (target / "model.onnx").exists()
    assert not list(target.glob("*.part"))


def test_pull_checksum_match_keeps_download(model_home, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/models")
    payload = b"expected"
    pinned = dataclasses.replace(
        reg.get_config("fd-intent-v1"), sha256=hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setitem(reg.REGISTRY, "fd-intent-v1", pinned)
    monkeypatch.setattr(dl, "_fetch", lambda url, dest: dest.write_bytes(payload))
    primary = fd.models.pull("fd-intent-v1")
    assert primary.read_bytes() == payload
    assert reg.verify("fd-intent-v1") is True


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _no_fetch(url: str, dest: Path) -> None:  # pragma: no cover - must not run
    raise AssertionError(f"unexpected download of {url}")


_ENCODER_FILES = {"model.onnx": b"GENUINE-MODEL", "tokenizer.json": b'{"vocab": []}'}
_ENCODER_PINS = tuple((name, _sha(data)) for name, data in _ENCODER_FILES.items())


def _replace_encoder(monkeypatch, **changes: object) -> None:
    cfg = dataclasses.replace(reg.get_config("fd-col-encoder-v1"), **changes)
    monkeypatch.setitem(reg.REGISTRY, "fd-col-encoder-v1", cfg)


def test_pull_refuses_pinned_secondary_file_mismatch(model_home, monkeypatch):
    """#346: every downloaded file is checked, not only the primary."""
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/models")
    _replace_encoder(monkeypatch, file_sha256=_ENCODER_PINS)
    served = {"model.onnx": _ENCODER_FILES["model.onnx"], "tokenizer.json": b'{"modified": true}'}
    monkeypatch.setattr(
        dl, "_fetch", lambda url, dest: dest.write_bytes(served[url.rsplit("/", 1)[-1]])
    )
    with pytest.raises(ModelChecksumError, match="tokenizer.json"):
        fd.models.pull("fd-col-encoder-v1")
    target = model_home / "fd-col-encoder-v1"
    assert not (target / "tokenizer.json").exists()
    assert reg.is_installed("fd-col-encoder-v1") is False
    assert not list(target.glob("*.part"))


def test_pull_all_pinned_files_matching_installs(model_home, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/models")
    _replace_encoder(
        monkeypatch,
        sha256=_sha(_ENCODER_FILES["model.onnx"]),
        file_sha256=(("tokenizer.json", _sha(_ENCODER_FILES["tokenizer.json"])),),
    )
    monkeypatch.setattr(
        dl, "_fetch", lambda url, dest: dest.write_bytes(_ENCODER_FILES[url.rsplit("/", 1)[-1]])
    )
    primary = fd.models.pull("fd-col-encoder-v1")
    assert primary.read_bytes() == _ENCODER_FILES["model.onnx"]
    assert reg.verify("fd-col-encoder-v1") is True
    assert fd.models.status()["fd-col-encoder-v1"]["verified"] is True


def test_pull_partially_pinned_model_raises(model_home, monkeypatch):
    """#346 repro: only the primary is pinned, so tokenizer.json cannot be checked."""
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/models")
    _replace_encoder(monkeypatch, sha256=_sha(_ENCODER_FILES["model.onnx"]))
    monkeypatch.setattr(dl, "_fetch", _no_fetch)
    with pytest.raises(ModelChecksumError, match="'tokenizer.json'"):
        fd.models.pull("fd-col-encoder-v1")
    assert list(model_home.iterdir()) == []


def test_pull_early_return_with_corrupt_primary_raises(model_home, monkeypatch):
    """#346 repro: already-present files are verified before pull reports success."""
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/models")
    monkeypatch.setitem(
        reg.REGISTRY,
        "fd-intent-v1",
        dataclasses.replace(reg.get_config("fd-intent-v1"), sha256=_sha(b"GOOD")),
    )
    target = model_home / "fd-intent-v1"
    target.mkdir()
    (target / "model.onnx").write_bytes(b"CORRUPT")
    monkeypatch.setattr(dl, "_fetch", _no_fetch)
    with pytest.raises(ModelChecksumError, match="force=True"):
        fd.models.pull("fd-intent-v1")
    assert (target / "model.onnx").read_bytes() == b"CORRUPT"  # left in place


def test_pull_early_return_with_corrupt_secondary_raises(model_home, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/models")
    _replace_encoder(monkeypatch, file_sha256=_ENCODER_PINS)
    target = model_home / "fd-col-encoder-v1"
    target.mkdir()
    (target / "model.onnx").write_bytes(_ENCODER_FILES["model.onnx"])
    (target / "tokenizer.json").write_bytes(b'{"modified": true}')
    monkeypatch.setattr(dl, "_fetch", _no_fetch)
    with pytest.raises(ModelChecksumError, match="tokenizer.json"):
        fd.models.pull("fd-col-encoder-v1")
    assert (target / "tokenizer.json").read_bytes() == b'{"modified": true}'


def test_pull_early_return_with_matching_pins_skips_fetch(model_home, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/models")
    _replace_encoder(monkeypatch, file_sha256=_ENCODER_PINS)
    target = model_home / "fd-col-encoder-v1"
    target.mkdir()
    for name, data in _ENCODER_FILES.items():
        (target / name).write_bytes(data)
    monkeypatch.setattr(dl, "_fetch", _no_fetch)
    assert fd.models.pull("fd-col-encoder-v1") == target / "model.onnx"


def test_pull_early_return_unpinned_is_unchanged(model_home, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/models")
    target = model_home / "fd-col-encoder-v1"
    target.mkdir()
    (target / "model.onnx").write_bytes(b"anything")
    (target / "tokenizer.json").write_bytes(b"anything")
    monkeypatch.setattr(dl, "_fetch", _no_fetch)
    assert fd.models.pull("fd-col-encoder-v1") == target / "model.onnx"
    assert (target / "model.onnx").read_bytes() == b"anything"


def test_clean_never_downloads(model_home, monkeypatch):
    """fd.clean must not touch the network even when embedding is requested."""

    def explode(url: str, dest: Path) -> None:  # pragma: no cover - must not run
        raise AssertionError("fd.clean attempted a model download")

    monkeypatch.setattr(dl, "_fetch", explode)
    df = pd.DataFrame({"status": ["activ", "active", "pendng"], "n": [1, 2, 3]})
    out, report = fd.clean(
        df,
        semantic_mode="auto",
        semantic_backends=("deterministic", "memory", "embedding"),
        return_report=True,
    )
    assert out is not None
    assert report is not None
