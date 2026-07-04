"""Artifact packaging: manifests, SHA verification, gating on missing pieces."""

from __future__ import annotations

import json

import pytest
from training.distill import package_artifacts as pa


@pytest.fixture()
def built_inputs(tmp_path, monkeypatch):
    """Minimal build tree satisfying one model id's packaging inputs."""
    export_dir = tmp_path / "export" / "fd-role-head-v1"
    export_dir.mkdir(parents=True)
    (export_dir / "tokenizer.json").write_text("{}")
    (export_dir / "weights.json").write_text("{}")
    (export_dir / "weights.int8.json").write_text("{}")
    metrics_dir = tmp_path / "role_head"
    metrics_dir.mkdir(parents=True)
    metrics_path = metrics_dir / "role_head.metrics.json"
    metrics_path.write_text(json.dumps({"macro_f1": 0.95}))

    monkeypatch.setitem(pa._SPECS["fd-col-encoder-v1"], "files", {
        "tokenizer.json": export_dir / "tokenizer.json",
        "role_head.weights.json": export_dir / "weights.json",
        "role_head.weights.int8.json": export_dir / "weights.int8.json",
    })
    monkeypatch.setitem(pa._SPECS["fd-col-encoder-v1"], "optional_files", {})
    monkeypatch.setitem(pa._SPECS["fd-col-encoder-v1"], "metrics", metrics_path)
    return tmp_path


def test_package_one_writes_manifest_and_card(built_inputs, tmp_path):
    out_root = tmp_path / "artifacts"
    manifest = pa.package_one("fd-col-encoder-v1", out_root=out_root, release=False)
    assert manifest["model_id"] == "fd-col-encoder-v1"
    assert manifest["sha256"]
    assert (out_root / "fd-col-encoder-v1" / "model_card.md").is_file()
    assert (out_root / "fd-col-encoder-v1" / "manifest.json").is_file()


def test_validate_package_passes_for_freshly_packaged(built_inputs, tmp_path):
    out_root = tmp_path / "artifacts"
    pa.package_one("fd-col-encoder-v1", out_root=out_root, release=False)
    assert pa.validate_package("fd-col-encoder-v1", out_root=out_root) == []


def test_validate_package_catches_sha_mismatch(built_inputs, tmp_path):
    out_root = tmp_path / "artifacts"
    pa.package_one("fd-col-encoder-v1", out_root=out_root, release=False)
    (out_root / "fd-col-encoder-v1" / "tokenizer.json").write_text('{"tampered": true}')
    problems = pa.validate_package("fd-col-encoder-v1", out_root=out_root)
    assert any("SHA mismatch" in p for p in problems)


def test_validate_package_catches_missing_card(built_inputs, tmp_path):
    out_root = tmp_path / "artifacts"
    pa.package_one("fd-col-encoder-v1", out_root=out_root, release=False)
    (out_root / "fd-col-encoder-v1" / "model_card.md").unlink()
    problems = pa.validate_package("fd-col-encoder-v1", out_root=out_root)
    assert any("model card" in p for p in problems)


def test_package_one_fails_on_missing_metrics(built_inputs, tmp_path, monkeypatch):
    monkeypatch.setitem(pa._SPECS["fd-col-encoder-v1"], "metrics", tmp_path / "nope.json")
    with pytest.raises(pa.PackagingError, match="missing eval metrics"):
        pa.package_one("fd-col-encoder-v1", out_root=tmp_path / "artifacts", release=False)


def test_package_one_fails_on_missing_source_file(built_inputs, tmp_path, monkeypatch):
    monkeypatch.setitem(pa._SPECS["fd-col-encoder-v1"], "files", {
        "missing.json": tmp_path / "does" / "not" / "exist.json",
    })
    with pytest.raises(pa.PackagingError, match="missing build input"):
        pa.package_one("fd-col-encoder-v1", out_root=tmp_path / "artifacts", release=False)


def test_size_limit_enforced(built_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(pa.PackagingError, match="exceeds limit"):
        pa.package_one("fd-col-encoder-v1", out_root=tmp_path / "artifacts", release=False)


def test_release_requires_onnx_file(built_inputs, tmp_path, monkeypatch):
    monkeypatch.setitem(pa._SPECS["fd-col-encoder-v1"], "optional_files", {
        "model.onnx": tmp_path / "nonexistent.onnx",
    })
    with pytest.raises(pa.PackagingError, match="requires"):
        pa.package_one("fd-col-encoder-v1", out_root=tmp_path / "artifacts", release=True)


def test_wheel_guard_flags_files_inside_src_freshdata(tmp_path):
    fake_wheel_dir = pa.REPO_ROOT / "src" / "freshdata" / "_test_artifact_guard_tmp"
    fake_wheel_dir.mkdir(exist_ok=True)
    try:
        (fake_wheel_dir / "sneaky.onnx").write_bytes(b"x")
        problems = pa._wheel_guard(fake_wheel_dir)
        assert any("inside src/freshdata" in p or "wheel tree" in p for p in problems)
    finally:
        import shutil

        shutil.rmtree(fake_wheel_dir)


def test_committed_dist_artifacts_pass_validation_when_built():
    """If dist/artifacts/ has already been built in this checkout, it must
    validate cleanly (each model dir self-consistent)."""
    if not pa.ARTIFACTS_DIR.is_dir():
        pytest.skip("dist/artifacts/ not built in this environment")
    for model_id in pa._SPECS:
        if (pa.ARTIFACTS_DIR / model_id).is_dir():
            assert pa.validate_package(model_id, out_root=pa.ARTIFACTS_DIR) == []
