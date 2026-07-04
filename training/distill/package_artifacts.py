"""Package trained artifacts into ``dist/artifacts/`` with manifests.

Layout (one directory per model id, outside the wheel by construction)::

    dist/artifacts/
      fd-col-encoder-v1/  model.onnx? tokenizer.json role_head.* manifest.json model_card.md
      fd-intent-v1/       model.onnx? tokenizer.json weights.*   manifest.json model_card.md
      calib-v1/           calib.json calibration.json            manifest.json model_card.md

Validation fails on: SHA mismatch, missing model card, missing license
summary, missing eval metrics, size over the configured limit, or any
artifact file that would land inside the wheel package tree
(``src/freshdata``).

CLI::

    python -m training.distill.package_artifacts [--release]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from ..common import (
    ARTIFACTS_DIR,
    BUILD_DIR,
    MAX_ARTIFACT_BYTES,
    REPO_ROOT,
    TRAINING_ROOT,
    freshdata_version,
    git_commit,
    read_json,
    sha256_file,
    utc_now_iso,
    write_json,
)

MODEL_CARDS = TRAINING_ROOT / "model_cards"
EXPORT_DIR = BUILD_DIR / "export"
CALIBRATION_DIR = BUILD_DIR / "calibration"

#: model_id -> (card file, source files: {artifact name: build path})
_SPECS: dict[str, dict[str, Any]] = {
    "fd-col-encoder-v1": {
        "card": "fd-col-encoder-v1.md",
        "license": "Apache-2.0",
        "training_data_summary": (
            "Synthetic PII-shaped seed corpus + corruptor-derived labels "
            "(training/seed/registry.json); no real PII, no scraped data."
        ),
        "files": {
            "tokenizer.json": EXPORT_DIR / "fd-role-head-v1" / "tokenizer.json",
            "role_head.weights.json": EXPORT_DIR / "fd-role-head-v1" / "weights.json",
            "role_head.weights.int8.json": EXPORT_DIR / "fd-role-head-v1" / "weights.int8.json",
        },
        "optional_files": {
            "model.onnx": EXPORT_DIR / "fd-role-head-v1" / "model.onnx",
            "role_head.onnx": EXPORT_DIR / "fd-role-head-v1" / "model.onnx",
        },
        "metrics": BUILD_DIR / "role_head" / "role_head.metrics.json",
    },
    "fd-intent-v1": {
        "card": "fd-intent-v1.md",
        "license": "Apache-2.0",
        "training_data_summary": (
            "Phase-1 golden context corpus + synthetic/Hinglish paraphrase set "
            "+ corruptor context variants; author-disjoint human-verified eval."
        ),
        "files": {
            "tokenizer.json": EXPORT_DIR / "fd-intent-v1" / "tokenizer.json",
            "weights.json": EXPORT_DIR / "fd-intent-v1" / "weights.json",
            "weights.int8.json": EXPORT_DIR / "fd-intent-v1" / "weights.int8.json",
        },
        "optional_files": {"model.onnx": EXPORT_DIR / "fd-intent-v1" / "model.onnx"},
        "metrics": BUILD_DIR / "intent_head" / "intent_head.metrics.json",
    },
    "calib-v1": {
        "card": "calib-v1.md",
        "license": "Apache-2.0",
        "training_data_summary": (
            "CleanBench T1-T4 fixture runs with corruptor ground truth; "
            "isotonic fit per (backend, issue family)."
        ),
        "files": {
            "calib.json": CALIBRATION_DIR / "calib.json",
            "calibration.json": CALIBRATION_DIR / "calibration.json",
        },
        "optional_files": {},
        "metrics": CALIBRATION_DIR / "calib-v1.metrics.json",
    },
}


class PackagingError(RuntimeError):
    """Artifact packaging failed validation."""


def _quantization(model_id: str, files: dict[str, Path]) -> str:
    if model_id == "calib-v1":
        return "json"
    if any(name.endswith(".onnx") for name in files):
        return "int8"
    return "int8-json"  # dev artifacts: our own int8 weight format, no onnx


def package_one(model_id: str, *, out_root: Path, release: bool) -> dict[str, Any]:
    spec = _SPECS[model_id]
    sources: dict[str, Path] = {}
    for name, path in spec["files"].items():
        if not Path(path).is_file():
            raise PackagingError(f"{model_id}: missing build input {path}")
        sources[name] = Path(path)
    for name, path in spec["optional_files"].items():
        if Path(path).is_file() and name not in sources:
            sources[name] = Path(path)
        elif release and name == "model.onnx":
            raise PackagingError(f"{model_id}: release packaging requires {path}")

    metrics_path = Path(spec["metrics"])
    if not metrics_path.is_file():
        raise PackagingError(f"{model_id}: missing eval metrics {metrics_path}")
    metrics = read_json(metrics_path)

    card_source = MODEL_CARDS / spec["card"]
    if not card_source.is_file():
        raise PackagingError(f"{model_id}: missing model card {card_source}")

    target = out_root / model_id
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    copied: list[dict[str, Any]] = []
    total = 0
    for name, source in sorted(sources.items()):
        destination = target / name
        shutil.copy(source, destination)
        size = destination.stat().st_size
        total += size
        copied.append({"name": name, "sha256": sha256_file(destination), "size_bytes": size})
    shutil.copy(card_source, target / "model_card.md")
    shutil.copy(metrics_path, target / "eval_metrics.json")

    if total > MAX_ARTIFACT_BYTES:
        raise PackagingError(
            f"{model_id}: artifact size {total} exceeds limit {MAX_ARTIFACT_BYTES}")

    primary = copied[0]
    manifest = {
        "model_id": model_id,
        "version": model_id.rsplit("-v", 1)[-1],
        "freshdata_min_version": freshdata_version(),
        "sha256": primary["sha256"],
        "size_bytes": total,
        "license": spec["license"],
        "training_data_summary": spec["training_data_summary"],
        "eval_summary": {k: v for k, v in metrics.items()
                         if isinstance(v, (int, float, str))},
        "quantization": _quantization(model_id, sources),
        "files": copied,
        "created_at": utc_now_iso(),
        "git_commit": git_commit(),
        "release": release,
    }
    write_json(target / "manifest.json", manifest)
    return manifest


def validate_package(model_id: str, *, out_root: Path) -> list[str]:
    """Independent re-validation of a packaged artifact directory."""
    problems: list[str] = []
    target = out_root / model_id
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        return [f"{model_id}: manifest.json missing"]
    manifest = read_json(manifest_path)
    for entry in manifest.get("files", []):
        path = target / entry["name"]
        if not path.is_file():
            problems.append(f"{model_id}: file {entry['name']} listed but missing")
            continue
        if sha256_file(path) != entry["sha256"]:
            problems.append(f"{model_id}: SHA mismatch for {entry['name']}")
    if not (target / "model_card.md").is_file():
        problems.append(f"{model_id}: model card missing")
    if not manifest.get("license"):
        problems.append(f"{model_id}: license summary missing")
    if not manifest.get("eval_summary"):
        problems.append(f"{model_id}: eval metrics missing")
    if int(manifest.get("size_bytes", 0)) > MAX_ARTIFACT_BYTES:
        problems.append(f"{model_id}: size over limit")
    problems.extend(_wheel_guard(target))
    return problems


def _wheel_guard(target: Path) -> list[str]:
    """No packaged artifact may live inside the wheel package tree."""
    problems = []
    wheel_root = (REPO_ROOT / "src" / "freshdata").resolve()
    for path in target.rglob("*"):
        try:
            path.resolve().relative_to(wheel_root)
        except ValueError:
            continue
        problems.append(f"artifact file {path} is inside src/freshdata (would ship in wheel)")
    stray = list(wheel_root.rglob("*.onnx"))
    problems.extend(f"model weights inside wheel tree: {p}" for p in stray)
    return problems


def package_all(*, out_root: Path | str = ARTIFACTS_DIR, release: bool = False) -> dict[str, Any]:
    root = Path(out_root)
    manifests = {}
    for model_id in _SPECS:
        manifests[model_id] = package_one(model_id, out_root=root, release=release)
        problems = validate_package(model_id, out_root=root)
        if problems:
            raise PackagingError("; ".join(problems))
    write_json(root / "artifacts_index.json", {
        "created_at": utc_now_iso(),
        "git_commit": git_commit(),
        "artifacts": {mid: m["sha256"] for mid, m in manifests.items()},
    })
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.distill.package_artifacts")
    parser.add_argument("--out", default=str(ARTIFACTS_DIR))
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifests = package_all(out_root=args.out, release=args.release)
    except PackagingError as exc:
        print(f"PACKAGING FAIL: {exc}", file=sys.stderr)
        return 1
    for model_id, manifest in manifests.items():
        print(f"{model_id}: sha256={manifest['sha256'][:16]}… "
              f"size={manifest['size_bytes']} files={len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
