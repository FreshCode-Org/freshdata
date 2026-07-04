"""Shared plumbing for the training pipeline: paths, hashing, stable JSON.

Everything downstream (datasets, teacher cache, artifact manifests) needs the
same three primitives — a stable canonical JSON form, sha256 helpers, and the
build/output directory layout — so they live here once.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Repository root (training/ is a top-level package next to src/).
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_ROOT = REPO_ROOT / "training"
#: Intermediate outputs (never committed, never packaged).
BUILD_DIR = TRAINING_ROOT / "build"
#: Teacher prompt/response cache (audit trail; content-addressed).
CACHE_DIR = TRAINING_ROOT / "cache"
#: Final packaged artifacts (outside the wheel by construction).
ARTIFACTS_DIR = REPO_ROOT / "dist" / "artifacts"

#: Hard ceiling for any single packaged model artifact directory.
MAX_ARTIFACT_BYTES = 60_000_000


def stable_json(data: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace drift, utf-8 safe."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data: Any) -> str:
    return sha256_text(stable_json(data))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> str:
    """Current commit hash, or "unknown" outside a git checkout (e.g. sdist)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False,
        )
    except OSError:  # pragma: no cover - git missing entirely
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


def freshdata_version() -> str:
    import freshdata  # noqa: PLC0415 - lazy so `training` imports stay cheap

    return str(freshdata.__version__)


def write_json(path: Path | str, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(stable_json(row) + "\n")
    return path


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
