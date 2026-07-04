"""Content-addressed teacher output cache with a full audit trail.

Key = ``provider + model + prompt_template_sha256 + input_sha256 +
schema_version``. Every entry stores the raw prompt, the raw response, and
provenance (provider, model, date, terms snapshot id) so any training label
can be traced to the exact call that produced it — and so re-runs never pay
for the same call twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import CACHE_DIR, read_json, sha256_text, utc_now_iso, write_json


@dataclass(frozen=True)
class CacheKey:
    provider: str
    model: str
    prompt_template_sha256: str
    input_sha256: str
    schema_version: str

    def digest(self) -> str:
        joined = "|".join((
            self.provider, self.model, self.prompt_template_sha256,
            self.input_sha256, self.schema_version,
        ))
        return sha256_text(joined)


class TeacherCache:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else CACHE_DIR / "teacher"

    def _path(self, key: CacheKey) -> Path:
        digest = key.digest()
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: CacheKey) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        return read_json(path)

    def put(
        self,
        key: CacheKey,
        *,
        prompt: str,
        response: str,
        payloads: list[dict[str, Any]],
        terms_snapshot_id: str,
    ) -> Path:
        entry = {
            "key": {
                "provider": key.provider,
                "model": key.model,
                "prompt_template_sha256": key.prompt_template_sha256,
                "input_sha256": key.input_sha256,
                "schema_version": key.schema_version,
            },
            "prompt": prompt,
            "response": response,
            "payloads": payloads,
            "provider": key.provider,
            "model": key.model,
            "created_at": utc_now_iso(),
            "terms_snapshot_id": terms_snapshot_id,
        }
        return write_json(self._path(key), entry)

    def entries(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        return [read_json(p) for p in sorted(self.root.rglob("*.json"))]
