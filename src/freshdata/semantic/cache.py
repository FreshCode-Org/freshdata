"""In-process LRU cache for value/column embeddings.

Keys are ``(model_id, model_sha256, sha1(normalized_text))`` so a model swap
can never serve stale vectors. Only embeddings are cached — never decisions,
rows, or frames — and nothing is written to disk, so no raw values can leak
into persistent storage. Set ``semantic_embedding_cache_size=0`` to disable.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np

from ..models.runtime import LocalEncoder

_DEFAULT_CAPACITY = 65_536


def _text_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()  # noqa: S324 - cache key only


class EmbeddingCache:
    """LRU wrapper around a :class:`LocalEncoder` for repeated distinct values."""

    def __init__(self, encoder: LocalEncoder, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._encoder = encoder
        self._capacity = max(0, int(capacity))
        self._entries: OrderedDict[tuple[str, str, str], np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.model_calls = 0

    @property
    def encoder(self) -> LocalEncoder:
        return self._encoder

    def _key(self, text: str) -> tuple[str, str, str]:
        return (self._encoder.model_id, self._encoder.model_sha256, _text_key(text))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode ``texts``, serving repeats from the cache in one batch call."""
        results: list[np.ndarray | None] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            cached = self._entries.get(self._key(text)) if self._capacity else None
            if cached is not None:
                self._entries.move_to_end(self._key(text))
                self.hits += 1
                results[i] = cached
            else:
                self.misses += 1
                missing.append(i)
        if missing:
            self.model_calls += 1
            fresh = self._encoder.encode_texts([texts[i] for i in missing])
            for row, i in enumerate(missing):
                vector = np.asarray(fresh[row], dtype=np.float32)
                results[i] = vector
                if self._capacity:
                    self._entries[self._key(texts[i])] = vector
                    while len(self._entries) > self._capacity:
                        self._entries.popitem(last=False)
        if not results:
            return np.empty((0, getattr(self._encoder, "dim", 0)), dtype=np.float32)
        return np.vstack([r for r in results if r is not None])

    def __len__(self) -> int:
        return len(self._entries)
