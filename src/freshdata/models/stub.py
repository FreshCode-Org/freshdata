"""Deterministic stub encoder for tests and model-free CI.

Produces stable pseudo-embeddings from sha256 digests of character trigrams —
no model files, no randomness, no ``hash()`` — so vectors are identical across
runs, platforms, and Python versions, while strings that share most of their
character n-grams (``"activ"`` / ``"active"``) still land close in cosine
space. It is a crude but honest similarity signal, good enough to exercise the
full embedding path end-to-end without real model weights.

Enable process-wide (e.g. for CLI end-to-end tests) with
``FRESHDATA_STUB_ENCODER=1``. Testing infrastructure only — not a public API.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Sequence

import numpy as np

_DIM = 384
_DIGEST_BYTES = 32
_REPEATS = _DIM // _DIGEST_BYTES  # 12 digests of 32 bytes -> 384 values


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()


def _token_vector(token: str) -> np.ndarray:
    seed = token.encode("utf-8")
    raw = bytearray()
    for salt in range(_REPEATS):
        raw += hashlib.sha256(bytes([salt]) + seed).digest()
    values = np.frombuffer(bytes(raw), dtype=np.uint8).astype(np.float32)
    return values / 127.5 - 1.0  # bytes -> [-1, 1]


class StubEncoder:
    """Hash-based :class:`~freshdata.models.runtime.LocalEncoder` stand-in."""

    model_id = "fd-col-encoder-v1"
    model_sha256 = "stub"
    dim = _DIM

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = np.empty((len(texts), _DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            padded = f"^{_normalize(text)}$"
            grams = (
                [padded[j : j + 3] for j in range(len(padded) - 2)]
                if len(padded) >= 3
                else [padded]
            )
            vector = np.zeros(_DIM, dtype=np.float32)
            for gram in grams:
                vector += _token_vector(gram)
            norm = float(np.linalg.norm(vector))
            out[i] = vector / norm if norm else vector
        return out
