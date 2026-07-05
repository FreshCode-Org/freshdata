"""Local encoder runtime abstraction over ONNX Runtime.

The rest of the package talks to a tiny :class:`LocalEncoder` protocol —
``encode_texts(texts) -> np.ndarray`` — and never sees ONNX session details.
The real encoder loads lazily once per process, batches inputs, runs CPU-only,
and fails safely (``availability`` reports why) when the ``[semantic]`` extra
or the model files are missing. This module never downloads anything.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from ._lazy import has_semantic_extra, require_onnxruntime, require_tokenizers
from .registry import COL_ENCODER_ID, get_config, is_installed, model_dir, verify
from .types import ModelError

_STUB_ENV = "FRESHDATA_STUB_ENCODER"
_BATCH_SIZE = 256


@runtime_checkable
class LocalEncoder(Protocol):
    """Minimal encoder interface consumed by the embedding backend."""

    model_id: str
    model_sha256: str
    dim: int

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def column_text(column_name: str, values: Sequence[object]) -> str:
    """Render the canonical encoder input for a column and sample values."""
    rendered = "; ".join(str(v) for v in values)
    return f"{column_name} | {rendered}"


class OnnxEncoder:
    """ONNX Runtime implementation of :class:`LocalEncoder` (CPU, int8)."""

    def __init__(self, model_id: str) -> None:
        cfg = get_config(model_id)
        self.model_id = model_id
        self.model_sha256 = cfg.sha256 or "unverified"
        self.dim = 384
        self._session: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    def _load(self) -> None:  # pragma: no cover - requires real model files
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            ort = require_onnxruntime()
            tokenizers = require_tokenizers()
            verify(self.model_id)  # refuses on pinned-checksum mismatch
            base = model_dir() / self.model_id
            options = ort.SessionOptions()
            options.intra_op_num_threads = min(4, os.cpu_count() or 1)
            self._session = ort.InferenceSession(
                str(base / "model.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = tokenizers.Tokenizer.from_file(str(base / "tokenizer.json"))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        self._load()
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = [str(t) for t in texts[start : start + _BATCH_SIZE]]
            encodings = self._tokenizer.encode_batch(batch)
            max_len = max(len(e.ids) for e in encodings)
            input_ids = np.zeros((len(batch), max_len), dtype=np.int64)
            attention = np.zeros((len(batch), max_len), dtype=np.int64)
            for i, enc in enumerate(encodings):
                input_ids[i, : len(enc.ids)] = enc.ids
                attention[i, : len(enc.ids)] = enc.attention_mask
            outputs = self._session.run(
                None, {"input_ids": input_ids, "attention_mask": attention}
            )
            hidden = outputs[0]  # (batch, seq, dim)
            mask = attention[:, :, None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            chunks.append((pooled / np.maximum(norms, 1e-12)).astype(np.float32))
        if not chunks:
            return np.empty((0, self.dim), dtype=np.float32)
        stacked = np.vstack(chunks)
        self.dim = int(stacked.shape[1])
        return stacked


_ENCODER_FACTORY: Callable[[str], LocalEncoder] | None = None
_encoders: dict[str, LocalEncoder] = {}
_encoders_lock = threading.Lock()


def set_encoder_factory(factory: Callable[[str], LocalEncoder] | None) -> None:
    """Install (or clear, with ``None``) a process-wide encoder factory.

    Testing seam: lets tests inject deterministic encoders without model files
    or the ``[semantic]`` extra. Clears the encoder cache on change.
    """
    global _ENCODER_FACTORY
    with _encoders_lock:
        _ENCODER_FACTORY = factory
        _encoders.clear()


def _stub_enabled() -> bool:
    return os.environ.get(_STUB_ENV, "") == "1"


def availability(model_id: str = COL_ENCODER_ID) -> tuple[bool, str]:
    """Report whether an encoder for ``model_id`` can be produced, and why not.

    Never raises and never loads the model; used by backends to self-disable
    with an auditable reason.
    """
    if _ENCODER_FACTORY is not None or _stub_enabled():
        return True, ""
    if not has_semantic_extra():
        return False, (
            "optional dependency missing: install 'freshdata-cleaner[semantic]' "
            "to enable the embedding backend"
        )
    if not is_installed(model_id):
        return False, (
            f"model {model_id!r} is not installed; run fd.models.pull({model_id!r}) "
            "explicitly (models are never downloaded automatically) or place the "
            f"files under {model_dir() / model_id}"
        )
    return True, ""


def get_encoder(model_id: str = COL_ENCODER_ID) -> LocalEncoder:
    """Return the lazily-created process-wide encoder for ``model_id``.

    Resolution order: injected factory (tests) -> ``FRESHDATA_STUB_ENCODER=1``
    stub -> real ONNX encoder. Raises :class:`ModelError` when unavailable —
    call :func:`availability` first to degrade gracefully instead.
    """
    with _encoders_lock:
        cached = _encoders.get(model_id)
        if cached is not None:
            return cached
        encoder: LocalEncoder
        if _ENCODER_FACTORY is not None:
            encoder = _ENCODER_FACTORY(model_id)
        elif _stub_enabled():
            from .stub import StubEncoder  # noqa: PLC0415 - keep numpy-light default import

            encoder = StubEncoder()
        else:
            ok, reason = availability(model_id)
            if not ok:
                raise ModelError(reason)
            encoder = OnnxEncoder(model_id)
        _encoders[model_id] = encoder
        return encoder


def reset_encoders() -> None:
    """Clear cached encoders (tests / after model files change on disk)."""
    with _encoders_lock:
        _encoders.clear()
