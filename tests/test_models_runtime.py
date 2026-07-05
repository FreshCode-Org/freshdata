"""Encoder runtime: availability, lazy singleton, stub determinism, cache."""

from __future__ import annotations

import numpy as np
import pytest

from freshdata.models import ModelError, runtime
from freshdata.models.stub import StubEncoder
from freshdata.semantic.cache import EmbeddingCache


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("FRESHDATA_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("FRESHDATA_STUB_ENCODER", raising=False)
    runtime.set_encoder_factory(None)
    yield
    runtime.set_encoder_factory(None)


def test_availability_without_extra_or_model():
    ok, reason = runtime.availability()
    assert ok is False
    # Either the extra or the model files are missing; both messages must
    # point at an explicit, user-driven fix (never an automatic download).
    assert "freshdata-cleaner[semantic]" in reason or "fd.models.pull" in reason


def test_get_encoder_unavailable_raises():
    with pytest.raises(ModelError):
        runtime.get_encoder()


def test_stub_env_enables_encoder(monkeypatch):
    monkeypatch.setenv("FRESHDATA_STUB_ENCODER", "1")
    ok, reason = runtime.availability()
    assert ok is True and reason == ""
    encoder = runtime.get_encoder()
    assert isinstance(encoder, StubEncoder)
    assert runtime.get_encoder() is encoder  # lazy singleton per process


def test_factory_hook_wins(monkeypatch):
    monkeypatch.setenv("FRESHDATA_STUB_ENCODER", "1")
    sentinel = StubEncoder()
    runtime.set_encoder_factory(lambda model_id: sentinel)
    assert runtime.get_encoder() is sentinel
    runtime.set_encoder_factory(None)
    assert runtime.get_encoder() is not sentinel


def test_stub_vectors_deterministic_golden():
    enc = StubEncoder()
    vec = enc.encode_texts(["status | activ; pendng"])[0]
    again = enc.encode_texts(["status | activ; pendng"])[0]
    np.testing.assert_array_equal(vec, again)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
    # Golden prefix: sha256-derived, must never drift across platforms/versions.
    np.testing.assert_allclose(
        vec[:4], [0.0298737, -0.044915, -0.03864779, 0.06678061], rtol=1e-4
    )


def test_stub_similarity_tracks_ngram_overlap():
    enc = StubEncoder()
    activ, active, pending = enc.encode_texts(["activ", "active", "pending"])
    assert float(activ @ active) > 0.5  # shared bigrams -> close
    assert float(activ @ pending) < 0.3  # unrelated strings -> far


def test_stub_normalization_folds_case_and_whitespace():
    enc = StubEncoder()
    a, b = enc.encode_texts(["Active ", "active"])
    np.testing.assert_array_equal(a, b)


def test_stub_batch_matches_single():
    enc = StubEncoder()
    batch = enc.encode_texts(["a", "b"])
    np.testing.assert_array_equal(batch[1], enc.encode_texts(["b"])[0])


def test_column_text_format():
    assert runtime.column_text("mob_no", ["98765", "91234"]) == "mob_no | 98765; 91234"


def test_embedding_cache_hits_and_eviction():
    class Counting(StubEncoder):
        calls = 0

        def encode_texts(self, texts):
            Counting.calls += 1
            return super().encode_texts(texts)

    cache = EmbeddingCache(Counting(), capacity=2)
    cache.encode(["a", "b"])
    cache.encode(["a"])  # served from cache
    assert Counting.calls == 1
    assert cache.hits == 1
    cache.encode(["c"])  # evicts the LRU entry ("b")
    cache.encode(["b"])  # miss again after eviction
    assert Counting.calls == 3
    assert len(cache) == 2


def test_embedding_cache_keyed_by_model_identity():
    stub = StubEncoder()
    cache = EmbeddingCache(stub, capacity=8)
    cache.encode(["a"])
    stub.model_sha256 = "different"  # simulated model swap
    cache.encode(["a"])
    assert cache.misses == 2  # never served across model identities


def test_embedding_cache_disabled():
    cache = EmbeddingCache(StubEncoder(), capacity=0)
    cache.encode(["a"])
    cache.encode(["a"])
    assert cache.hits == 0
    assert len(cache) == 0


def test_onnx_encoder_requires_extra():
    pytest.importorskip("onnxruntime")
    # With the extra installed but no model files, availability names the pull path.
    ok, reason = runtime.availability()
    assert ok is False
    assert "fd.models.pull" in reason
