"""Model download robustness and race-free lazy encoder loading.

Covers #340 (``OnnxEncoder._load`` must never expose a half-loaded encoder to
a concurrent caller) and #341 (``fd.models.pull`` must pass a network timeout
and refuse truncated transfers). Everything is offline: ``onnxruntime`` and
``tokenizers`` are replaced with fakes, and ``urllib.request.urlopen`` with an
in-memory response.
"""

from __future__ import annotations

import io
import sys
import threading
import types
import urllib.request
from email.message import Message

import numpy as np
import pytest

import freshdata as fd
from freshdata.models import download as dl
from freshdata.models import runtime

# --------------------------------------------------------------------------
# #340: lazy encoder load race
# --------------------------------------------------------------------------


class _Encoding:
    ids = [1, 2]
    attention_mask = [1, 1]


class _FakeTokenizer:
    def encode_batch(self, batch):
        return [_Encoding() for _ in batch]


class _FakeSession:
    def run(self, _outputs, feeds):
        return [np.ones(feeds["input_ids"].shape + (4,), dtype=np.float32)]


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    """Install fake onnxruntime/tokenizers; ``hooks`` controls from_file."""
    monkeypatch.setenv("FRESHDATA_MODEL_DIR", str(tmp_path))
    hooks: dict = {"from_file": lambda: None, "from_file_calls": 0, "session_calls": 0}

    ort = types.ModuleType("onnxruntime")
    tok = types.ModuleType("tokenizers")

    class SessionOptions:
        intra_op_num_threads = 1

    def inference_session(*_args, **_kwargs):
        hooks["session_calls"] += 1
        return _FakeSession()

    class Tokenizer:
        @staticmethod
        def from_file(_path):
            hooks["from_file_calls"] += 1
            hooks["from_file"]()
            return _FakeTokenizer()

    ort.SessionOptions = SessionOptions
    ort.InferenceSession = inference_session
    tok.Tokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "tokenizers", tok)
    monkeypatch.setattr(runtime, "verify", lambda model_id: False)
    return hooks


def test_concurrent_first_use_waits_for_complete_load(fake_runtime):
    started = threading.Event()
    release = threading.Event()

    def slow_from_file():
        started.set()
        assert release.wait(10), "test did not release the tokenizer load"

    fake_runtime["from_file"] = slow_from_file
    enc = runtime.OnnxEncoder("fd-col-encoder-v1")
    errors: list[str] = []
    results: list[np.ndarray] = []

    def work():
        try:
            results.append(enc.encode_texts(["a"]))
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion
            errors.append(repr(exc))

    first = threading.Thread(target=work)
    first.start()
    assert started.wait(10)
    # The first thread is now inside Tokenizer.from_file holding the lock.
    second = threading.Thread(target=work)
    second.start()
    second.join(0.3)
    # Before the fix the second thread saw ``_session`` set and crashed on a
    # None tokenizer; now it must still be waiting for the load to finish.
    assert second.is_alive(), errors
    assert errors == []
    release.set()
    first.join(10)
    second.join(10)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert len(results) == 2
    assert all(r.shape == (1, 4) for r in results)
    assert fake_runtime["from_file_calls"] == 1
    assert fake_runtime["session_calls"] == 1


def test_many_threads_first_use_load_once(fake_runtime):
    n = 8
    barrier = threading.Barrier(n)
    enc = runtime.OnnxEncoder("fd-col-encoder-v1")
    errors: list[str] = []

    def work():
        try:
            barrier.wait(10)
            enc.encode_texts(["a", "b"])
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion
            errors.append(repr(exc))

    threads = [threading.Thread(target=work) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert errors == []
    assert fake_runtime["from_file_calls"] == 1
    assert fake_runtime["session_calls"] == 1


def test_fast_path_requires_both_session_and_tokenizer(fake_runtime):
    enc = runtime.OnnxEncoder("fd-col-encoder-v1")
    enc._session = _FakeSession()  # half-loaded: session without tokenizer
    enc._load()
    assert enc._tokenizer is not None
    assert fake_runtime["from_file_calls"] == 1

    enc._load()  # fully loaded: fast path, no reload
    assert fake_runtime["from_file_calls"] == 1


def test_failed_tokenizer_load_leaves_encoder_unloaded(fake_runtime):
    def boom():
        raise RuntimeError("tokenizer.json unreadable")

    fake_runtime["from_file"] = boom
    enc = runtime.OnnxEncoder("fd-col-encoder-v1")
    with pytest.raises(RuntimeError, match="unreadable"):
        enc.encode_texts(["a"])
    assert enc._session is None
    assert enc._tokenizer is None

    fake_runtime["from_file"] = lambda: None
    assert enc.encode_texts(["a"]).shape == (1, 4)


# --------------------------------------------------------------------------
# #341: download timeout and truncation detection
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes, content_length: int | None) -> None:
        self._stream = io.BytesIO(body)
        self.headers = Message()
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def read(self, amt: int = -1) -> bytes:
        return self._stream.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.fixture
def model_home(tmp_path, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("FRESHDATA_MODEL_URL_BASE", "https://example.test/models")
    monkeypatch.delenv("FRESHDATA_MODEL_TIMEOUT", raising=False)
    return tmp_path


def _install_urlopen(monkeypatch, body: bytes, content_length: int | None):
    calls: list[dict] = []

    def fake_urlopen(url, *args, **kwargs):
        calls.append({"url": url, "args": args, "kwargs": kwargs})
        return _FakeResponse(body, content_length)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def test_truncated_download_raises_and_installs_nothing(model_home, monkeypatch):
    _install_urlopen(monkeypatch, b"X" * 1000, content_length=1_000_000)
    with pytest.raises(OSError, match=r"incomplete download of .*got 1000 of 1000000 bytes"):
        fd.models.pull("fd-intent-v1")
    target = model_home / "fd-intent-v1"
    assert not (target / "model.onnx").exists()
    assert not list(target.glob("*.part"))
    assert fd.models.status()["fd-intent-v1"]["installed"] is False


def test_fetch_truncated_leaves_partial_for_caller_cleanup(tmp_path, monkeypatch):
    monkeypatch.delenv("FRESHDATA_MODEL_TIMEOUT", raising=False)
    _install_urlopen(monkeypatch, b"abc", content_length=10)
    dest = tmp_path / "x.part"
    with pytest.raises(OSError, match="got 3 of 10 bytes"):
        dl._fetch("https://example.test/m/x", dest)
    assert dest.read_bytes() == b"abc"  # pull()'s finally removes it


def test_complete_download_installs(model_home, monkeypatch):
    payload = b"model-bytes" * 500_000  # larger than one read chunk
    calls = _install_urlopen(monkeypatch, payload, content_length=len(payload))
    primary = fd.models.pull("fd-intent-v1")
    assert primary.read_bytes() == payload
    assert not list(primary.parent.glob("*.part"))
    assert [c["url"] for c in calls] == ["https://example.test/models/fd-intent-v1/model.onnx"]


def test_download_without_content_length_is_accepted(model_home, monkeypatch):
    _install_urlopen(monkeypatch, b"payload", content_length=None)
    primary = fd.models.pull("fd-intent-v1")
    assert primary.read_bytes() == b"payload"


def test_default_timeout_is_passed(model_home, monkeypatch):
    calls = _install_urlopen(monkeypatch, b"ok", content_length=2)
    fd.models.pull("fd-intent-v1")
    assert calls[0]["kwargs"]["timeout"] == 60.0


def test_timeout_env_override_is_passed(model_home, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_TIMEOUT", "2.5")
    calls = _install_urlopen(monkeypatch, b"ok", content_length=2)
    fd.models.pull("fd-intent-v1")
    assert calls[0]["kwargs"]["timeout"] == 2.5


@pytest.mark.parametrize("raw", ["abc", "", "0", "-1", "nan", "inf"])
def test_invalid_timeout_env_raises_value_error(model_home, monkeypatch, raw):
    monkeypatch.setenv("FRESHDATA_MODEL_TIMEOUT", raw)
    calls = _install_urlopen(monkeypatch, b"ok", content_length=2)
    with pytest.raises(ValueError, match="FRESHDATA_MODEL_TIMEOUT must be a positive number"):
        fd.models.pull("fd-intent-v1")
    assert calls == []
    target = model_home / "fd-intent-v1"
    assert not (target / "model.onnx").exists()
    assert not list(target.glob("*.part"))


def test_timeout_error_discards_partial(model_home, monkeypatch):
    def stalled(url, *args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", stalled)
    with pytest.raises(TimeoutError):
        fd.models.pull("fd-intent-v1")
    target = model_home / "fd-intent-v1"
    assert not (target / "model.onnx").exists()
    assert not list(target.glob("*.part"))
