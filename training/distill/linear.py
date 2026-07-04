"""Deterministic pure-numpy linear heads over hashed character n-grams.

The Phase-5 heads (semantic-type role head, context intent head) are tiny
multinomial logistic regressions: strong enough for the eval corpora, small
enough to export, and — crucially — trainable with zero optional
dependencies so the dev pipeline runs anywhere the test suite runs.
Everything is deterministic: hashing uses sha256, training is full-batch
gradient descent from a zero init (no RNG at all).
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURIZER_VERSION = "hashed-ngram-1"


@dataclass(frozen=True)
class FeaturizerConfig:
    dim: int = 2048
    ngram_sizes: tuple[int, ...] = (2, 3)
    lowercase: bool = True
    #: Hash word unigrams/bigrams too — intent-bearing words ("unique",
    #: "never modify") generalize across paraphrases better than char grams.
    word_ngrams: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "version": FEATURIZER_VERSION,
            "dim": self.dim,
            "ngram_sizes": list(self.ngram_sizes),
            "lowercase": self.lowercase,
            "word_ngrams": self.word_ngrams,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FeaturizerConfig:
        return cls(
            dim=int(data["dim"]),  # type: ignore[arg-type]
            ngram_sizes=tuple(data["ngram_sizes"]),  # type: ignore[arg-type]
            lowercase=bool(data["lowercase"]),
            word_ngrams=bool(data.get("word_ngrams", False)),
        )


def _bucket(gram: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(gram.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


def featurize(texts: Sequence[str], config: FeaturizerConfig) -> np.ndarray:
    """L2-normalized hashed n-gram counts, shape ``(n, dim)`` float32."""
    out = np.zeros((len(texts), config.dim), dtype=np.float32)
    for i, text in enumerate(texts):
        body = str(text)
        if config.lowercase:
            body = body.casefold()
        padded = f"^{body}$"
        for n in config.ngram_sizes:
            for j in range(max(1, len(padded) - n + 1)):
                index, sign = _bucket(f"{n}:{padded[j:j + n]}", config.dim)
                out[i, index] += sign
        if config.word_ngrams:
            words = body.split()
            for word in words:
                index, sign = _bucket(f"w1:{word}", config.dim)
                out[i, index] += 2.0 * sign
            for j in range(len(words) - 1):
                index, sign = _bucket(f"w2:{words[j]} {words[j + 1]}", config.dim)
                out[i, index] += 2.0 * sign
        norm = float(np.linalg.norm(out[i]))
        if norm:
            out[i] /= norm
    return out


@dataclass
class LinearHead:
    """Multinomial logistic regression with optional abstention."""

    classes: tuple[str, ...]
    weights: np.ndarray  # (C, D) float32
    bias: np.ndarray     # (C,) float32
    featurizer: FeaturizerConfig
    abstain_class: str | None = "unknown"
    abstain_threshold: float = 0.5

    # -- training ----------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        classes: Sequence[str],
        featurizer: FeaturizerConfig | None = None,
        epochs: int = 300,
        lr: float = 2.0,
        l2: float = 1e-4,
        class_weights: dict[str, float] | None = None,
        abstain_class: str | None = "unknown",
        abstain_threshold: float = 0.5,
    ) -> LinearHead:
        config = featurizer or FeaturizerConfig()
        class_list = tuple(classes)
        index = {c: i for i, c in enumerate(class_list)}
        x = featurize(texts, config)
        y = np.array([index[label] for label in labels], dtype=np.int64)
        n, d = x.shape
        c = len(class_list)
        weights = np.zeros((c, d), dtype=np.float64)
        bias = np.zeros(c, dtype=np.float64)
        sample_weight = np.ones(n, dtype=np.float64)
        if class_weights:
            for i, label in enumerate(labels):
                sample_weight[i] = float(class_weights.get(label, 1.0))
        sample_weight /= sample_weight.mean()
        onehot = np.zeros((n, c), dtype=np.float64)
        onehot[np.arange(n), y] = 1.0
        velocity_w = np.zeros_like(weights)
        velocity_b = np.zeros_like(bias)
        momentum = 0.9
        for _ in range(epochs):
            logits = x @ weights.T + bias
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            grad = (probs - onehot) * sample_weight[:, None]
            grad_w = grad.T @ x / n + l2 * weights
            grad_b = grad.mean(axis=0)
            velocity_w = momentum * velocity_w - lr * grad_w
            velocity_b = momentum * velocity_b - lr * grad_b
            weights += velocity_w
            bias += velocity_b
        return cls(
            classes=class_list,
            weights=weights.astype(np.float32),
            bias=bias.astype(np.float32),
            featurizer=config,
            abstain_class=abstain_class,
            abstain_threshold=abstain_threshold,
        )

    # -- inference ----------------------------------------------------------------

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        x = featurize(texts, self.featurizer)
        logits = x @ self.weights.T.astype(np.float64) + self.bias.astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, texts: Sequence[str]) -> list[tuple[str, float]]:
        """(label, confidence) per text, abstaining below the threshold."""
        probs = self.predict_proba(texts)
        out: list[tuple[str, float]] = []
        for row in probs:
            best = int(row.argmax())
            label, confidence = self.classes[best], float(row[best])
            if (
                self.abstain_class is not None
                and confidence < self.abstain_threshold
                and label != self.abstain_class
            ):
                label = self.abstain_class
            out.append((label, round(confidence, 6)))
        return out

    # -- persistence ----------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "freshdata-linear-head-1",
            "classes": list(self.classes),
            "featurizer": self.featurizer.to_dict(),
            "abstain_class": self.abstain_class,
            "abstain_threshold": self.abstain_threshold,
            "weights_b64": base64.b64encode(self.weights.astype(np.float32).tobytes()).decode(),
            "bias_b64": base64.b64encode(self.bias.astype(np.float32).tobytes()).decode(),
            "shape": list(self.weights.shape),
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> LinearHead:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        shape = tuple(int(v) for v in data["shape"])
        weights = np.frombuffer(
            base64.b64decode(data["weights_b64"]), dtype=np.float32
        ).reshape(shape).copy()
        bias = np.frombuffer(base64.b64decode(data["bias_b64"]), dtype=np.float32).copy()
        return cls(
            classes=tuple(data["classes"]),
            weights=weights,
            bias=bias,
            featurizer=FeaturizerConfig.from_dict(data["featurizer"]),
            abstain_class=data.get("abstain_class"),
            abstain_threshold=float(data.get("abstain_threshold", 0.5)),
        )


# --------------------------------------------------------------------------- #
# int8 quantization (our own artifact format; ONNX quantization is optional)
# --------------------------------------------------------------------------- #

def quantize_int8(head: LinearHead) -> dict[str, object]:
    """Symmetric per-row int8 quantization of the weight matrix."""
    scales = np.abs(head.weights).max(axis=1) / 127.0
    scales[scales == 0.0] = 1.0
    quantized = np.clip(np.round(head.weights / scales[:, None]), -127, 127).astype(np.int8)
    return {
        "format": "freshdata-linear-head-int8-1",
        "classes": list(head.classes),
        "featurizer": head.featurizer.to_dict(),
        "abstain_class": head.abstain_class,
        "abstain_threshold": head.abstain_threshold,
        "weights_int8_b64": base64.b64encode(quantized.tobytes()).decode(),
        "scales_b64": base64.b64encode(scales.astype(np.float32).tobytes()).decode(),
        "bias_b64": base64.b64encode(head.bias.astype(np.float32).tobytes()).decode(),
        "shape": list(head.weights.shape),
    }


def dequantize(data: dict[str, object]) -> LinearHead:
    shape = tuple(int(v) for v in data["shape"])  # type: ignore[arg-type]
    quantized = np.frombuffer(
        base64.b64decode(data["weights_int8_b64"]), dtype=np.int8  # type: ignore[arg-type]
    ).reshape(shape)
    scales = np.frombuffer(base64.b64decode(data["scales_b64"]), dtype=np.float32)  # type: ignore[arg-type]
    bias = np.frombuffer(base64.b64decode(data["bias_b64"]), dtype=np.float32).copy()  # type: ignore[arg-type]
    weights = (quantized.astype(np.float32) * scales[:, None]).astype(np.float32)
    return LinearHead(
        classes=tuple(data["classes"]),  # type: ignore[arg-type]
        weights=weights,
        bias=bias,
        featurizer=FeaturizerConfig.from_dict(data["featurizer"]),  # type: ignore[arg-type]
        abstain_class=data.get("abstain_class"),  # type: ignore[arg-type]
        abstain_threshold=float(data.get("abstain_threshold", 0.5)),  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# shared metrics
# --------------------------------------------------------------------------- #

def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> float:
    scores = per_class_f1(y_true, y_pred, classes)
    present = [f1 for cls, f1 in scores.items() if any(t == cls for t in y_true)]
    return float(np.mean(present)) if present else 0.0


def per_class_f1(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        denominator = 2 * tp + fp + fn
        out[cls] = (2 * tp / denominator) if denominator else 0.0
    return out


def confusion(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> dict[str, dict[str, int]]:
    matrix = {t: dict.fromkeys(classes, 0) for t in classes}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix
