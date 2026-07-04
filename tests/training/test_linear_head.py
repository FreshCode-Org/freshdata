"""Core linear-head trainer: featurization, training, quantization."""

from __future__ import annotations

from training.distill.linear import (
    FeaturizerConfig,
    LinearHead,
    dequantize,
    featurize,
    macro_f1,
    quantize_int8,
)

TEXTS = ["email column: a@b.com", "phone column: +919876543210",
         "email column: c@d.com", "phone column: +919812345670"] * 5
LABELS = ["email", "phone", "email", "phone"] * 5
CLASSES = ("email", "phone", "unknown")


def test_featurize_is_deterministic():
    config = FeaturizerConfig(dim=256)
    a = featurize(TEXTS, config)
    b = featurize(TEXTS, config)
    assert (a == b).all()


def test_train_predict_roundtrip():
    head = LinearHead.train(TEXTS, LABELS, classes=CLASSES, epochs=100,
                            featurizer=FeaturizerConfig(dim=256))
    predictions = head.predict(TEXTS)
    accuracy = sum(1 for (p, _), t in zip(predictions, LABELS) if p == t) / len(LABELS)
    assert accuracy >= 0.9


def test_save_load_roundtrip(tmp_path):
    head = LinearHead.train(TEXTS, LABELS, classes=CLASSES, epochs=50)
    path = head.save(tmp_path / "head.json")
    loaded = LinearHead.load(path)
    assert loaded.classes == head.classes
    assert (loaded.weights == head.weights).all()
    assert head.predict(TEXTS) == loaded.predict(TEXTS)


def test_quantize_dequantize_close_to_original():
    head = LinearHead.train(TEXTS, LABELS, classes=CLASSES, epochs=100)
    quantized = quantize_int8(head)
    restored = dequantize(quantized)
    original_predictions = [p for p, _ in head.predict(TEXTS)]
    restored_predictions = [p for p, _ in restored.predict(TEXTS)]
    agreement = sum(1 for a, b in zip(original_predictions, restored_predictions) if a == b)
    assert agreement / len(TEXTS) >= 0.9


def test_macro_f1_perfect_predictions():
    assert macro_f1(["a", "b", "a"], ["a", "b", "a"], ("a", "b")) == 1.0


def test_macro_f1_all_wrong():
    assert macro_f1(["a", "b"], ["b", "a"], ("a", "b")) == 0.0


def test_abstention_below_threshold():
    head = LinearHead.train(TEXTS, LABELS, classes=CLASSES, epochs=50,
                            abstain_class="unknown", abstain_threshold=0.999)
    predictions = head.predict(["completely unrelated gibberish text zzz"])
    assert predictions[0][0] == "unknown"
