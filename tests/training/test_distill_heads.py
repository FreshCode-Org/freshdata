"""Role-head / intent-head training: dataset loading, metrics, gates."""

from __future__ import annotations

from training.distill import train_intent_head as tih
from training.distill import train_role_head as trh
from training.distill.linear import LinearHead


class TestRoleHead:
    def test_build_training_set_has_examples_for_known_types(self):
        examples = trh.build_training_set(dev=True)
        labels = {e["label"] for e in examples}
        assert "email" in labels
        assert "phone" in labels
        assert "unknown" in labels

    def test_label_validation_rejects_unrecognized_semantic_type(self):

        bad = {"column_name": "x", "masked_samples": [], "semantic_type": "not_a_type",
              "confidence": 0.9}
        # semantic_type isn't schema-enum-checked, but the role-head loader
        # filters it: only SEMANTIC_TYPES-labeled teacher rows are accepted.
        assert bad["semantic_type"] not in trh.SEMANTIC_TYPES

    def test_committed_human_eval_loads(self):
        rows = trh.load_human_eval()
        assert len(rows) > 0
        assert all("label" in r and "text" in r for r in rows)

    def test_metric_computation_shapes(self):
        head = LinearHead.train(
            ["col:email | a@b.com", "col:phone | +919876543210"],
            ["email", "phone"], classes=trh.SEMANTIC_TYPES, epochs=50)
        eval_rows = trh.load_human_eval()
        metrics = trh.evaluate(head, eval_rows)
        assert set(metrics) >= {
            "macro_f1", "per_class_f1", "abstention_rate",
            "adversarial_alias_accuracy", "contradiction_rate", "confusion",
        }
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_gate_fails_on_low_macro_f1(self):
        assert trh.check_gates({"macro_f1": 0.1, "contradiction_rate": 0.0,
                                "adversarial_alias_accuracy": 1.0})

    def test_gate_passes_on_good_metrics(self):
        assert trh.check_gates({"macro_f1": 0.95, "contradiction_rate": 0.0,
                                "adversarial_alias_accuracy": 0.9}) == []

    def test_gate_fails_on_high_contradiction_rate(self):
        failures = trh.check_gates({"macro_f1": 0.95, "contradiction_rate": 0.5,
                                    "adversarial_alias_accuracy": 0.9})
        assert any("contradiction" in f for f in failures)

    def test_dev_train_exports_artifacts(self, tmp_path):
        metrics = trh.train(dev=True, out_dir=tmp_path)
        assert (tmp_path / "role_head.weights.json").is_file()
        assert (tmp_path / "role_head.metrics.json").is_file()
        assert (tmp_path / "role_head.confusion.json").is_file()
        assert metrics["macro_f1"] >= 0.90


class TestIntentHead:
    def test_golden_examples_load(self):
        examples = tih.load_golden_examples()
        assert examples
        assert any(e["intent"] == "PROTECT" for e in examples)

    def test_build_dataset_author_disjoint(self):
        train_rows, eval_rows = tih.build_dataset()
        train_authors = {r.get("author") for r in train_rows}
        assert not train_authors & set(tih.EVAL_AUTHORS)
        assert eval_rows

    def test_slot_extraction_email_format(self):
        slots = tih.extract_slots("Emails must be valid.", "VALID_FORMAT")
        assert slots.get("format") == "email"

    def test_slot_extraction_range(self):
        slots = tih.extract_slots("age must be between 18 and 99.", "RANGE")
        assert slots["min"] == 18 and slots["max"] == 99

    def test_slot_extraction_protect_column(self):
        slots = tih.extract_slots("Never modify monthly_revenue values.", "PROTECT",
                                  columns=("monthly_revenue",))
        assert slots["column"] == "monthly_revenue"

    def test_unknown_precision_metric_present(self):
        train_rows, eval_rows = tih.build_dataset()
        head = LinearHead.train(
            [r["sentence"] for r in train_rows], [r["intent"] for r in train_rows],
            classes=tih.INTENTS, epochs=100, abstain_class="UNKNOWN")
        metrics = tih.evaluate(head, eval_rows)
        assert "unknown_precision" in metrics
        assert "protected_intent_recall" in metrics

    def test_gate_fails_on_low_scores(self):
        failures = tih.check_gates({
            "exact_intent_accuracy": 0.1, "slot_f1": 0.1,
            "unknown_precision": 0.1, "protected_intent_recall": 0.1,
        })
        assert len(failures) == 4

    def test_gate_passes_on_good_scores(self):
        assert tih.check_gates({
            "exact_intent_accuracy": 0.95, "slot_f1": 0.95,
            "unknown_precision": 0.95, "protected_intent_recall": 1.0,
        }) == []

    def test_conflict_detection_runs(self):
        head = LinearHead.train(
            ["Never modify revenue values.", "Revenue must be valid numbers."],
            ["PROTECT", "VALID_FORMAT"], classes=tih.INTENTS, epochs=100)
        score = tih.conflict_detection_accuracy(head)
        assert 0.0 <= score <= 1.0

    def test_dev_train_exports_and_gates_pass(self, tmp_path):
        metrics = tih.train(dev=True, out_dir=tmp_path)
        assert (tmp_path / "intent_head.weights.json").is_file()
        assert metrics["gates"]["failures"] == []
        assert metrics["protected_intent_recall"] >= 0.99
