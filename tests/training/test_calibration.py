"""Calibration pipeline: feature extraction, isotonic fit, export, eval, gates."""

from __future__ import annotations

import json

import pytest
from training.calibration import build_features, eval_calibration, export_tables, fit_isotonic


@pytest.fixture(scope="module")
def features():
    return build_features.build_features(seeds=2)


class TestFeatureExtraction:
    def test_features_have_required_keys(self, features):
        assert features
        required = {
            "raw_score", "backend", "issue_type", "risk", "role_confidence",
            "semantic_type_confidence", "distinct_support", "coverage",
            "memory_support_count", "learned_precision", "policy_rule_present",
            "allowed_values_present", "margin_to_second_candidate", "semantic_mode",
            "correct",
        }
        assert required <= set(features[0])

    def test_multiple_backends_present(self, features):
        backends = {f["backend"] for f in features}
        assert "deterministic" in backends


class TestIsotonicFit:
    def test_pav_is_monotone(self):
        import numpy as np

        xs, ys = fit_isotonic.pav(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        )
        assert list(ys) == sorted(ys)

    def test_fit_curve_returns_none_below_min_samples(self):
        assert fit_isotonic.fit_curve([0.1, 0.2], [True, False]) is None

    def test_fit_curve_produces_bracketing_knots(self):
        curve = fit_isotonic.fit_curve(
            [0.1 * i for i in range(10)], [i >= 5 for i in range(10)])
        assert curve is not None
        assert curve["x"][0] == 0.0
        assert curve["x"][-1] == 1.0
        assert curve["y"] == sorted(curve["y"])

    def test_fit_tables_skips_deterministic_by_default(self, features):
        tables = fit_isotonic.fit_tables(features, fit_deterministic=False)
        assert "deterministic" not in tables

    def test_fit_tables_includes_deterministic_when_requested(self, features):
        tables = fit_isotonic.fit_tables(features, fit_deterministic=True)
        assert "deterministic" in tables


class TestExport:
    def test_export_produces_runtime_loadable_table(self, tmp_path, features):
        tables = fit_isotonic.fit_tables(features, fit_deterministic=True)
        curves_path = tmp_path / "curves.json"
        curves_path.write_text(json.dumps({"tables": tables, "n_records": len(features)}))
        primary = export_tables.export(curves_path, out_dir=tmp_path)
        assert primary.is_file()
        assert (tmp_path / "calib.json").is_file()
        data = json.loads(primary.read_text())
        assert data["version"] == "calib-v1"

        from freshdata.semantic.scoring import _IsotonicTable

        table = _IsotonicTable.from_json(primary.read_text())
        assert table.version == "calib-v1"


class TestEvalGates:
    def test_ece_and_precision_computed(self, tmp_path, features):
        from training.common import write_jsonl

        tables = fit_isotonic.fit_tables(features, fit_deterministic=True)
        curves_path = tmp_path / "curves.json"
        curves_path.write_text(json.dumps({"tables": tables, "n_records": len(features)}))
        export_tables.export(curves_path, out_dir=tmp_path)
        write_jsonl(tmp_path / "features.jsonl", features)
        metrics = eval_calibration.evaluate(
            features_path=tmp_path / "features.jsonl",
            table_path=tmp_path / "calibration.json", out_dir=tmp_path)
        assert "ece" in metrics
        assert "precision_at_0.95" in metrics
        assert metrics["ece"] >= 0.0

    def test_gate_fails_on_bad_ece(self):
        failures = eval_calibration.check_gates({"ece": 0.5, "precision_at_0.95": 1.0})
        assert any("ECE" in f for f in failures)

    def test_gate_fails_on_bad_precision(self):
        failures = eval_calibration.check_gates({"ece": 0.0, "precision_at_0.95": 0.1})
        assert any("P@0.95" in f for f in failures)

    def test_gate_passes_on_good_metrics(self):
        assert eval_calibration.check_gates({"ece": 0.01, "precision_at_0.95": 1.0}) == []


class TestRuntimeCanLoadExportedTable:
    def test_runtime_loads_via_model_registry(self, tmp_path, monkeypatch, features):
        tables = fit_isotonic.fit_tables(features, fit_deterministic=True)
        curves_path = tmp_path / "curves.json"
        curves_path.write_text(json.dumps({"tables": tables, "n_records": len(features)}))
        export_tables.export(curves_path, out_dir=tmp_path / "artifact")

        model_dir = tmp_path / "models"
        target = model_dir / "calib-v1"
        target.mkdir(parents=True)
        (target / "calibration.json").write_text(
            (tmp_path / "artifact" / "calibration.json").read_text())
        monkeypatch.setenv("FRESHDATA_MODEL_DIR", str(model_dir))

        from freshdata.models import registry
        from freshdata.semantic import scoring

        scoring.reset_calibration_cache()
        try:
            assert registry.is_installed("calib-v1")
            table = scoring._load_calibration_table()
            assert table is not None
        finally:
            scoring.reset_calibration_cache()
