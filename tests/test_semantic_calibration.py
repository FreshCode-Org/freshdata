"""Confidence calibration: table lookup, fallbacks, provenance, gate wiring."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from freshdata.config import merge_options
from freshdata.report import CleanReport
from freshdata.semantic import scoring
from freshdata.semantic.context import build_semantic_context
from freshdata.semantic.policy import decide
from freshdata.semantic.scoring import (
    ActionConfidence,
    _IsotonicTable,
    calibrate_proposals,
    calibration_features,
    features_hash,
    make_proposal,
)
from freshdata.semantic.types import SemanticEvidence


@pytest.fixture(autouse=True)
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_DIR", str(tmp_path))
    scoring.reset_calibration_cache()
    yield tmp_path
    scoring.reset_calibration_cache()


def _proposal(backend: str = "embedding", issue_type: str = "reference_value", conf: float = 0.9):
    return make_proposal(
        column="status",
        raw_value="aktyve",
        proposed_value="active",
        issue_type=issue_type,
        expert="embedding" if backend == "embedding" else "reference",
        base_confidence=conf,
        evidence=(
            SemanticEvidence(
                "embedding", "cos=0.900 margin=0.850 model=fd-col-encoder-v1 sha=test", 0.0
            ),
        ),
        count=1,
        rationale="test",
        backend=backend,
    )


def _ctx():
    df = pd.DataFrame({"status": ["aktyve", "active", "pending"]})
    config = merge_options(
        None,
        semantic_mode="auto",
        semantic_context={
            "columns": {"status": {"allowed_values": ["active", "inactive", "pending"]}}
        },
    )
    return config, build_semantic_context(df, config)


def test_isotonic_interpolation_and_clamps():
    table = _IsotonicTable.from_json(
        json.dumps(
            {
                "version": "t1",
                "tables": {
                    "embedding": {
                        "reference_value": {"x": [0.0, 0.5, 1.0], "y": [0.1, 0.4, 0.9]}
                    }
                },
            }
        )
    )
    assert table.apply("embedding", "reference_value", 0.25) == pytest.approx(0.25)
    assert table.apply("embedding", "reference_value", 0.75) == pytest.approx(0.65)
    assert table.apply("embedding", "reference_value", -1.0) == pytest.approx(0.1)
    assert table.apply("embedding", "reference_value", 2.0) == pytest.approx(0.9)
    # Unknown backend/family -> identity.
    assert table.apply("deterministic", "reference_value", 0.8) == 0.8


def test_isotonic_rejects_malformed_tables():
    with pytest.raises(ValueError, match="at least two points"):
        _IsotonicTable.from_json(
            json.dumps(
                {"version": "x", "tables": {"embedding": {"*": {"x": [0.5], "y": [0.5]}}}}
            )
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        _IsotonicTable.from_json(
            json.dumps(
                {
                    "version": "x",
                    "tables": {"embedding": {"*": {"x": [0.0, 1.0], "y": [0.9, 0.1]}}},
                }
            )
        )


def test_packaged_default_calibrates_embedding():
    config, ctx = _ctx()
    calibrated = calibrate_proposals([_proposal(conf=0.85)], config, ctx)[0]
    assert calibrated.calibration is not None
    assert calibrated.calibration.calibration_version == "calib-default-1"
    assert calibrated.calibration.raw == 0.85
    assert calibrated.confidence < 0.85  # conservative curve pulls it down
    assert calibrated.confidence == calibrated.calibration.point


def test_deterministic_proposals_pass_through_untouched():
    config, ctx = _ctx()
    original = _proposal(backend="deterministic")
    calibrated = calibrate_proposals([original], config, ctx)[0]
    assert calibrated is original  # byte-identical object, no new metadata


def test_missing_table_falls_back_to_raw_with_note(monkeypatch, tmp_path):
    monkeypatch.setattr(scoring, "_PACKAGED_TABLE", tmp_path / "nope.json")
    scoring.reset_calibration_cache()
    config, ctx = _ctx()
    report = CleanReport(rows_before=1, rows_after=1, cols_before=1, cols_after=1)
    calibrated = calibrate_proposals([_proposal(conf=0.9)], config, ctx, report=report)[0]
    assert calibrated.confidence == 0.9
    assert calibrated.calibration.calibration_version == "uncalibrated"
    assert any("calibration table missing" in w for w in report.warnings)


def test_registry_calib_model_overrides_packaged(fresh_cache):
    calib_dir = fresh_cache / "calib-v1"
    calib_dir.mkdir(parents=True)
    (calib_dir / "calibration.json").write_text(
        json.dumps(
            {
                "version": "calib-user-9",
                "tables": {"embedding": {"reference_value": {"x": [0.0, 1.0], "y": [0.0, 0.5]}}},
            }
        )
    )
    scoring.reset_calibration_cache()
    config, ctx = _ctx()
    calibrated = calibrate_proposals([_proposal(conf=0.8)], config, ctx)[0]
    assert calibrated.calibration.calibration_version == "calib-user-9"
    assert calibrated.confidence == pytest.approx(0.4)


def test_embedding_never_reaches_certainty():
    table = _IsotonicTable.from_json(
        json.dumps(
            {"version": "x", "tables": {"embedding": {"*": {"x": [0.0, 1.0], "y": [1.0, 1.0]}}}}
        )
    )
    scoring._table_cache["table"] = table
    config, ctx = _ctx()
    calibrated = calibrate_proposals([_proposal(conf=0.999)], config, ctx)[0]
    assert calibrated.confidence <= 0.999  # ceiling holds even for a rogue table


def test_features_hash_stable_and_sensitive():
    config, ctx = _ctx()
    p = _proposal()
    features_a = calibration_features(p, ctx, ctx.info("status"))
    features_b = calibration_features(p, ctx, ctx.info("status"))
    assert features_hash(features_a) == features_hash(features_b)
    assert len(features_hash(features_a)) == 16
    features_b["raw_score"] = 0.123
    assert features_hash(features_a) != features_hash(features_b)


def test_features_include_required_signals():
    config, ctx = _ctx()
    features = calibration_features(_proposal(), ctx, ctx.info("status"))
    for key in (
        "raw_score",
        "backend",
        "issue_type",
        "risk",
        "role_confidence",
        "semantic_type_confidence",
        "distinct_support",
        "coverage",
        "memory_support_count",
        "policy_rule_present",
        "allowed_values_present",
        "margin_to_second_candidate",
    ):
        assert key in features
    assert features["backend"] == "embedding"
    assert features["allowed_values_present"] is True
    assert features["margin_to_second_candidate"] == pytest.approx(0.85)


def test_gate_consumes_calibrated_confidence():
    """A calibration table that demotes a raw score must flip auto -> suggest."""
    demoting = _IsotonicTable.from_json(
        json.dumps(
            {
                "version": "demote",
                "tables": {"embedding": {"*": {"x": [0.0, 1.0], "y": [0.0, 0.5]}}},
            }
        )
    )
    scoring._table_cache["table"] = demoting
    config, ctx = _ctx()
    calibrated = calibrate_proposals([_proposal(conf=0.98)], config, ctx)[0]
    decision = decide(calibrated, config, ctx)
    assert calibrated.confidence < 0.70  # 0.98 -> 0.49, below the review floor
    assert decision.action == "skip"


def test_action_confidence_is_frozen():
    ac = ActionConfidence(point=0.9, raw=0.95, calibration_version="v", features_hash="h")
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
        ac.point = 0.1  # type: ignore[misc]
