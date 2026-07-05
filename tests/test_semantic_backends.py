"""Named semantic backends: default-path identity, graceful degradation,
embedding proposals, budget, and provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig, merge_options
from freshdata.models import runtime as model_runtime
from freshdata.repairplan import build_repair_plan
from freshdata.semantic.backends import Budget, gather_proposals
from freshdata.semantic.context import build_semantic_context


class ControlledEncoder:
    """Test encoder with exact, hand-authored cosine geometry.

    Known texts map to fixed unit vectors; unknown texts get their own
    orthogonal axis (cosine 0 to everything else).
    """

    model_id = "fd-col-encoder-v1"
    model_sha256 = "test-controlled"
    dim = 64

    def __init__(self, vectors: dict[str, np.ndarray] | None = None) -> None:
        self._vectors = dict(vectors or {})
        self._next_axis = 32
        self.calls: list[list[str]] = []

    @staticmethod
    def axis(i: int, dim: int = 64) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        v[i] = 1.0
        return v

    @classmethod
    def blend(cls, i: int, j: int, cosine: float) -> np.ndarray:
        """A unit vector with exactly ``cosine`` similarity to axis ``i``."""
        v = cosine * cls.axis(i) + float(np.sqrt(1 - cosine**2)) * cls.axis(j)
        return v.astype(np.float32)

    def encode_texts(self, texts):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            vec = self._vectors.get(text)
            if vec is None:
                vec = self.axis(self._next_axis)
                self._vectors[text] = vec
                self._next_axis += 1
            out[row] = vec
        return out


@pytest.fixture
def encoder_factory():
    """Install a ControlledEncoder and hand it back for assertions."""
    holder: dict[str, ControlledEncoder] = {}

    def install(vectors: dict[str, np.ndarray] | None = None) -> ControlledEncoder:
        encoder = ControlledEncoder(vectors)
        holder["encoder"] = encoder
        model_runtime.set_encoder_factory(lambda model_id: encoder)
        return encoder

    yield install
    model_runtime.set_encoder_factory(None)


def _status_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "status": ["aktyve", "active", "inactive", "pending", "active", "pending"],
            "amount": [1, 2, 3, 4, 5, 6],
        }
    )


def _status_config(**overrides) -> CleanConfig:
    base = {
        "semantic_mode": "auto",
        "semantic_backends": ("deterministic", "embedding"),
        "semantic_context": {
            "columns": {"status": {"allowed_values": ["active", "inactive", "pending"]}}
        },
    }
    base.update(overrides)
    return merge_options(None, **base)


def _status_vectors() -> dict[str, np.ndarray]:
    return {
        "active": ControlledEncoder.axis(0),
        "inactive": ControlledEncoder.axis(1),
        "pending": ControlledEncoder.axis(2),
        # "aktyve": clearly active (cos 0.90), orthogonal to the others.
        "aktyve": ControlledEncoder.blend(0, 40, 0.90),
    }


# --------------------------------------------------------------------------- #
# Default-path identity (the keystone)
# --------------------------------------------------------------------------- #


def test_default_backends_identical_to_explicit_deterministic(messy):
    out_default, rep_default = fd.clean(messy, semantic_mode="auto", return_report=True)
    out_explicit, rep_explicit = fd.clean(
        messy, semantic_mode="auto", semantic_backends=("deterministic",), return_report=True
    )
    pd.testing.assert_frame_equal(out_default, out_explicit)
    sem_a = [(a.description, a.status, a.confidence) for a in rep_default.actions]
    sem_b = [(a.description, a.status, a.confidence) for a in rep_explicit.actions]
    assert sem_a == sem_b
    assert rep_default.fallback_events == rep_explicit.fallback_events == []


def test_default_plan_decisions_hash_stable(messy):
    config = merge_options(None, semantic_mode="auto")
    hash_a = build_repair_plan(messy, config).decisions_hash()
    hash_b = build_repair_plan(messy, config).decisions_hash()
    assert hash_a == hash_b


# --------------------------------------------------------------------------- #
# Degradation and unknown backends
# --------------------------------------------------------------------------- #


def test_embedding_requested_without_model_degrades(tmp_path, monkeypatch):
    monkeypatch.setenv("FRESHDATA_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("FRESHDATA_STUB_ENCODER", raising=False)
    df = _status_frame()
    out, report = fd.clean(
        df,
        semantic_mode="auto",
        semantic_backends=("deterministic", "embedding"),
        return_report=True,
    )
    assert out is not None
    events = [e for e in report.fallback_events if e["backend"] == "embedding"]
    assert len(events) == 1
    reason = events[0]["fallback_reason"]
    assert "freshdata[semantic]" in reason or "fd.models.pull" in reason
    assert any("embedding" in w and "skipped" in w for w in report.warnings)


def test_unknown_backend_warns_and_continues():
    df = _status_frame()
    out, report = fd.clean(
        df,
        semantic_mode="auto",
        semantic_backends=("deterministic", "quantum"),
        return_report=True,
    )
    assert out is not None
    assert any("unknown semantic backend 'quantum'" in w for w in report.warnings)


def test_unknown_backend_strict_raises():
    df = _status_frame()
    with pytest.raises(fd.PolicyError, match="quantum"):
        fd.clean(
            df,
            semantic_mode="auto",
            semantic_backends=("deterministic", "quantum"),
            strict=True,
        )


def test_memory_in_tuple_without_memory_notes():
    df = _status_frame()
    _, report = fd.clean(
        df,
        semantic_mode="auto",
        semantic_backends=("deterministic", "memory"),
        return_report=True,
    )
    assert any(e["backend"] == "memory" for e in report.fallback_events)


# --------------------------------------------------------------------------- #
# Embedding backend behavior (controlled encoder)
# --------------------------------------------------------------------------- #


def test_embedding_rescues_reference_value_deterministic_missed(encoder_factory):
    encoder_factory(_status_vectors())
    df = _status_frame()
    out, report = fd.clean(_status_frame(), config=_status_config(), return_report=True)
    # "aktyve" is 2 edits from "active" (deterministic tolerance is 1) but the
    # embedding match is unambiguous, so it repairs through the normal gate.
    assert out["status"].tolist()[0] == "active"
    actions = [a for a in report.actions if a.model_id == "semantic:reference_value:embedding"]
    assert len(actions) == 1
    action = actions[0]
    assert action.status == "automatic"
    assert action.metadata["backend"] == "embedding"
    assert action.metadata["model_evidence"]["model_id"] == "fd-col-encoder-v1"
    assert action.metadata["model_evidence"]["model_sha256"] == "test-controlled"
    assert 0 < action.metadata["model_evidence"]["margin"] <= 1
    assert action.metadata["calibration_version"] == "calib-default-1"
    assert action.metadata["raw_score"] >= action.metadata["calibrated_confidence"] > 0.9
    assert df["amount"].equals(out["amount"])


def test_embedding_ambiguous_match_abstains(encoder_factory):
    vectors = _status_vectors()
    # cos to active 0.82, cos to inactive 0.79 -> margin 0.03 < 0.05: ambiguous.
    v = 0.82 * ControlledEncoder.axis(0) + 0.79 * ControlledEncoder.axis(1)
    vectors["aktyve"] = (v / np.linalg.norm(v)).astype(np.float32)
    encoder_factory(vectors)
    _, report = fd.clean(_status_frame(), config=_status_config(), return_report=True)
    assert not [a for a in report.actions if a.model_id.endswith(":embedding")]


def test_embedding_unknown_value_abstains(encoder_factory):
    encoder_factory(_status_vectors())  # "closed" gets its own orthogonal axis
    df = _status_frame()
    df.loc[0, "status"] = "closed"
    _, report = fd.clean(df, config=_status_config(), return_report=True)
    embedding_actions = [a for a in report.actions if a.model_id.endswith(":embedding")]
    assert embedding_actions == []


def test_embedding_never_overrides_deterministic_repair(encoder_factory):
    encoder = encoder_factory(_status_vectors())
    df = _status_frame()
    df.loc[0, "status"] = "activ"  # 1 edit: deterministic ReferenceExpert repairs it
    _, report = fd.clean(df, config=_status_config(), return_report=True)
    det = [a for a in report.actions if a.model_id == "semantic:reference_value:v1"]
    emb = [a for a in report.actions if a.model_id.endswith(":embedding")]
    assert det and not emb
    assert all("activ" not in call for call in encoder.calls)  # residual only


def test_embedding_skips_protected_id_target_and_high_cardinality(encoder_factory):
    encoder = encoder_factory()
    n = 60
    df = pd.DataFrame(
        {
            "cust_id": [f"C{i:04d}" for i in range(n)],
            "revenue": ["1000"] * n,
            "label": ["yes", "no"] * (n // 2),
            "notes": [f"free text row {i} with several words here" for i in range(n)],
            "wide": [f"v{i}" for i in range(n)],
            "status": (["aktyve"] + ["active"] * (n - 1)),
        }
    )
    config = merge_options(
        None,
        semantic_mode="auto",
        semantic_backends=("deterministic", "embedding"),
        target_column="label",
        semantic_max_distinct_values=50,
        semantic_context={
            "columns": {
                "revenue": {"mutable": False},
                "status": {"allowed_values": ["active", "inactive", "pending"]},
            }
        },
    )
    ctx = build_semantic_context(df, config)
    proposals = gather_proposals(df, ctx, config)
    embedding_cols = {p.column for p in proposals if p.backend == "embedding"}
    assert embedding_cols <= {"status"}
    encoded = {text for call in encoder.calls for text in call}
    # No value from protected/id/target/free-text/high-cardinality columns
    # may ever reach the model.
    assert not encoded & {"1000", "yes", "no"}
    assert not any(t.startswith(("C0", "v", "free text")) for t in encoded)


def test_embedding_uses_distinct_values_only(encoder_factory):
    encoder = encoder_factory(_status_vectors())
    df = pd.DataFrame({"status": ["aktyve"] * 500 + ["active"] * 500})
    fd.clean(df, config=_status_config(), return_report=True)
    total_texts = sum(len(call) for call in encoder.calls)
    assert total_texts <= 5  # residual distinct + allowed values, never rows


def test_embedding_clustering_is_suggest_only(encoder_factory):
    vectors = {
        "delhi": ControlledEncoder.axis(5),
        "dehli": ControlledEncoder.blend(5, 41, 0.95),
    }
    encoder_factory(vectors)
    df = pd.DataFrame(
        {"city": ["delhi", "delhi", "delhi", "dehli"], "amount": [1, 2, 3, 4]}
    )
    out, report = fd.clean(
        df,
        semantic_mode="auto",
        semantic_backends=("deterministic", "embedding"),
        return_report=True,
    )
    actions = [a for a in report.actions if a.model_id == "semantic:category_synonym:embedding"]
    assert len(actions) == 1
    action = actions[0]
    assert action.status == "suggested"  # never auto without allowed-values evidence
    assert action.risk in ("medium", "high")
    assert action.confidence < 0.95
    assert out["city"].tolist() == df["city"].tolist()  # unchanged


def test_embedding_cluster_maps_rare_to_modal_only(encoder_factory):
    vectors = {
        "delhi": ControlledEncoder.axis(5),
        "dehli": ControlledEncoder.blend(5, 41, 0.95),
    }
    encoder_factory(vectors)
    df = pd.DataFrame({"city": ["dehli", "dehli", "dehli", "delhi"], "n": [1, 2, 3, 4]})
    _, report = fd.clean(
        df,
        semantic_mode="auto",
        semantic_backends=("deterministic", "embedding"),
        return_report=True,
    )
    actions = [a for a in report.actions if a.model_id == "semantic:category_synonym:embedding"]
    # "delhi" is the rarer spelling here, so it maps toward "dehli", never both ways.
    assert len(actions) == 1
    assert actions[0].metadata["raw_value"] == "delhi"


def test_protected_columns_byte_identical_with_embedding(encoder_factory):
    encoder_factory(_status_vectors())
    df = _status_frame()
    df["revenue"] = ["1,000", "2,000", "3,000", "4,000", "5,000", "6,000"]
    config = _status_config(
        semantic_context={
            "columns": {
                "status": {"allowed_values": ["active", "inactive", "pending"]},
                "revenue": {"mutable": False},
            }
        }
    )
    out, _ = fd.clean(df, config=config, return_report=True)
    assert out["revenue"].equals(df["revenue"])  # byte-identical, guard-verified


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


def test_budget_from_config_parses_fields():
    budget = Budget.from_config(
        {"max_columns": 2, "max_distinct_values": 10, "max_model_calls": 3, "max_seconds": 1.5}
    )
    assert (budget.max_columns, budget.max_distinct_values) == (2, 10)
    assert (budget.max_model_calls, budget.max_seconds) == (3, 1.5)
    assert Budget.from_config(None).max_columns is None
    assert Budget.from_config({"max_columns": -1}).max_columns is None


@pytest.mark.parametrize(
    "budget_dict, expected_reason",
    [
        ({"max_columns": 0}, "max_columns"),
        ({"max_distinct_values": 1}, "max_distinct_values"),
        ({"max_model_calls": 0}, "max_model_calls"),
        ({"max_seconds": 0.0}, "max_seconds"),
    ],
)
def test_budget_exhaustion_stops_cleanly(encoder_factory, budget_dict, expected_reason):
    encoder_factory(_status_vectors())
    df = _status_frame()
    out, report = fd.clean(
        df, config=_status_config(semantic_budget=budget_dict), return_report=True
    )
    assert not [a for a in report.actions if a.model_id.endswith(":embedding")]
    events = [e for e in report.fallback_events if e["backend"] == "embedding"]
    assert events and expected_reason in events[0]["fallback_reason"]
    # Deterministic experts still ran despite the exhausted model budget.
    assert [a for a in report.actions if a.step == "semantic" and a.model_id.endswith(":v1")]


def test_budget_does_not_meter_deterministic():
    df = _status_frame()
    _, report = fd.clean(
        df,
        config=_status_config(
            semantic_backends=("deterministic",), semantic_budget={"max_columns": 0}
        ),
        return_report=True,
    )
    assert [a for a in report.actions if a.step == "semantic"]
    assert report.fallback_events == []  # embedding not requested, nothing skipped
